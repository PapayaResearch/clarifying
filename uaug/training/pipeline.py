import dataclasses
import json
import logging
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, List

from tqdm import tqdm

from uaug.utils.judge import RubricJudge
from uaug.utils.misc import read_jsonl
from uaug.utils.schema import TaskInstance, Trajectory
from uaug.evaluation.user import UserSimulator
from uaug.training.rollouts import assign_group_advantages, collect_trajectory
from uaug.training.tinker import TinkerTrainer


@dataclasses.dataclass
class TrainingPipeline:
    """Trains the policy over synthesized instances with grouped rollouts."""

    instances_dir: str
    split: str
    mode: str
    max_questions: int
    importance_weighting: str
    n_iterations: int
    instances_per_iteration: int
    group_size: int
    save_every: int
    ask_prompt: str
    final_prompt: str
    decide_prompt: str
    random_seed: int
    reward: Any
    trainer: TinkerTrainer
    user: UserSimulator
    judge: RubricJudge

    def _load_instances(self) -> List[TaskInstance]:
        path = os.path.join(self.instances_dir, f"instances.{self.split}.jsonl")
        instances = [TaskInstance.from_dict(row) for row in read_jsonl(path)]
        logging.info("Loaded %d %s instances from %s", len(instances), self.split, path)
        return instances

    def _collect_batch(self, batch: List[TaskInstance], model, iteration: int, max_workers: int) -> List[Trajectory]:
        jobs = [instance for instance in batch for _ in range(self.group_size)]
        trajectories: List[Trajectory] = []
        lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    collect_trajectory, instance, model, self.user, self.judge,
                    self.reward, self.mode, self.max_questions, self.importance_weighting,
                    self.ask_prompt, self.final_prompt, self.decide_prompt,
                ): instance
                for instance in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"rollouts[iter {iteration}]"):
                try:
                    trajectory = future.result()
                except Exception:
                    logging.exception("Rollout failed on %s", futures[future].instance_id)
                    continue
                with lock:
                    trajectories.append(trajectory)
        assign_group_advantages(trajectories)
        return trajectories

    def run(self, results_dir: str, max_workers: int = 8):
        instances = self._load_instances()
        rng = random.Random(self.random_seed)
        log_path = os.path.join(results_dir, "training_log.jsonl")
        trajectories_path = os.path.join(results_dir, "trajectories.jsonl")

        for iteration in range(self.n_iterations):
            model = self.trainer.make_model()  # refresh sampler to the latest weights
            batch = rng.sample(instances, min(self.instances_per_iteration, len(instances)))
            trajectories = self._collect_batch(batch, model, iteration, max_workers)
            if not trajectories:
                logging.error("Iteration %d produced no trajectories; stopping", iteration)
                break
            # The training data itself: every judged rollout's transcript and scores
            with open(trajectories_path, "a", encoding="utf-8") as f:
                for t in trajectories:
                    f.write(json.dumps({
                        "iteration": iteration,
                        "reward": round(t.reward, 4),
                        "advantage": round(t.advantage, 4),
                        "judgment": dataclasses.asdict(t.judgment),
                        "episode": dataclasses.asdict(t.episode),
                    }) + "\n")
            metrics = self.trainer.update(trajectories)

            row = {
                "iteration": iteration,
                "n_trajectories": len(trajectories),
                "reward_mean": round(statistics.mean(t.reward for t in trajectories), 4),
                "u_rec_llm_mean": round(statistics.mean(t.judgment.u_rec["llm"] for t in trajectories), 4),
                "u_rec_equal_mean": round(statistics.mean(t.judgment.u_rec["equal"] for t in trajectories), 4),
                "n_questions_mean": round(statistics.mean(t.episode.n_questions for t in trajectories), 2),
                **metrics,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            logging.info("Training iteration: %s", json.dumps(row))
            print(json.dumps(row))

            if (iteration + 1) % self.save_every == 0 or iteration == self.n_iterations - 1:
                self.trainer.save(results_dir, iteration)
