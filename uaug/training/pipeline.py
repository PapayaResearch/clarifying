import dataclasses
import json
import logging
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, List, Optional

from tqdm import tqdm

from uaug.utils.judge import RubricJudge
from uaug.utils.misc import read_jsonl
from uaug.utils.schema import TaskInstance, Trajectory
from uaug.utils.user import UserSimulator
from uaug.utils.agents import question_cap
from uaug.training.rollouts import assign_group_advantages, collect_trajectory
from uaug.training.tinker import TinkerTrainer


@dataclasses.dataclass
class TrainingPipeline:
    """Trains the policy over synthesized instances with grouped rollouts."""

    instances_dir: str
    split: str
    mode: str
    k_multiplier: Optional[int]
    max_questions: Optional[int]
    importance_weighting: str
    resume_dir: Optional[str]
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

    def _budget(self, instance: TaskInstance, iteration: int) -> int:
        """The instance's question cap, cut to what the user answers. Drawn once per group, so every rollout of the
        instance shares the budget and the luck of the draw stays out of the group-centred advantage."""
        cap = question_cap(len(instance.redacted_ids), self.k_multiplier, self.max_questions)
        return self.user.budget(cap, seed=f"{self.random_seed}:{iteration}:{instance.instance_id}")

    def _collect_batch(self, batch: List[TaskInstance], model, iteration: int, max_workers: int) -> List[Trajectory]:
        budgets = {instance.instance_id: self._budget(instance, iteration) for instance in batch}
        jobs = [instance for instance in batch for _ in range(self.group_size)]
        trajectories: List[Trajectory] = []
        lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    collect_trajectory, instance, model, self.user, self.judge,
                    self.reward, self.mode, budgets[instance.instance_id], self.importance_weighting,
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

        start_iteration = 0
        if self.resume_dir:
            with open(os.path.join(self.resume_dir, "checkpoints.jsonl"), encoding="utf-8") as f:
                last = json.loads(f.readlines()[-1])
            self.trainer.load_state(last["state"])
            start_iteration = last["step"] + 1
            logging.info("Resuming from step %d (%s)", last["step"], last["state"])
            if start_iteration >= self.n_iterations:
                logging.error("Run already has %d iterations, raise training.n_iterations to continue", start_iteration)
                return
        # On resume, replay the sampling schedule so instance batches continue where they left off
        for _ in range(start_iteration):
            rng.sample(instances, min(self.instances_per_iteration, len(instances)))
        for iteration in range(start_iteration, self.n_iterations):
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

            completions = [turn.completion for t in trajectories for turn in t.agent_turns]
            row = {
                "iteration": iteration,
                "n_trajectories": len(trajectories),
                "reward_mean": round(statistics.mean(t.reward for t in trajectories), 4),
                "u_rec_llm_mean": round(statistics.mean(t.judgment.u_rec["llm"] for t in trajectories), 4),
                "u_rec_equal_mean": round(statistics.mean(t.judgment.u_rec["equal"] for t in trajectories), 4),
                "n_questions_mean": round(statistics.mean(t.episode.n_questions for t in trajectories), 2),
                "budget_mean": round(statistics.mean(t.episode.budget for t in trajectories), 2),
                "sampled_tokens_mean": round(statistics.mean(c.n_tokens for c in completions), 1),
                "truncation_rate": round(statistics.mean(c.truncated for c in completions), 4),
                **metrics,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            logging.info("Training iteration: %s", json.dumps(row))
            print(json.dumps(row))

            if (iteration + 1) % self.save_every == 0 or iteration == self.n_iterations - 1:
                self.trainer.save(results_dir, iteration)
