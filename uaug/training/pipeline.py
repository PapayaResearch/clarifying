"""Stage B training loop: sample instances -> group rollouts -> judge -> reward ->
trainer update, iterated. Backend- and reward-agnostic; both are injected config
groups (conf/training/trainer/*, conf/training/reward/*)."""

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

from uaug.judging.judge import RubricJudge
from uaug.schema import TaskInstance
from uaug.episodes.user import UserSimulator
from uaug.training.rollouts import Trajectory, assign_group_advantages, collect_trajectory
from uaug.utils.misc import read_jsonl


@dataclasses.dataclass
class TrainingPipeline:
    """Trains a policy over synthesized instances with grouped rollouts."""

    instances_dir: Optional[str]
    split: str
    budget_override: Optional[int]
    n_iterations: int
    instances_per_iteration: int
    group_size: int
    save_every: int
    normalize_advantages: bool
    policy_prompt: str
    random_seed: int
    reward: Any
    trainer: Any
    user: UserSimulator
    judge: RubricJudge

    def _load_instances(self) -> List[TaskInstance]:
        if not self.instances_dir:
            raise ValueError("training.instances_dir must point to a synthesis results directory (use the -pruned one)")
        path = os.path.join(self.instances_dir, f"instances.{self.split}.jsonl")
        instances = [TaskInstance.from_dict(row) for row in read_jsonl(path)]
        logging.info("Loaded %d %s instances from %s", len(instances), self.split, path)
        return instances

    def _collect_batch(self, batch: List[TaskInstance], policy, iteration: int, max_workers: int) -> List[Trajectory]:
        jobs = [(instance, iteration * self.group_size + g) for instance in batch for g in range(self.group_size)]
        trajectories: List[Trajectory] = []
        lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    collect_trajectory, instance, policy, self.user, self.judge,
                    self.reward, self.policy_prompt, seed, self.budget_override,
                ): instance
                for instance, seed in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"rollouts[iter {iteration}]"):
                try:
                    trajectory = future.result()
                except Exception:
                    logging.exception("Rollout failed on %s", futures[future].instance_id)
                    continue
                with lock:
                    trajectories.append(trajectory)
        assign_group_advantages(trajectories, normalize=self.normalize_advantages)
        return trajectories

    def run(self, results_dir: str, max_workers: int = 8):
        instances = self._load_instances()
        rng = random.Random(self.random_seed)
        log_path = os.path.join(results_dir, "training_log.jsonl")

        for iteration in range(self.n_iterations):
            policy = self.trainer.make_policy()  # refresh sampler to the latest weights
            batch = rng.sample(instances, min(self.instances_per_iteration, len(instances)))
            trajectories = self._collect_batch(batch, policy, iteration, max_workers)
            if not trajectories:
                logging.error("Iteration %d produced no trajectories; stopping", iteration)
                break
            metrics = self.trainer.update(trajectories)

            row = {
                "iteration": iteration,
                "n_trajectories": len(trajectories),
                "reward_mean": round(statistics.mean(t.reward for t in trajectories), 4),
                "u_rec_mean": round(statistics.mean(t.judgment.u_rec for t in trajectories), 4),
                "n_questions_mean": round(statistics.mean(t.episode.n_questions for t in trajectories), 2),
                **metrics,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            logging.info("Training iteration: %s", json.dumps(row))
            print(json.dumps(row))

            if (iteration + 1) % self.save_every == 0 or iteration == self.n_iterations - 1:
                self.trainer.save(results_dir, iteration)
