import csv
import dataclasses
import json
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Tuple

from tqdm import tqdm

from uaug.utils.judge import RubricJudge
from uaug.utils.misc import read_jsonl, write_jsonl
from uaug.utils.schema import Episode, JudgeResult, TaskInstance, Turn
from uaug.evaluation.agents import Asker
from uaug.evaluation.plots import write_plots
from uaug.evaluation.user import UserSimulator


@dataclasses.dataclass
class EvaluationPipeline:
    """Runs one agent over synthesized instances, judges every episode, and aggregates."""

    instances_dir: str
    split: str
    agent: Asker
    user: UserSimulator
    judge: RubricJudge

    def _load_instances(self) -> List[TaskInstance]:
        path = os.path.join(self.instances_dir, f"instances.{self.split}.jsonl")
        instances = [TaskInstance.from_dict(row) for row in read_jsonl(path)]
        logging.info("Loaded %d %s instances from %s", len(instances), self.split, path)
        return instances

    def _run_episode(self, instance: TaskInstance) -> Tuple[Episode, JudgeResult]:
        turns, final_output, n_questions = self.agent.run(
            request=instance.eroded_prompt,
            answer=lambda question, turns: self.user.answer(instance, question, turns),
        )
        episode = Episode(
            instance_id=instance.instance_id,
            agent_name=self.agent.name,
            turns=turns + [Turn(role="agent_final", text=final_output)],
            final_output=final_output,
            n_questions=n_questions,
        )
        return episode, self.judge.judge(instance, episode)

    def run(self, results_dir: str, max_workers: int = 8):
        instances = self._load_instances()

        episodes: List[Episode] = []
        judgments: List[JudgeResult] = []
        lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._run_episode, inst): inst for inst in instances}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"evaluation[{self.agent.name}]"):
                instance = futures[future]
                try:
                    episode, judgment = future.result()
                except Exception:
                    logging.exception("Episode failed on %s", instance.instance_id)
                    continue
                with lock:
                    episodes.append(episode)
                    judgments.append(judgment)

        write_jsonl(os.path.join(results_dir, "episodes.jsonl"), episodes)
        write_jsonl(os.path.join(results_dir, "judgments.jsonl"), judgments)

        k_by_instance = {i.instance_id: len(i.redacted_ids) for i in instances}

        summary = {
            "agent": self.agent.name,
            "mode": self.agent.mode,
            "split": self.split,
            "n_episodes": len(episodes),
            # Every judged metric under both weightings, e.g. u_rec_llm_mean and u_rec_equal_mean
            **{
                f"{metric}_{weighting}_mean": round(statistics.mean(getattr(j, metric)[weighting] for j in judgments), 3) if judgments else None
                for metric in ("u_rec", "eps_c_hat", "eps_e_hat")
                for weighting in ("llm", "equal")
            },
            "n_questions_mean": round(statistics.mean(e.n_questions for e in episodes), 2) if episodes else None,
            "planning_gap_mean": round(statistics.mean(e.n_questions - k_by_instance[e.instance_id] for e in episodes), 2) if episodes else None,
        }
        with open(os.path.join(results_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        logging.info("Evaluation summary: %s", json.dumps(summary))
        print(json.dumps(summary, indent=2))

        write_plots(results_dir, episodes, judgments)
