import dataclasses
import json
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from uaug.judging.judge import RubricJudge
from uaug.schema import Episode, JudgeResult, TaskInstance, Turn
from uaug.report.episodes import write_episodes_report
from uaug.episodes.user import UserSimulator
from uaug.utils.misc import bootstrap_ci, read_jsonl, write_jsonl


@dataclasses.dataclass
class EpisodePipeline:
    """Runs one agent over synthesized instances, judges every episode, and aggregates."""

    instances_dir: Optional[str]
    split: str
    seeds: List[int]
    budget_override: Optional[int]
    agent: Any
    user: UserSimulator
    judge: RubricJudge

    def _load_instances(self) -> List[TaskInstance]:
        if not self.instances_dir:
            raise ValueError("episodes.instances_dir must point to a synthesis results directory")
        path = os.path.join(self.instances_dir, f"instances.{self.split}.jsonl")
        instances = [TaskInstance.from_dict(row) for row in read_jsonl(path)]
        logging.info("Loaded %d %s instances from %s", len(instances), self.split, path)
        return instances

    def _run_episode(self, instance: TaskInstance, seed: int) -> Tuple[Episode, JudgeResult]:
        budget = self.budget_override if self.budget_override is not None else instance.budget
        turns: List[Turn] = []
        n_questions = 0

        while True:
            kind, text = self.agent.act(instance.eroded_prompt, turns, budget - n_questions)
            if kind == "ask" and n_questions < budget:
                answer = self.user.answer(instance, text, turns)
                turns.append(Turn(role="agent_question", text=text))
                turns.append(Turn(role="user_answer", text=answer))
                n_questions += 1
            else:
                final_output = text if kind == "finish" else self.agent.finalize(instance.eroded_prompt, turns)
                turns.append(Turn(role="agent_final", text=final_output))
                break

        episode = Episode(
            instance_id=instance.instance_id,
            agent_name=self.agent.name,
            turns=turns,
            final_output=final_output,
            n_questions=n_questions,
            seed=seed,
        )
        return episode, self.judge.judge(instance, episode)

    def run(self, results_dir: str, max_workers: int = 8):
        instances = self._load_instances()
        jobs = [(instance, seed) for instance in instances for seed in self.seeds]

        episodes: List[Episode] = []
        judgments: List[JudgeResult] = []
        lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._run_episode, inst, seed): (inst, seed) for inst, seed in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"episodes[{self.agent.name}]"):
                instance, seed = futures[future]
                try:
                    episode, judgment = future.result()
                except Exception:
                    logging.exception("Episode failed on %s (seed %d)", instance.instance_id, seed)
                    continue
                with lock:
                    episodes.append(episode)
                    judgments.append(judgment)

        write_jsonl(os.path.join(results_dir, f"episodes.{self.agent.name}.jsonl"), [e.to_dict() for e in episodes])
        write_jsonl(os.path.join(results_dir, f"judgments.{self.agent.name}.jsonl"), [j.to_dict() for j in judgments])

        preserved_values = [j.preserved for j in judgments if j.preserved is not None]
        k_by_instance = {i.instance_id: len(i.redacted_ids) for i in instances}
        u_rec_values = [j.u_rec for j in judgments]

        # u_rec grouped by questions actually used (v0 elicitation curve; note the
        # grouping is by the agent's own stopping choice, not a controlled budget)
        by_questions = {}
        for episode, judgment in zip(episodes, judgments):
            by_questions.setdefault(episode.n_questions, []).append(judgment.u_rec)
        u_rec_by_questions = {
            n: {"n_episodes": len(vals), "u_rec_mean": round(statistics.mean(vals), 3)}
            for n, vals in sorted(by_questions.items())
        }

        report = {
            "agent": self.agent.name,
            "user": self.user.name,
            "split": self.split,
            "n_episodes": len(episodes),
            "u_rec_mean": round(statistics.mean(u_rec_values), 3) if judgments else None,
            "u_rec_ci95": bootstrap_ci(u_rec_values),
            "eps_c_hat_mean": round(statistics.mean(j.eps_c_hat for j in judgments), 3) if judgments else None,
            "eps_e_hat_mean": round(statistics.mean(j.eps_e_hat for j in judgments), 3) if judgments else None,
            "preserved_mean": round(statistics.mean(preserved_values), 3) if preserved_values else None,
            "n_questions_mean": round(statistics.mean(e.n_questions for e in episodes), 2) if episodes else None,
            # Planning gap vs. k, the number of redacted details (read alongside
            # u_rec - asking 0 questions has gap -k but recovers nothing; some
            # redacted details may be prior-inferable, see ZeroQ delivery rates)
            "k_mean": round(statistics.mean(k_by_instance[e.instance_id] for e in episodes), 2) if episodes else None,
            "planning_gap_mean": round(statistics.mean(e.n_questions - k_by_instance[e.instance_id] for e in episodes), 2) if episodes else None,
            "full_recovery_rate": round(sum(1 for j in judgments if j.u_rec > 0.999) / len(judgments), 3) if judgments else None,
            "u_rec_by_questions": u_rec_by_questions,
        }
        with open(os.path.join(results_dir, f"report.{self.agent.name}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logging.info("Episode report: %s", json.dumps(report))
        print(json.dumps(report, indent=2))

        report_path = os.path.join(results_dir, f"report.{self.agent.name}.html")
        write_episodes_report(report_path, instances, episodes, judgments, report)
        logging.info("HTML report: %s", report_path)
