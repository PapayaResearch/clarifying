import os
import statistics
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uaug.utils.misc import clairvoyant_u_rec
from uaug.utils.schema import Episode, JudgeResult, TaskInstance


def write_plots(results_dir: str, episodes: List[Episode], judgments: List[JudgeResult],
                instances: List[TaskInstance] = None) -> None:
    if not judgments:
        return
    instance_by_id = {i.instance_id: i for i in instances}

    for weighting in ("llm", "equal"):
        # Distribution of per-episode recovery, with the mean marked
        u_rec_values = [j.u_rec[weighting] for j in judgments]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(u_rec_values, bins=20, range=(0, 1), edgecolor="white")
        ax.axvline(statistics.mean(u_rec_values), linestyle="--", linewidth=1, label=f"mean {statistics.mean(u_rec_values):.3f}")
        ax.set_xlabel(f"u_rec ({weighting})")
        ax.set_ylabel("episodes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"u_rec_hist_{weighting}.png"), dpi=150)
        plt.close(fig)

        # Mean recovery grouped by questions actually asked (per-episode count)
        by_questions = {}
        for episode, judgment in zip(episodes, judgments):
            by_questions.setdefault(episode.n_questions, []).append(judgment.u_rec[weighting])
        counts = sorted(by_questions)
        means = [statistics.mean(by_questions[n]) for n in counts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(n) for n in counts], means)
        for i, n in enumerate(counts):
            ax.annotate(f"n={len(by_questions[n])}", (i, means[i]), ha="center", va="bottom", fontsize=8)
        # Optimal recovery at each question count: the clairvoyant ceiling per bucket
        clair = {}
        for episode in episodes:
            importances = [d.importance for d in instance_by_id[episode.instance_id].redacted]
            clair.setdefault(episode.n_questions, []).append(
                clairvoyant_u_rec(importances, episode.n_questions, weighting))
        ax.plot(range(len(counts)), [statistics.mean(clair[n]) for n in counts], linestyle="none",
                marker="_", markersize=22, markeredgewidth=2, color="#3c4043",
                label="clairvoyant (optimal at count)")
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel("questions asked")
        ax.set_ylabel(f"mean u_rec ({weighting})")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"u_rec_by_questions_{weighting}.png"), dpi=150)
        plt.close(fig)
