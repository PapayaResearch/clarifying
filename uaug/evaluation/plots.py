import os
import statistics
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uaug.utils.schema import Episode, JudgeResult


def write_plots(results_dir: str, episodes: List[Episode], judgments: List[JudgeResult]) -> None:
    if not judgments:
        return

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
        ax.set_xlabel("questions asked")
        ax.set_ylabel(f"mean u_rec ({weighting})")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"u_rec_by_questions_{weighting}.png"), dpi=150)
        plt.close(fig)
