"""Reward functions for RL training.

The reward is a swappable module behind the two-method RewardFn protocol, selected
via conf/training/reward/*. Stage B uses terminal rewards only; Stage C adds
per-question shaping from a value ensemble (critics.py).
"""

import dataclasses
from typing import Protocol

from uaug.schema import Episode, JudgeResult


class RewardFn(Protocol):
    def step_reward(self, episode: Episode, t: int) -> float:
        """Shaping reward for the agent's t-th question (0 for terminal-only rewards)."""
        ...

    def final_reward(self, episode: Episode, judgment: JudgeResult) -> float:
        """Reward granted at episode end, after judging."""
        ...


def total_reward(reward_fn: RewardFn, episode: Episode, judgment: JudgeResult) -> float:
    """Undiscounted episode return: all shaping terms plus the terminal reward."""
    shaping = sum(reward_fn.step_reward(episode, t) for t in range(episode.n_questions))
    return shaping + reward_fn.final_reward(episode, judgment)


@dataclasses.dataclass
class TerminalReward:
    """Stage B, hard-budget variant: reward = importance-weighted recovery, questions are free."""

    def step_reward(self, episode: Episode, t: int) -> float:
        return 0.0

    def final_reward(self, episode: Episode, judgment: JudgeResult) -> float:
        return judgment.u_rec


@dataclasses.dataclass
class TerminalCostedReward:
    """Stage B default: reward = u_rec - c * n_questions."""

    question_cost: float

    def step_reward(self, episode: Episode, t: int) -> float:
        return -self.question_cost

    def final_reward(self, episode: Episode, judgment: JudgeResult) -> float:
        return judgment.u_rec
