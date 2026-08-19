import abc
import dataclasses

from uaug.utils.schema import Episode, JudgeResult


class RewardFn(abc.ABC):
    def step_reward(self, episode: Episode, t: int) -> float:
        """Shaping reward for the agent's t-th question (default: none)."""
        return 0.0

    @abc.abstractmethod
    def final_reward(self, episode: Episode, judgment: JudgeResult) -> float:
        """Reward granted at episode end, after judging."""


def total_reward(reward_fn: RewardFn, episode: Episode, judgment: JudgeResult) -> float:
    """Undiscounted episode return: all shaping terms plus the terminal reward."""
    shaping = sum(reward_fn.step_reward(episode, t) for t in range(episode.n_questions))
    return shaping + reward_fn.final_reward(episode, judgment)


@dataclasses.dataclass
class TerminalReward(RewardFn):
    """Recovery only: reward = importance-weighted recovery, questions are free."""

    name: str = "terminal"

    def final_reward(self, episode: Episode, judgment: JudgeResult) -> float:
        return judgment.u_rec


@dataclasses.dataclass
class TerminalCostedReward(RewardFn):
    """Recovery minus question cost: reward = u_rec - c * n_questions."""

    question_cost: float
    name: str = "terminal_costed"

    def step_reward(self, episode: Episode, t: int) -> float:
        return -self.question_cost

    def final_reward(self, episode: Episode, judgment: JudgeResult) -> float:
        return judgment.u_rec
