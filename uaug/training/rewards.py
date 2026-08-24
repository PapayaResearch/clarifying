import abc
import dataclasses

from uaug.utils.schema import Episode, JudgeResult


class RewardFn(abc.ABC):
    def step_reward(self, episode: Episode, t: int) -> float:
        """Shaping reward for the agent's t-th question (default: none)."""
        return 0.0

    @abc.abstractmethod
    def final_reward(self, episode: Episode, judgment: JudgeResult, weighting: str) -> float:
        """Reward granted at episode end, after judging, under the chosen importance weighting."""


def total_reward(reward_fn: RewardFn, episode: Episode, judgment: JudgeResult, weighting: str) -> float:
    """Undiscounted episode return: all shaping terms plus the terminal reward."""
    shaping = sum(reward_fn.step_reward(episode, t) for t in range(episode.n_questions))
    return shaping + reward_fn.final_reward(episode, judgment, weighting)


@dataclasses.dataclass
class TerminalReward(RewardFn):
    """Recovery only: reward = importance-weighted recovery, questions are free."""

    name: str = "terminal"

    def final_reward(self, episode: Episode, judgment: JudgeResult, weighting: str) -> float:
        return judgment.u_rec[weighting]


@dataclasses.dataclass
class ConstantPenaltyReward(RewardFn):
    """Recovery minus a flat question cost: reward = u_rec - c * n_questions."""

    question_cost: float
    name: str = "constant_penalty"

    def step_reward(self, episode: Episode, t: int) -> float:
        return -self.question_cost

    def final_reward(self, episode: Episode, judgment: JudgeResult, weighting: str) -> float:
        return judgment.u_rec[weighting]


@dataclasses.dataclass
class ExponentialPenaltyReward(RewardFn):
    """Escalating additive cost: question t costs c * g^t, so reward = u_rec - c * (g^n - 1) / (g - 1)."""

    question_cost: float
    growth: float
    name: str = "exponential_penalty"

    def step_reward(self, episode: Episode, t: int) -> float:
        return -self.question_cost * self.growth ** t

    def final_reward(self, episode: Episode, judgment: JudgeResult, weighting: str) -> float:
        return judgment.u_rec[weighting]


@dataclasses.dataclass
class DiscountedPenaltyReward(RewardFn):
    """Multiplicative discount per question: reward = gamma^n * u_rec."""

    gamma: float
    name: str = "discounted_penalty"

    def final_reward(self, episode: Episode, judgment: JudgeResult, weighting: str) -> float:
        return self.gamma ** episode.n_questions * judgment.u_rec[weighting]
