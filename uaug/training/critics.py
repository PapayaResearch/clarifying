"""Stage C: value ensemble for the information-gain shaping reward.

r_info at question t = lambda * (V(s_{t+1}) - V(s_t)) where V predicts the final
u_rec achievable from a transcript state; an ensemble's mean is the estimate and
its spread flags unreliable shaping. Snapshot the ensemble per rollout batch so
the shaping reward is stationary within a batch.

Skeleton only until Stage B lands. v0 will be K prompted scorers on a cheap cached
model; upgrade path is K LoRA reward heads trained on Stage B episode data.
"""

import dataclasses
from typing import Any, List

from uaug.schema import Turn


@dataclasses.dataclass
class ValueEnsemble:
    """K predictors of final u_rec from (request, transcript-so-far)."""

    scorers: List[Any]  # K LanguageModel scorers (prompted v0) or reward heads

    def predict(self, request: str, turns: List[Turn]) -> float:
        raise NotImplementedError("Stage C: implement after Stage B baselines exist")


@dataclasses.dataclass
class HybridInfoReward:
    """TerminalCosted plus per-question value-difference shaping from the ensemble."""

    question_cost: float
    shaping_weight: float
    critic: ValueEnsemble

    def step_reward(self, episode, t: int) -> float:
        raise NotImplementedError("Stage C: implement after Stage B baselines exist")

    def final_reward(self, episode, judgment) -> float:
        return judgment.u_rec
