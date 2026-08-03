"""Backend-blind rollout collection: policy -> episode -> judgment -> reward.

Mirrors EpisodePipeline._run_episode but records, for every agent turn, the exact
messages the policy saw and the Completion it returned (with backend meta), which
is what trainers consume.
"""

import dataclasses
import statistics
from typing import Dict, List, Optional

from uaug.judging.judge import RubricJudge
from uaug.schema import Episode, JudgeResult, TaskInstance, Turn
from uaug.episodes.user import UserSimulator
from uaug.training.policy import Completion, Policy, build_messages, parse_action
from uaug.training.rewards import RewardFn, total_reward


@dataclasses.dataclass
class AgentTurn:
    """One policy call: the prompt messages at sampling time and the sampled completion."""
    messages: List[Dict[str, str]]
    completion: Completion


@dataclasses.dataclass
class Trajectory:
    instance_id: str
    seed: int
    episode: Episode
    judgment: JudgeResult
    agent_turns: List[AgentTurn]
    reward: float
    advantage: float = 0.0


def collect_trajectory(
    instance: TaskInstance,
    policy: Policy,
    user: UserSimulator,
    judge: RubricJudge,
    reward_fn: RewardFn,
    policy_prompt: str,
    seed: int,
    budget_override: Optional[int] = None,
) -> Trajectory:
    budget = budget_override if budget_override is not None else instance.budget
    turns: List[Turn] = []
    agent_turns: List[AgentTurn] = []
    n_questions = 0

    while True:
        messages = build_messages(policy_prompt, instance.eroded_prompt, turns, budget - n_questions)
        completion = policy.complete(messages, seed=seed)
        agent_turns.append(AgentTurn(messages=messages, completion=completion))
        kind, text = parse_action(completion.text)
        if kind == "ask" and n_questions < budget:
            answer = user.answer(instance, text, turns)
            turns.append(Turn(role="agent_question", text=text))
            turns.append(Turn(role="user_answer", text=answer))
            n_questions += 1
        else:
            turns.append(Turn(role="agent_final", text=text))
            break

    episode = Episode(
        instance_id=instance.instance_id,
        agent_name="policy",
        turns=turns,
        final_output=text,
        n_questions=n_questions,
        seed=seed,
    )
    judgment = judge.judge(instance, episode)
    reward = total_reward(reward_fn, episode, judgment)
    return Trajectory(
        instance_id=instance.instance_id,
        seed=seed,
        episode=episode,
        judgment=judgment,
        agent_turns=agent_turns,
        reward=reward,
    )


def assign_group_advantages(trajectories: List[Trajectory], normalize: bool = False) -> None:
    """GRPO-style advantages: center rewards within each instance's rollout group.

    With normalize=True also divide by the group's reward std (skipped for
    degenerate groups). Mutates trajectories in place.
    """
    groups: Dict[str, List[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.instance_id, []).append(trajectory)
    for group in groups.values():
        mean = statistics.mean(t.reward for t in group)
        std = statistics.pstdev(t.reward for t in group) if len(group) > 1 else 0.0
        for trajectory in group:
            advantage = trajectory.reward - mean
            if normalize and std > 1e-8:
                advantage /= std
            trajectory.advantage = advantage
