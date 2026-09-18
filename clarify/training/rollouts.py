import statistics
from typing import Dict, List

from clarify.utils.judge import RubricJudge
from clarify.utils.schema import AgentTurn, Completion, Episode, TaskInstance, Trajectory, Turn
from clarify.utils.agents import build_chat_messages, build_transcript_messages, parse_decision
from clarify.utils.tinker import TinkerModel
from clarify.utils.user import UserSimulator
from clarify.training.rewards import RewardFn, total_reward


def collect_trajectory(
    instance: TaskInstance,
    model: TinkerModel,
    user: UserSimulator,
    judge: RubricJudge,
    reward_fn: RewardFn,
    mode: str,
    max_questions: int,
    importance_weighting: str,
    ask_prompt: str,
    final_prompt: str,
    decide_prompt: str,
) -> Trajectory:
    if mode not in ("fixed", "dynamic"):
        raise ValueError(f"Unknown rollout mode: {mode!r} (expected 'fixed' or 'dynamic')")
    if importance_weighting not in ("llm", "equal"):
        raise ValueError(f"Unknown importance weighting: {importance_weighting!r} (expected 'llm' or 'equal')")

    turns: List[Turn] = []
    agent_turns: List[AgentTurn] = []
    request = instance.eroded_prompt

    def sample(messages: List[Dict[str, str]]) -> Completion:
        completion = model.complete(messages)
        agent_turns.append(AgentTurn(messages=messages, completion=completion))
        return completion

    def wants_question(asked: int) -> bool:
        if asked >= max_questions:
            return False
        if mode == "fixed":
            return True
        return parse_decision(sample(build_transcript_messages(decide_prompt, request, turns)).text)

    n = 0
    while wants_question(n):
        asked = sample(build_chat_messages(ask_prompt, request, turns))
        answer = user.answer(instance, asked.text, turns)
        turns.append(Turn(role="agent_question", text=asked.text, reasoning=asked.reasoning))
        turns.append(Turn(role="user_answer", text=answer))
        n += 1
    final = sample(build_transcript_messages(final_prompt, request, turns))

    episode = Episode(
        instance_id=instance.instance_id,
        agent_name="training",
        turns=turns + [Turn(role="agent_final", text=final.text, reasoning=final.reasoning)],
        final_output=final.text,
        n_questions=n,
        budget=max_questions,
    )
    judgment = judge.judge(instance, episode)
    reward = total_reward(reward_fn, episode, judgment, importance_weighting)
    return Trajectory(
        instance_id=instance.instance_id,
        episode=episode,
        judgment=judgment,
        agent_turns=agent_turns,
        reward=reward,
    )


def assign_group_advantages(trajectories: List[Trajectory]) -> None:
    """GRPO-style advantages: center rewards within each instance's rollout group."""
    groups: Dict[str, List[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.instance_id, []).append(trajectory)
    for group in groups.values():
        mean = statistics.mean(t.reward for t in group)
        for trajectory in group:
            trajectory.advantage = trajectory.reward - mean
