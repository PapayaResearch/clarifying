import dataclasses
import random
from typing import Any, List, Optional, Type

from pydantic import Field

from clarify.utils.schema import TaskInstance, Turn
from clarify.utils.wrappers import IOSchema, LanguageModel


def format_conversation(request: str, turns: List[Turn]) -> str:
    """Render the transcript the way every episode prompt expects it."""
    lines = [f"User request: {request}"]
    for turn in turns:
        if turn.role == "agent_question":
            lines.append(f"Assistant asked: {turn.text}")
        elif turn.role == "user_answer":
            lines.append(f"User answered: {turn.text}")
    return "\n".join(lines)


def user_answer_input_schema(
    intent_description: str,
    conversation_description: str,
    question_description: str,
) -> Type[IOSchema]:
    class UserAnswerInput(IOSchema):
        """Input schema for the simulated user's answer."""
        intent: str = Field(description=intent_description)
        conversation: str = Field(description=conversation_description)
        question: str = Field(description=question_description)

    return UserAnswerInput


def user_answer_output_schema(answer_description: str) -> Type[IOSchema]:
    class UserAnswerOutput(IOSchema):
        """Output schema for the simulated user's answer."""
        answer: str = Field(description=answer_description)

    return UserAnswerOutput


@dataclasses.dataclass
class PowerPatience:
    """How many questions the user answers before asking for the result: T over 1..cap with P(T <= t) = (t / cap) ** shape."""

    shape: float = 1.5
    name: str = "power"

    def cdf(self, t: int, cap: int) -> float:
        """P(T <= t)."""
        return (min(max(t, 0), cap) / cap) ** self.shape

    def pmf(self, cap: int) -> List[float]:
        """P(T = t) for t = 1..cap."""
        return [self.cdf(t, cap) - self.cdf(t - 1, cap) for t in range(1, cap + 1)]

    def hazard(self, t: int, cap: int) -> float:
        """P(T = t | T >= t): the chance question t is the last one answered, given it was asked."""
        remaining = 1.0 - self.cdf(t - 1, cap)
        return 1.0 if remaining <= 0 else (self.cdf(t, cap) - self.cdf(t - 1, cap)) / remaining

    def sample(self, rng: random.Random, cap: int) -> int:
        """Draw T by inverting the cdf: the smallest t with P(T <= t) >= u."""
        if cap <= 0:
            return 0
        u = rng.random()
        return next((t for t in range(1, cap + 1) if self.cdf(t, cap) >= u), cap)


@dataclasses.dataclass
class UserSimulator:
    """Answers agent questions from the hidden full spec, and may run out of patience before the agent's cap."""

    name: str
    model: LanguageModel
    patience: Optional[PowerPatience] = None

    def budget(self, cap: int, seed: str) -> int:
        """Questions this user answers in an episode capped at `cap`."""
        if self.patience is None:
            return cap
        return self.patience.sample(random.Random(seed), cap)

    def answer(self, instance: TaskInstance, question: str, turns: List[Turn]) -> str:
        response = self.model.get_response(
            intent=instance.full_spec,
            conversation=format_conversation(instance.eroded_prompt, turns),
            question=question,
        )
        if response is None:
            return "Sorry, I'm not sure."
        return response.answer.strip()
