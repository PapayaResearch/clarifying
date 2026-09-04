import dataclasses
import random
from typing import Any, List, Optional, Type

from pydantic import Field

from uaug.utils.schema import TaskInstance, Turn
from uaug.utils.wrappers import IOSchema, LanguageModel


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
class UserSimulator:
    """Answers agent questions from the hidden full spec, and may run out of patience before the agent's cap."""

    name: str
    model: LanguageModel
    patience: Optional[Any] = None  # a stopping rule with sample(rng, cap), None answers every question up to the cap

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
