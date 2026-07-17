import dataclasses
from typing import List

from uaug.schema import TaskInstance, Turn
from uaug.utils.wrappers import LanguageModel


def format_conversation(request: str, turns: List[Turn]) -> str:
    """Render the transcript the way every episode prompt expects it."""
    lines = [f"User request: {request}"]
    for turn in turns:
        if turn.role == "agent_question":
            lines.append(f"Assistant asked: {turn.text}")
        elif turn.role == "user_answer":
            lines.append(f"User answered: {turn.text}")
    return "\n".join(lines)


@dataclasses.dataclass
class UserSimulator:
    """Answers agent questions from the hidden full spec, concisely and only what was asked."""

    name: str
    model: LanguageModel

    def answer(self, instance: TaskInstance, question: str, turns: List[Turn]) -> str:
        response = self.model.get_response(
            intent=instance.full_spec,
            conversation=format_conversation(instance.eroded_prompt, turns),
            question=question,
        )
        if response is None:
            return "Sorry, I'm not sure."
        return response.answer.strip()
