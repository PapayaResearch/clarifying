import dataclasses
from typing import Callable, Dict, List, Tuple

from clarify.utils.schema import Completion, Turn
from clarify.utils.wrappers import ChatModel
from clarify.utils.user import format_conversation


def build_transcript_messages(system_prompt: str, request: str, turns: List[Turn]) -> List[Dict[str, str]]:
    """The decide and final calls as a rendered transcript: the whole conversation arrives in one user message."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_conversation(request, turns)},
    ]


def build_chat_messages(system_prompt: str, request: str, turns: List[Turn]) -> List[Dict[str, str]]:
    """The ask call as a native chat: past questions are assistant turns."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]
    for turn in turns:
        role = "assistant" if turn.role == "agent_question" else "user"
        messages.append({"role": role, "content": turn.text})
    return messages


def parse_decision(text: str) -> bool:
    """True when the model wants one more question."""
    return text.strip().lower().startswith("yes")


@dataclasses.dataclass
class Asker:
    """The conversation model asking clarifying questions."""

    name: str
    mode: str  # "fixed" | "dynamic"
    model: ChatModel
    ask_prompt: str
    final_prompt: str
    decide_prompt: str
    k_multiplier: int

    def cap(self, n_redacted: int) -> int:
        """The episode's question cap before the user's patience is applied: k_multiplier times the redacted count."""
        return self.k_multiplier * n_redacted

    def run(
        self,
        request: str,
        answer: Callable[[str, List[Turn]], str],
        max_questions: int,
    ) -> Tuple[List[Turn], Completion, int]:
        """One episode within a question budget: returns (question/answer turns, final completion, questions asked)."""
        if self.mode not in ("fixed", "dynamic"):
            raise ValueError(f"Unknown agent mode: {self.mode!r} (expected 'fixed' or 'dynamic')")

        turns: List[Turn] = []
        n = 0
        while self._wants_question(n, max_questions, request, turns):
            asked = self.model.complete(build_chat_messages(self.ask_prompt, request, turns))
            reply = answer(asked.text, turns)
            turns.append(Turn(role="agent_question", text=asked.text, reasoning=asked.reasoning))
            turns.append(Turn(role="user_answer", text=reply))
            n += 1
        final = self.model.complete(build_transcript_messages(self.final_prompt, request, turns))
        return turns, final, n

    def _wants_question(self, asked: int, max_questions: int, request: str, turns: List[Turn]) -> bool:
        if asked >= max_questions:
            return False
        if self.mode == "fixed":
            return True
        decision = self.model.complete(build_transcript_messages(self.decide_prompt, request, turns))
        return parse_decision(decision.text)
