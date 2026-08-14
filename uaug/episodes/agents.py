import dataclasses
import random
from typing import Any, List, Optional, Tuple

from uaug.schema import Turn
from uaug.episodes.user import format_conversation
from uaug.training.policy import build_messages, parse_action
from uaug.utils.wrappers import LanguageModel

# An agent action: ("ask", question) or ("finish", final_description)
Action = Tuple[str, str]


@dataclasses.dataclass
class ZeroQAgent:
    """Budget-zero baseline: composes the final description immediately, filling gaps from the prior.

    Doubles as the ground-truth inferability measurement: any redacted detail this
    agent delivers was recoverable from the eroded prompt alone.
    """

    name: str
    composer: LanguageModel

    def act(self, request: str, turns: List[Turn], questions_left: int) -> Action:
        return "finish", self.finalize(request, turns)

    def finalize(self, request: str, turns: List[Turn]) -> str:
        response = self.composer.get_response(
            request=request, conversation=format_conversation(request, turns)
        )
        return response.description.strip() if response else request


@dataclasses.dataclass
class PromptedAskerAgent:
    """Prompted questioner: one structured ask/stop decision per turn, then composes."""

    name: str
    asker: LanguageModel
    composer: LanguageModel

    def act(self, request: str, turns: List[Turn], questions_left: int) -> Action:
        if questions_left > 0:
            decision = self.asker.get_response(
                request=request,
                conversation=format_conversation(request, turns),
                questions_left=questions_left,
            )
            if decision is not None and decision.ask and decision.question.strip():
                return "ask", decision.question.strip()
        return "finish", self.finalize(request, turns)

    def finalize(self, request: str, turns: List[Turn]) -> str:
        response = self.composer.get_response(
            request=request, conversation=format_conversation(request, turns)
        )
        return response.description.strip() if response else request


@dataclasses.dataclass
class RandomQAgent:
    """Asks template questions in seeded random order - separates "asking targeted
    questions" from "asking at all"."""

    name: str
    questions: List[str]
    n_questions: Optional[int]  # null = ask until the budget runs out
    composer: LanguageModel
    random_seed: int

    def _order(self, request: str) -> List[str]:
        rng = random.Random(f"{self.random_seed}:{request}")
        return rng.sample(self.questions, len(self.questions))

    def act(self, request: str, turns: List[Turn], questions_left: int) -> Action:
        asked = sum(1 for t in turns if t.role == "agent_question")
        order = self._order(request)
        limit = self.n_questions if self.n_questions is not None else len(order)
        if questions_left > 0 and asked < min(limit, len(order)):
            return "ask", order[asked]
        return "finish", self.finalize(request, turns)

    def finalize(self, request: str, turns: List[Turn]) -> str:
        response = self.composer.get_response(
            request=request, conversation=format_conversation(request, turns)
        )
        return response.description.strip() if response else request


@dataclasses.dataclass
class PolicyAgent:
    """The model group's ChatModel (base weights or a trained checkpoint) evaluated
    through the same episode loop as the prompted baselines. Speaks the
    QUESTION:/FINAL: convention from training.policy."""

    name: str
    policy: Any
    policy_prompt: str

    def act(self, request: str, turns: List[Turn], questions_left: int) -> Action:
        messages = build_messages(self.policy_prompt, request, turns, questions_left)
        kind, text = parse_action(self.policy.complete(messages, seed=0).text)
        if kind == "ask" and questions_left > 0:
            return "ask", text
        if kind == "ask":  # wanted to ask with no budget left - force a final
            return "finish", self.finalize(request, turns)
        return "finish", text

    def finalize(self, request: str, turns: List[Turn]) -> str:
        messages = build_messages(self.policy_prompt, request, turns, questions_left=0)
        _, text = parse_action(self.policy.complete(messages, seed=0).text)
        return text


@dataclasses.dataclass
class UpfrontAgent:
    """Asks one fixed broad question, then composes - the questionnaire-style baseline."""

    name: str
    question: str
    composer: LanguageModel

    def act(self, request: str, turns: List[Turn], questions_left: int) -> Action:
        if questions_left > 0 and not turns:
            return "ask", self.question
        return "finish", self.finalize(request, turns)

    def finalize(self, request: str, turns: List[Turn]) -> str:
        response = self.composer.get_response(
            request=request, conversation=format_conversation(request, turns)
        )
        return response.description.strip() if response else request
