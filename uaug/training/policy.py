"""The trainable policy seam.

A Policy turns a chat-message list into one free-text completion. Backends attach
whatever they need for the gradient step (token ids, sampling logprobs) as opaque
`meta` that flows through Trajectory back into Trainer.update - the rollout loop
never inspects it.

Unlike the prompted baselines (structured ask/stop schema), a trained policy emits
plain text; the QUESTION:/FINAL: prefix convention below is its action space.
"""

import dataclasses
from typing import Any, Dict, List, Protocol, Tuple

from uaug.schema import Turn
from uaug.episodes.user import format_conversation


@dataclasses.dataclass
class Completion:
    """One sampled policy output plus backend-opaque training data."""
    text: str
    meta: Any = None


class Policy(Protocol):
    def complete(self, messages: List[Dict[str, str]], seed: int) -> Completion:
        ...


def build_messages(policy_prompt: str, request: str, turns: List[Turn], questions_left: int) -> List[Dict[str, str]]:
    """Chat messages for one agent turn: system prompt + current episode state."""
    state = (
        f"{format_conversation(request, turns)}\n\n"
        f"questions_left: {questions_left}"
    )
    return [
        {"role": "system", "content": policy_prompt},
        {"role": "user", "content": state},
    ]


def parse_action(text: str) -> Tuple[str, str]:
    """Map a policy completion onto the agent action space.

    "QUESTION: ..." -> ("ask", question); "FINAL: ..." -> ("finish", description).
    Anything malformed counts as a final answer of the raw text - the judge then
    scores it, so format violations are penalized through the reward, not masked.
    """
    stripped = text.strip()
    upper = stripped.upper()
    if upper.startswith("QUESTION:"):
        return "ask", stripped[len("QUESTION:"):].strip()
    if upper.startswith("FINAL:"):
        return "finish", stripped[len("FINAL:"):].strip()
    return "finish", stripped
