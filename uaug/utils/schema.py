import dataclasses
from typing import Any, Dict, List, Optional

# Detail categories follow the Proactive T2I Agents belief-graph vocabulary
# (entities/attributes/relations) extended with global scene properties.
DETAIL_CATEGORIES = ["entity", "attribute", "relation", "style", "count", "layout"]


@dataclasses.dataclass
class Detail:
    """One atomic, checkable fact extracted from a full specification."""
    id: int
    text: str
    category: str
    importance: float
    check: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detail":
        return cls(**d)


@dataclasses.dataclass
class TaskInstance:
    """A single elicitation problem: an eroded prompt hiding known redacted details."""
    instance_id: str
    task: str
    source_id: str
    full_spec: str
    details: List[Detail]
    redacted_ids: List[int]
    eroded_prompt: str
    split: str

    @property
    def redacted(self) -> List[Detail]:
        redacted_set = set(self.redacted_ids)
        return [d for d in self.details if d.id in redacted_set]

    @property
    def kept(self) -> List[Detail]:
        redacted_set = set(self.redacted_ids)
        return [d for d in self.details if d.id not in redacted_set]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskInstance":
        d = dict(d)
        d["details"] = [Detail.from_dict(x) for x in d["details"]]
        return cls(**d)


@dataclasses.dataclass
class Turn:
    """One transcript event."""
    role: str  # "agent_question" | "user_answer" | "agent_final"
    text: str


@dataclasses.dataclass
class Episode:
    """One full agent-user interaction on a task instance."""
    instance_id: str
    agent_name: str
    turns: List[Turn]
    final_output: str
    n_questions: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        d = dict(d)
        d["turns"] = [Turn(**x) for x in d["turns"]]
        return cls(**d)


@dataclasses.dataclass
class JudgeResult:
    """Per-detail verdicts for an episode, split into elicited (in transcript) and delivered (in final output)."""
    instance_id: str
    agent_name: str
    seed: int
    per_detail: Dict[int, Dict[str, bool]]
    u_rec: float
    eps_c_hat: float
    eps_e_hat: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeResult":
        d = dict(d)
        d["per_detail"] = {int(k): v for k, v in d["per_detail"].items()}
        return cls(**d)
