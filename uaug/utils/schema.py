import dataclasses
from typing import Any, Dict, List


@dataclasses.dataclass
class Detail:
    """One atomic, checkable fact extracted from a full specification."""
    id: int
    text: str
    category: str
    importance: float
    check: str


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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskInstance":
        d = dict(d)
        d["details"] = [Detail(**x) for x in d["details"]]
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


@dataclasses.dataclass
class Completion:
    """One sampled model output plus backend-opaque training data (token ids, logprobs)."""
    text: str
    meta: Any = None


@dataclasses.dataclass
class AgentTurn:
    """One model call during a rollout: the exact prompt messages and the sampled completion."""
    messages: List[Dict[str, str]]
    completion: Completion


@dataclasses.dataclass
class JudgeResult:
    """Per-detail verdicts for an episode, split into elicited (in transcript) and delivered (in final output)."""
    instance_id: str
    agent_name: str
    per_detail: Dict[int, Dict[str, bool]]
    u_rec: float
    eps_c_hat: float
    eps_e_hat: float


@dataclasses.dataclass
class Trajectory:
    """One judged rollout with everything the training update consumes."""
    instance_id: str
    episode: Episode
    judgment: JudgeResult
    agent_turns: List[AgentTurn]
    reward: float
    advantage: float = 0.0
