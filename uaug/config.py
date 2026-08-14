from dataclasses import dataclass
from typing import Any, Dict, List, Optional


##############################
# Shared Settings
##############################


@dataclass
class ApiCall:
    _target_: str
    model: str
    delay: float
    params: dict
    cache_dir: Optional[str] = None
    max_usd: Optional[float] = None


@dataclass
class Model:
    # The model under test (uaug.models.ChatModel): the conversation side of
    # every agent - ${model} instantiates to the object itself (see conf/model/)
    _target_: str
    name: str
    provider: str
    model_id: str
    checkpoint: Optional[str]
    renderer_name: Optional[str]
    max_tokens: int
    temperature: float
    cache_dir: Optional[str] = None


@dataclass
class SchemaFactory:
    _target_: str


@dataclass
class LanguageModel:
    _target_: str
    system_prompt: str
    input_schema: SchemaFactory
    output_schema: SchemaFactory
    api_call: ApiCall
    max_retries: int = 3
    enable_json_schema_validation: bool = True


@dataclass
class General:
    max_workers: int
    random_seed: int
    max_usd: Optional[float]


@dataclass
class Logging:
    log_dir: str
    results_dir: str
    cache_dir: str
    console: bool


##############################
# Dataset Settings
##############################


@dataclass
class Dataset:
    name: str
    sources_path: str


##############################
# Prompt Settings
##############################


@dataclass
class Prompt:
    decompose_prompt: str
    erode_prompt: str
    user_prompt: str
    asker_prompt: str
    compose_prompt: str
    judge_prompt: str
    policy_prompt: str


##############################
# Redaction Settings
##############################


@dataclass
class Redaction:
    # Number of details to redact per instance
    k: int
    # Restrict redaction to these categories (null = all)
    categories: Optional[List[str]]


##############################
# Synthesis Settings
##############################


@dataclass
class Synthesis:
    _target_: str
    sources_path: str
    num_sources: Optional[int]
    k: int
    categories: Optional[List[str]]
    budget: int
    min_details: int
    report_examples: int
    split_fractions: Dict[str, float]
    random_seed: int
    resume: bool
    resume_path: Optional[str]
    decomposer: LanguageModel
    eroder: LanguageModel


##############################
# Episode Settings
##############################


@dataclass
class Episodes:
    _target_: str
    instances_dir: Optional[str]
    split: str
    seeds: List[int]
    budget_override: Optional[int]
    agent: Any
    user: Any
    judge: Any


##############################
# Training Settings
##############################


@dataclass
class Training:
    _target_: str
    instances_dir: Optional[str]
    split: str
    budget_override: Optional[int]
    n_iterations: int
    instances_per_iteration: int
    group_size: int
    save_every: int
    normalize_advantages: bool
    policy_prompt: str
    random_seed: int
    reward: Any
    trainer: Any
    user: Any
    judge: Any


##############################
# Main Config
##############################


@dataclass
class Config:
    llm: ApiCall
    model: Model
    dataset: Dataset
    prompt: Prompt
    redaction: Redaction
    synthesis: Synthesis
    episodes: Episodes
    training: Training
    agent: Any
    user: Any
    judge: Any
    general: General
    logging: Logging
