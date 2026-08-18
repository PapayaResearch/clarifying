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


@dataclass
class Model:
    _target_: str
    name: str
    model_id: str
    checkpoint: Optional[str]
    max_tokens: int
    temperature: float


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


@dataclass
class Logging:
    log_dir: str
    results_dir: str
    console: bool


##############################
# Dataset Settings
##############################


@dataclass
class Dataset:
    name: str
    sources_path: str


##############################
# Task Settings
##############################


@dataclass
class Task:
    name: str
    categories: List[str]
    decompose_prompt: str
    erode_prompt: str
    user_prompt: str
    ask_prompt: str
    final_prompt: str
    decide_prompt: str
    judge_prompt: str


##############################
# Agent, User & Judge Settings
##############################


@dataclass
class Agent:
    _target_: str
    name: str
    mode: str
    max_questions: int
    model: Model
    ask_prompt: str
    final_prompt: str
    decide_prompt: str


@dataclass
class User:
    _target_: str
    model: LanguageModel


@dataclass
class Judge:
    _target_: str
    checker: LanguageModel


##############################
# Synthesis Settings
##############################


@dataclass
class Synthesis:
    _target_: str
    sources_path: str
    num_sources: Optional[int]
    task: str
    detail_categories: List[str]
    k: int
    importance_weighting: str
    min_details: int
    split_fractions: Dict[str, float]
    random_seed: int
    resume: bool
    resume_path: Optional[str]
    decomposer: LanguageModel
    eroder: LanguageModel


##############################
# Training Settings
##############################


@dataclass
class Trainer:
    _target_: str
    base_model: str
    lora_rank: int
    learning_rate: float
    max_tokens: int
    temperature: float


@dataclass
class Training:
    _target_: str
    instances_dir: Optional[str]
    split: str
    mode: str
    max_questions: int
    n_iterations: int
    instances_per_iteration: int
    group_size: int
    save_every: int
    normalize_advantages: bool
    ask_prompt: str
    final_prompt: str
    decide_prompt: str
    random_seed: int
    reward: Any  # shape depends on the conf/training/reward/* group choice
    trainer: Trainer
    user: User
    judge: Judge


##############################
# Evaluation Settings
##############################


@dataclass
class Evaluation:
    _target_: str
    instances_dir: Optional[str]
    split: str
    seeds: List[int]
    checkpoint: Optional[str]
    agent: Agent
    user: User
    judge: Judge


##############################
# Main Config
##############################


@dataclass
class Config:
    llm: ApiCall
    model: Model
    dataset: Dataset
    task: Task
    synthesis: Synthesis
    training: Training
    evaluation: Evaluation
    agent: Agent
    user: User
    judge: Judge
    general: General
    logging: Logging
