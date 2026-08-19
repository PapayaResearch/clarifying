# UAUG

## Overview

**The User-Agent Utility Gap**: measuring, and training agents to close, the gap between what users want and what agents deliver. Hyper-detailed image descriptions are systematically redacted. An agent recovers the missing details by asking a simulated user clarifying questions, is scored per detail by a rubric judge, and is RL-trained to ask better.

The pipeline consists of the following stages, chained only by result paths:

1. **Synthesis**: Turn full descriptions into redacted task instances (decompose → redact → erode)
2. **Training**: RL fine-tune the asking policy with LoRA via the Tinker API
3. **Evaluation**: Run agents against the simulated user and judge per-detail recovery

The `llm` config group is the frozen environment for synthesis, the user simulator, and the judge (default `gemini-3-flash`). The `model` config group is the model under test that holds the conversation for every agent (default `qwen3-8b`, sampled via Tinker). The `task` config group holds the detail category vocabulary and every task-specific prompt, so a new domain is one yaml file.

## Installation

Clone the repo:

```bash
git clone https://github.com/PapayaResearch/uaug.git
cd uaug
```

Create a Python environment:

```bash
conda create -n uaug python=3.12
conda activate uaug
```

install the dependencies (the `train` extra is required since every agent samples through Tinker):

```bash
pip install -e ".[train]"
```

and create a `.env` file in the project root with your API keys:

> [!IMPORTANT]
> You must include a Gemini key (the environment LLM) and a Tinker key (the conversation model and RL training).

```bash
GEMINI_API_KEY="<YOUR_GEMINI_KEY>"
TINKER_API_KEY="<YOUR_TINKER_KEY>"

# HuggingFace Configuration (optional, for tokenizer downloads)
HF_TOKEN="<YOUR_HF_TOKEN>"
```

Finally, download and normalize the DOCCI descriptions:

```bash
python scripts/setup_docci.py --limit 500
```

> [!NOTE]
> All commands run from the repo root. `results/` and `logs/` are created there.

## 1. Synthesis

Decompose each source description into atomic details (category, importance weight, yes/no check), redact `k` of them (importance-weighted, preferring distinct categories), and erode the rest into a natural user request.

### Run
```bash
python -m uaug.synthesis.run
```

### Options
- `general.k`: Details redacted per instance (question counts and `min_details` derive from it)
- `synthesis.num_sources`: Number of sources to process (`null` for all)
- `synthesis.min_details`: Drop sources whose decomposition yields fewer details
- `synthesis.resume` / `synthesis.resume_path`: Resume a previous synthesis run
- `llm`: Environment LLM used for decompose/erode

### Outputs
```
results/synthesis/<task>/<dataset>/<llm>/k_<k>/<datetime>/
├── instances.train.jsonl
├── instances.val.jsonl
├── instances.test.jsonl
├── synthesis_report.json
└── config.yaml
```

**Synthesis report**: yields, drop reasons, and category counts

## 2. Training

Per iteration: sample instances, collect grouped rollouts (GRPO-style group-mean-centered advantages), judge them, and take one importance-sampling policy-gradient step on Tinker-managed LoRA weights. Rollouts mirror the selected agent's mode, question count, and prompts, so a checkpoint is evaluated as exactly the agent it was trained as.

### Run
```bash
python -m uaug.training.run \
  agent=<agent_name> \
  training/reward=<reward_name> \
  "training.instances_dir=<synthesis_dir>"
```

### Options
- `agent`: The agent to train as (`asker_fixed` | `asker_dynamic`)
- `training/reward`: `terminal` (reward = u_rec, questions free) | `terminal_costed` (u_rec − c·questions)
- `training.reward.question_cost`: The per-question cost c (terminal_costed)
- `training.importance_weighting`: Which weighting the reward optimizes (`llm` | `equal`)
- `training.n_iterations`: Number of training iterations (one gradient step each)
- `training.instances_per_iteration` / `training.group_size`: Rollout batch shape
- `training.save_every`: Checkpoint interval (the final iteration always saves)
- `training.trainer.lora_rank` / `training.trainer.learning_rate`: LoRA update knobs
- `training.trainer.temperature`: Rollout sampling temperature

### Outputs
```
results/training/<model>/<agent>/k_<k>/<reward>/<weighting>/<datetime>/
├── training_log.jsonl
├── trajectories.jsonl
├── checkpoints.jsonl
└── config.yaml
```

**Training log**: per-iteration reward, u_rec (both weightings), and question means

**Trajectories**: every judged rollout's transcript, judgment, reward, and advantage

**Checkpoints**: steps mapped to Tinker sampler paths for the evaluation stage

## 3. Evaluation

Run one agent over instances and judge every episode: the agent sees only the eroded request, asks questions per its mode, and commits a final description. The judge checks each redacted detail in the user's answers (ε̂C if never elicited) and in the final output (ε̂E if elicited but dropped), aggregated into the recovery metric `u_rec` under both weightings (`llm` importance-weighted and `equal` unweighted). `model.checkpoint=null` evaluates the base weights, and a training-stage checkpoint evaluates the fine-tuned model through the identical code path.

### Run
```bash
python -m uaug.evaluation.run \
  agent=<agent_name> \
  "evaluation.instances_dir=<synthesis_dir>"
```

### Options
- `agent`: `zeroq` (no questions, the inferability floor) | `asker_fixed` (exactly k questions) | `asker_dynamic` (model decides when to stop)
- `general.k`: Match the instances' k (fixed-mode count = k, dynamic cap = 2k)
- `agent.max_questions`: Explicit override of the derived question count/cap
- `evaluation.split`: `val` | `test`
- `model.checkpoint`: Tinker sampler path from the training stage (`null` = base weights)
- `model.temperature`: Agent sampling temperature (default 0 = greedy)
- `model.thinking`: Whether the model thinks before answering (default `false`)
- `general.max_workers`: Parallelization level

### Outputs
```
results/evaluation/<model>/<agent>/k_<k>/<base|checkpoint>/<datetime>/
├── episodes.jsonl
├── judgments.jsonl
├── summary.csv
├── u_rec_hist_<llm|equal>.png
├── u_rec_by_questions_<llm|equal>.png
└── config.yaml
```

**Summary CSV**: one row per run (agent, mode, split, u_rec/ε̂C/ε̂E means under both weightings, mean questions), with rows concatenated across runs forming the comparison table
