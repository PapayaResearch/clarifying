# UAUG

## Overview

**The User-Agent Utility Gap**: measuring, and training agents to close, the gap between what users want and what agents deliver. Hyper-detailed image descriptions are redacted, an agent recovers the missing details by asking a simulated user clarifying questions, a rubric judge scores each detail, and RL trains the agent to ask better.

The pipeline consists of the following stages:

1. **Synthesis**: Turn full descriptions into redacted task instances
2. **Training**: RL fine-tune the asking policy with LoRA via the Tinker API
3. **Evaluation**: Run agents against the simulated user and judge per-detail recovery

The `llm` config group is the frozen environment (synthesis, user simulator, judge, default `gemini-3.5-flash-lite`), the `model` group is the model under test that holds the conversation (default `qwen3-8b`, sampled via Tinker), and the `task` group holds the detail categories and prompts.

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

install the dependencies:

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
Decompose each description into atomic details, redact some of them (importance-weighted, preferring distinct categories), and erode the rest into a natural user request. The redaction scheme sets how many: `fixed-redactions` redacts exactly `k` per instance, `dynamic-redactions` draws a count per instance uniformly from 1 to a maximum.

### Run
```bash
python -m uaug.synthesis.run synthesis=<fixed-redactions|dynamic-redactions>
```

### Options
- `synthesis`: Redaction scheme (`fixed-redactions` | `dynamic-redactions`)
- `general.k`: Details redacted per instance (`fixed-redactions`), or the upper end of the per-instance count (`dynamic-redactions`)
- `synthesis.min_details`: Drop sources whose decomposition yields fewer details
- `synthesis.num_sources`: Number of sources to process (`null` for all)
- `synthesis.resume` / `synthesis.resume_path`: Resume a previous synthesis run
- `llm`: Environment LLM used to decompose and erode

### Outputs
```
results/<task>/<dataset>/synthesis/<llm>/<k_<k>|k_dyn<max>>/<datetime>/
├── instances.train.jsonl
├── instances.val.jsonl
├── instances.test.jsonl
├── synthesis_report.json
└── config.yaml
```

**Synthesis report**: yields, drop reasons, category counts, and the distribution of redacted counts

## 2. Training
Per iteration: sample instances, collect grouped rollouts (GRPO-style group-mean-centered advantages), judge them, and take one policy-gradient step on Tinker-managed LoRA weights. Rollouts mirror the selected agent, so a checkpoint is evaluated as exactly the agent it was trained as.

### Run
```bash
python -m uaug.training.run \
  agent=<agent_name> \
  training/reward=<reward_name> \
  "training.instances_dir=<synthesis_dir>"
```

### Options
- `agent`: The agent to train as (`fixed-asking` | `open-asking`)
- `agent.k_multiplier`: Question cap per episode as a multiple of the instance's redacted count
- `training/reward`: `terminal` (u_rec, questions free) | `constant_penalty` (u_rec − c·n) | `exponential_penalty` (question t costs c·g^t) | `discounted_penalty` (γ^n · u_rec)
- `training.importance_weighting`: Which weighting the reward optimizes (`llm` | `equal`)
- `training.n_iterations`: Total number of iterations, one gradient step each
- `training.instances_per_iteration` / `training.group_size`: Rollout batch shape
- `training.save_every`: Checkpoint interval
- `training.resume_dir`: Previous run directory to continue from its last checkpoint
- `training.trainer.lora_rank` / `training.trainer.learning_rate`: LoRA update knobs

### Outputs
```
results/<task>/<dataset>/training/<model>/<agent>/k_<k>/<reward>/<weighting>/<datetime>/
├── training_log.jsonl
├── trajectories.jsonl
├── checkpoints.jsonl
└── config.yaml
```

**Training log**: per-iteration reward, u_rec (both weightings), and question means

**Trajectories**: every judged rollout's transcript, judgment, reward, and advantage

**Checkpoints**: steps mapped to Tinker sampler paths (for evaluation) and state paths (for resuming)

## 3. Evaluation
Run one agent over instances and judge every episode: the agent sees only the eroded request, asks questions per its mode, and commits a final description. The judge checks each redacted detail against it, giving the recovery metric `u_rec` under both weightings plus `clairvoyant`, the best possible u_rec at each episode's question count.

### Run
```bash
python -m uaug.evaluation.run \
  agent=<agent_name> \
  "evaluation.instances_dir=<synthesis_dir>"
```

### Options
- `agent`: `no-asking` (no questions, the inferability floor) | `fixed-asking` (exactly as many questions as the instance has redacted details) | `open-asking` (model decides when to stop, capped at twice that count)
- `agent.k_multiplier`: Question cap per episode as a multiple of the instance's redacted count (0, 1 and 2 for the three agents)
- `general.k`: Match the instances' k
- `evaluation.split`: `val` | `test`
- `model.checkpoint`: Tinker sampler path from the training stage (`null` for base weights)
- `general.max_workers`: Parallelization level

### Outputs
```
results/<task>/<dataset>/evaluation/<model>/<agent>/k_<k>/<base|checkpoint>/<datetime>/
├── episodes.jsonl
├── judgments.jsonl
├── summary.csv
├── u_rec_hist_<llm|equal>.png
├── u_rec_by_questions_<llm|equal>.png
└── config.yaml
```

**Summary CSV**: one row per run (agent, mode, split, u_rec and clairvoyant means under both weightings, mean questions)

**Instance visuals**: `python scripts/visualize_eval.py <eval_dir> -n <n>` renders n random episodes as one image each into `<eval_dir>/figures/`
