# Learning to Clarify Underspecified Intents Under Limited Interaction

![Python version](https://img.shields.io/badge/python-3.12-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/clarifying)

> [!NOTE]
> Code for the paper **[Learning to Clarify Underspecified Intents Under Limited Interaction](https://arxiv.org/abs/2610.04719)**.

AI assistants receive requests that leave out information needed for a good outcome, for example about users' preferences or goals. They must then either speculate or ask for more information before proceeding. We reconceptualize this as a value-of-information problem: the assistant should acquire information whose absence causes the greatest avoidable loss in user utility. This is rarely known *ex ante*; rather, assistants must *predict* it in order to optimally allocate limited user interactions. We instantiate this problem in image generation and derive a reinforcement learning framework using multi-turn simulated users to maximize utility recovery under uncertainty. In a preregistered study with 456 interactive sessions across 76 human participants, this helped users significantly better match reference images with significantly fewer questions, less total interaction time, and lower cost. This points toward a simple and scalable framework for training language model assistants to better disambiguate user intent by asking more informative questions.

<img width="1858" alt="banner" src="assets/main_figure.png" />

## Installation

Create a Python environment:

```bash
conda create -n clarify python=3.12
conda activate clarify
```

install the dependencies:

```bash
pip install uv
uv pip install -e .
```

and create a `.env` file in the project root with your API keys:

```bash
TINKER_API_KEY="<YOUR_TINKER_KEY>"
GEMINI_API_KEY="<YOUR_GEMINI_KEY>"
```

> [!IMPORTANT]
> You need a Tinker key (the assistant model and RL training) and a key for the environment LLM, which plays the simulated user, the judge and the decomposition and redaction models. The paper uses Gemini 3.5 Flash-Lite as the environment LLM, so a Gemini key is needed to fully replicate its results. Any other provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works as well, just choose its model with the `llm` option and add an API key to `.env`.

> [!NOTE]
> All commands run from the repository root. The data used in the paper is in `data/docci_instances/`, and every command uses it by default.

## Synthesis

Decompose each description into atomic details with an importance weight and a check, redact some of them, and rewrite what remains as a natural user request. Skip this step to use the paper's instances. Otherwise download the DOCCI descriptions and build your own:

```bash
python scripts/setup_docci.py
python -m clarify.synthesis.run
```

**Options:**
- `synthesis`: variable-redactions (1 to `k` details per instance), fixed-redactions (exactly `k`)
- `general.k`: maximum (or fixed) number of redacted details (default: 15)
- `synthesis.num_sources`: number of sources to process (null for all)
- `llm`: environment LLM for the decomposition and redaction models (default: gemini-3.5-flash-lite)

Results are saved in `results/image-gen/docci/synthesis/LLM/k_REDACTIONS/TIMESTAMP`, with `instances.{train,val,test}.jsonl` and a `synthesis_report.json`.

## Training

Each iteration samples instances, collects a group of rollouts per instance against the user simulator, judges them, and takes one policy-gradient step on LoRA weights with group-centered advantages.

```bash
python -m clarify.training.run \
  agent=fixed-asking \
  user=patient
```

**Options:**
- `agent`: fixed-asking (one question per hidden detail), open-asking (the assistant decides when to stop)
- `user`: patient (answers every question), impatient (stops after a random number of them)
- `model`: qwen3-8b, qwen3.5-9b
- `llm`: environment LLM for the user simulator and the judge (default: gemini-3.5-flash-lite)
- `training.instances_dir`: instances to train on (default: data/docci_instances)
- `training.n_iterations`: number of iterations, one gradient step each (default: 300)

Results are saved in `results/image-gen/docci/training/MODEL/AGENT/USER/k_15/REWARD/WEIGHTING/TIMESTAMP`, with the per-iteration `training_log.jsonl`, every judged rollout in `trajectories.jsonl`, and `checkpoints.jsonl` mapping steps to Tinker sampler paths.

## Evaluation

Run one assistant over a split and judge every episode: it sees only the request, asks questions per its asking mode, and composes a final description that the judge checks against each hidden detail.

```bash
python -m clarify.evaluation.run \
  agent=open-asking \
  user=impatient \
  evaluation.split=test \
  model.checkpoint=tinker://CHECKPOINT
```

**Options:**
- `agent`: no-asking, fixed-asking, open-asking
- `user`: patient, impatient
- `evaluation.split`: val, test
- `llm`: environment LLM for the user simulator and the judge (default: gemini-3.5-flash-lite)
- `evaluation.instances_dir`: instances to evaluate on (default: data/docci_instances)
- `model.checkpoint`: Tinker sampler path from `checkpoints.jsonl` (omit for the base weights)

Results are saved in `results/image-gen/docci/evaluation/MODEL/AGENT/USER/k_15/SPLIT/CHECKPOINT/TIMESTAMP`, with `episodes.jsonl`, `judgments.jsonl`, a one-row `summary.csv`, and recovery plots.

## Images & Analysis

The scripts below default to the 15 evaluations of the paper, which can be downloaded using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html):

```bash
aws s3 sync s3://clarifying-questions/results/ paper_results/ --no-sign-request
```

> [!NOTE]
> The scripts render pictures with Gemini 3.1 Flash Image, so they need `GEMINI_API_KEY` in `.env` to run.

Render the reference, request and final images with Gemini:

```bash
python scripts/setup_docci_images.py
python scripts/generate_images_for_data.py --split test
python scripts/generate_images_for_eval.py
```

Make the figures of the paper from its evaluations (pass an evaluation directory to use another):

```bash
python scripts/visualize_images.py
python scripts/visualize_instance.py INSTANCE_ID
```

Figures are saved in `scripts/figures/`.

## Trained Assistants

The two assistants of the paper, Qwen3-8B trained against the patient user and against the impatient user, can be downloaded into `assistants/` the same way:

```bash
aws s3 sync s3://clarifying-questions/assistants/ assistants/ --no-sign-request
```

They are LoRA adapters in [PEFT](https://huggingface.co/docs/peft) format, `patient_trained` and `impatient_trained`, to load on `Qwen/Qwen3-8B` for inference outside Tinker.

## Configuration

The framework uses [Hydra](https://hydra.cc/) for configuration management. Take a look into `clarify/conf/` to see all the details and options.

## Citing & Acknowledgements

If you use this code in your research, please cite the following paper:

```bibtex
@article{mr2024clarifying,
  title     = {Learning to Clarify Underspecified Intents Under Limited Interaction},
  author    = {M R, Pranav and Cherep, Manuel and Maes, Pattie and Singh, Nikhil},
  year      = {2026},
  url       = {https://arxiv.org/abs/2610.04719}
}
```

We received funding from SK Telecom with MIT’s Generative AI Impact Consortium (MGAIC). Research reported in this publication was supported by an Amazon Research Award, Fall 2024. Experiments conducted in this paper were generously supported via API credits provided by Thinking Machines and Google. MC is supported by a fellowship from “la Caixa” Foundation (ID 100010434) with code LCF/BQ/EU23/12010079. The authors acknowledge the MIT Office of Research Computing and Data for providing high performance computing resources that have contributed to the research results reported within this paper.
