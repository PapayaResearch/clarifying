# Learning to Clarify Underspecified Intents Under Limited Interaction

![Python version](https://img.shields.io/badge/python-3.12-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/uaug)

> [!NOTE]
> Code for the paper **Learning to Clarify Underspecified Intents Under Limited Interaction**.

AI assistants receive requests that leave out information needed for a good outcome, for example about users' preferences or goals. They must then either speculate or ask for more information before proceeding. We reconceptualize this as a value-of-information problem: the assistant should acquire information whose absence causes the greatest avoidable loss in user utility. This is rarely known *ex ante*; rather, assistants must *predict* it in order to optimally allocate limited user interactions. We instantiate this problem in image generation and derive a reinforcement learning framework using multi-turn simulated users to maximize utility recovery under uncertainty. In a preregistered study with 456 interactive sessions across 76 human participants, this helped users significantly better match reference images with significantly fewer questions, less total interaction time, and lower cost. This points toward a simple and scalable framework for training language model assistants to better disambiguate user intent by asking more informative questions.

<img width="1858" alt="banner" src="assets/main_figure.png" />

## Features

- 🧩 Instance synthesis from hyper-detailed image descriptions (DOCCI): decompose into atomic details, redact a fixed or variable number of them, and rewrite the rest as a natural request
- 🗣️ Privileged user simulators that answer only what is asked, with patient and impatient stopping behavior
- 🎯 Asking modes: no-asking, fixed-asking, and open-asking
- 🏋️ RL fine-tuning of the asking policy with group-centered policy gradients and LoRA through the Tinker API
- ⚖️ Rubric judge scoring the recovered utility per hidden detail
- 🖼️ Image generation for references, requests, and final descriptions with Gemini
- 🤖 Support for multiple orchestration LLMs through LiteLLM and base models through Tinker
- ⚙️ Hydra-based configuration system for reproducible experiments

## Installation

Create a Python environment:

```bash
conda create -n uaug python=3.12
conda activate uaug
```

install the dependencies:

```bash
pip install -e .
```

and create a `.env` file in the project root with your API keys:

> [!IMPORTANT]
> You must include a Gemini key (the environment LLM and image generation) and a Tinker key (the assistant model and RL training).

```bash
GEMINI_API_KEY="<YOUR_GEMINI_KEY>"
TINKER_API_KEY="<YOUR_TINKER_KEY>"
```

Finally, download and normalize the DOCCI descriptions:

```bash
python scripts/setup_docci.py
```

> [!NOTE]
> All commands run from the repository root.

## Synthesis

Decompose each description into atomic details with an importance weight and a check, redact some of them, and rewrite what remains as a natural user request.

```bash
python -m uaug.synthesis.run \
  synthesis=variable-redactions \
  general.k=15
```

**Options:**
- `synthesis`: variable-redactions (1 to `k` details per instance), fixed-redactions (exactly `k`)
- `general.k`: maximum (or fixed) number of redacted details (default: 15)
- `synthesis.num_sources`: number of sources to process (null for all)
- `llm`: gemini-3.5-flash-lite, gemini-3.7-flash
- `general.max_workers`: number of parallel workers (default: 256)

Results are saved in `results/image-gen/docci/synthesis/LLM/k_REDACTIONS/TIMESTAMP`, with `instances.{train,val,test}.jsonl` and a `synthesis_report.json`.

## Training

Each iteration samples instances, collects a group of rollouts per instance against the user simulator, judges them, and takes one policy-gradient step on LoRA weights with group-centered advantages.

```bash
python -m uaug.training.run \
  agent=fixed-asking \
  user=patient \
  training.instances_dir=results/image-gen/docci/synthesis/LLM/k_dyn15/TIMESTAMP
```

**Options:**
- `agent`: fixed-asking (one question per hidden detail), open-asking (the model decides when to stop)
- `user`: patient (answers every question), impatient (stops after a random number of them)
- `model`: qwen3-8b, qwen3.5-9b
- `training.n_iterations`: number of iterations, one gradient step each (default: 400)
- `training.resume_dir`: previous run directory to continue from its last checkpoint
- `general.max_workers`: number of parallel workers (default: 256)

Results are saved in `results/image-gen/docci/training/MODEL/AGENT/USER/k_15/REWARD/WEIGHTING/TIMESTAMP`, with the per-iteration `training_log.jsonl`, every judged rollout in `trajectories.jsonl`, and `checkpoints.jsonl` mapping steps to Tinker sampler paths.

## Evaluation

Run one assistant over a split and judge every episode: it sees only the request, asks questions per its asking mode, and composes a final description that the judge checks against each hidden detail.

```bash
python -m uaug.evaluation.run \
  agent=open-asking \
  user=impatient \
  evaluation.split=test \
  model.checkpoint=tinker://CHECKPOINT \
  evaluation.instances_dir=results/image-gen/docci/synthesis/LLM/k_dyn15/TIMESTAMP
```

**Options:**
- `agent`: no-asking, fixed-asking, open-asking
- `user`: patient, impatient
- `evaluation.split`: val, test
- `model.checkpoint`: Tinker sampler path from `checkpoints.jsonl` (null for the base weights)
- `general.max_workers`: number of parallel workers (default: 256)

Results are saved in `results/image-gen/docci/evaluation/MODEL/AGENT/USER/k_15/SPLIT/CHECKPOINT/TIMESTAMP`, with `episodes.jsonl`, `judgments.jsonl`, a one-row `summary.csv`, and recovery plots.

## Images & Figures

Render images with Gemini for the reference, the request, and the final description of each episode:

```bash
python scripts/setup_docci_images.py
python scripts/generate_images_for_data.py SYNTHESIS_DIR --split test
python scripts/generate_images_for_eval.py EVAL_DIR
```

**Options:**
- `--instances`: instance ids to render instead of the whole split or run
- `--workers`: number of parallel workers (default: 64)

Images are saved in `SYNTHESIS_DIR/images/` and `EVAL_DIR/images/`, and reruns only render what is missing.

Figures of single episodes, image comparisons across runs, and examples with the model's reasoning:

```bash
python scripts/visualize_episode.py EVAL_DIR -n 5
python scripts/visualize_instance.py INSTANCE_ID EVAL_DIR
python scripts/visualize_images.py --naive EVAL_DIR --fixed EVAL_DIR --open EVAL_DIR
python scripts/visualize_thinking.py EVAL_DIR --out DIR
```

## Configuration

The framework uses [Hydra](https://hydra.cc/) for configuration management. Take a look into `uaug/conf/` to see all the details and options.

## Custom Tasks & Datasets

Create `uaug/conf/task/your_task.yaml` with the detail categories and the decompose, erode, user, ask, final, decide and judge prompts, and run with `task=your_task`. For a new dataset, create `uaug/conf/dataset/your_dataset.yaml` pointing at a `sources.jsonl` with one `{source_id, full_spec}` per line, and run with `dataset=your_dataset`.

## FAQs

### Can I access the data, checkpoints and results from the experiments in the paper?

Reach out to us!

## Citing & Acknowledgements

If you use this code in your research, please cite the following paper:

```bibtex
TEMPORARY: add bibtex here
```
