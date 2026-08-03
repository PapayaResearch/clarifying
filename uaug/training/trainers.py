"""Training backends behind one Trainer protocol, selected via conf/training/trainer/*.

- TinkerTrainer: managed LoRA RL via the Tinker API + importance-sampling policy
  gradient over full multi-turn episodes (needs TINKER_API_KEY in .env and the
  train extra: pip install -e .[train]).
- LocalDPOTrainer: local-GPU backend - offline contrastive pairs (best vs worst
  rollout per instance group) written to jsonl, optionally fit with trl's
  DPOTrainer + peft LoRA (pip install -e .[train_local] on the GPU box).

Both backends only implement make_policy / update / save. Rollouts, rewards, and
judging are backend-blind (rollouts.py).
"""

import dataclasses
import json
import logging
import os
import statistics
from typing import Any, Dict, List, Optional, Protocol

from uaug.training.policy import Completion, Policy
from uaug.training.rollouts import Trajectory


class Trainer(Protocol):
    name: str

    def make_policy(self) -> Policy:
        """A Policy sampling from the current weights (call again after update)."""
        ...

    def update(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """One gradient step (or pair-accumulation step) from a batch of trajectories."""
        ...

    def save(self, results_dir: str, step: int) -> str:
        """Persist a checkpoint (or dataset); returns its path/identifier."""
        ...


##############################
# Tinker backend
##############################


@dataclasses.dataclass
class _TinkerPolicy:
    """Samples one completion per call from a Tinker sampling client; meta carries
    the token ids and sampling logprobs the PG update needs."""

    sampling_client: Any
    renderer: Any
    sampling_params: Any

    def complete(self, messages: List[Dict[str, str]], seed: int) -> Completion:
        from tinker_cookbook.renderers import get_text_content  # deferred: tinker backend only

        prompt = self.renderer.build_generation_prompt(messages)
        result = self.sampling_client.sample(
            prompt=prompt, sampling_params=self.sampling_params, num_samples=1
        ).result()
        sequence = result.sequences[0]
        # parse_response -> (Message, ParseTermination); get_text_content strips
        # any <think> parts so parse_action sees only the QUESTION:/FINAL: text
        message, _ = self.renderer.parse_response(sequence.tokens)
        text = get_text_content(message).strip()
        meta = {
            "prompt_tokens": prompt.to_ints(),
            "tokens": list(sequence.tokens),
            "logprobs": list(sequence.logprobs),
        }
        return Completion(text=text, meta=meta)


@dataclasses.dataclass
class TinkerTrainer:
    """Importance-sampling policy gradient on Tinker-managed LoRA weights.

    Every agent turn of a trajectory becomes one datum: the turn's advantage is
    spread over its sampled tokens; prompt tokens get zero advantage.
    """

    name: str
    base_model: str
    lora_rank: int
    learning_rate: float
    renderer_name: str
    max_tokens: int
    temperature: float

    def __post_init__(self):
        import tinker  # deferred: only the tinker backend needs the SDK
        from tinker_cookbook import renderers

        self._tinker = tinker
        service_client = tinker.ServiceClient()
        self.training_client = service_client.create_lora_training_client(
            base_model=self.base_model, rank=self.lora_rank
        )
        tokenizer = self.training_client.get_tokenizer()
        self.renderer = renderers.get_renderer(self.renderer_name, tokenizer)
        self.sampling_params = tinker.types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.renderer.get_stop_sequences(),
        )

    def make_policy(self) -> Policy:
        sampling_client = self.training_client.save_weights_and_get_sampling_client()
        return _TinkerPolicy(sampling_client, self.renderer, self.sampling_params)

    def _make_datum(self, turn_meta: Dict[str, Any], advantage: float):
        """Standard next-token alignment: model_input is the full sequence minus the
        last token; targets/logprobs/advantages are zero-padded over the prompt."""
        prompt_tokens = turn_meta["prompt_tokens"]
        sampled_tokens = turn_meta["tokens"]
        sampled_logprobs = turn_meta["logprobs"]
        full = prompt_tokens + sampled_tokens
        n_prompt = len(prompt_tokens)
        return self._tinker.types.Datum(
            model_input=self._tinker.types.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": full[1:],
                "logprobs": [0.0] * (n_prompt - 1) + sampled_logprobs,
                "advantages": [0.0] * (n_prompt - 1) + [advantage] * len(sampled_tokens),
            },
        )

    def update(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        datums = [
            self._make_datum(turn.completion.meta, trajectory.advantage)
            for trajectory in trajectories
            for turn in trajectory.agent_turns
        ]
        fwd_bwd = self.training_client.forward_backward(datums, loss_fn="importance_sampling")
        optim = self.training_client.optim_step(
            self._tinker.types.AdamParams(learning_rate=self.learning_rate)
        )
        fwd_bwd_result, _ = fwd_bwd.result(), optim.result()
        # loss_fn_outputs is per-datum TensorData; the scalar summary lives in metrics
        metrics = {k: round(float(v), 6) for k, v in (fwd_bwd_result.metrics or {}).items()}
        return {"n_datums": len(datums), **metrics}

    def save(self, results_dir: str, step: int) -> str:
        # two checkpoint kinds: state (resume training) and sampler weights (the
        # only kind create_sampling_client accepts -> what PolicyAgent eval uses)
        state = self.training_client.save_state(name=f"step-{step:05d}")
        sampler = self.training_client.save_weights_for_sampler(name=f"step-{step:05d}")
        state_path, sampler_path = str(state.result().path), str(sampler.result().path)
        with open(os.path.join(results_dir, "checkpoints.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "checkpoint": sampler_path, "state": state_path}) + "\n")
        logging.info("Saved Tinker checkpoints for step %d: %s (sampler), %s (state)", step, sampler_path, state_path)
        return sampler_path


##############################
# Local backend (HF + trl DPO)
##############################


def build_dpo_pairs(trajectories: List[Trajectory], min_reward_gap: float) -> List[Dict[str, Any]]:
    """Contrastive pairs from grouped rollouts: within each instance group, pair the
    best and worst trajectory's FIRST agent turn (the only state all group members
    share) when the reward gap is informative."""
    groups: Dict[str, List[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.instance_id, []).append(trajectory)

    pairs = []
    for instance_id, group in groups.items():
        if len(group) < 2:
            continue
        best = max(group, key=lambda t: t.reward)
        worst = min(group, key=lambda t: t.reward)
        if best.reward - worst.reward < min_reward_gap:
            continue
        pairs.append({
            "instance_id": instance_id,
            "prompt_messages": best.agent_turns[0].messages,
            "chosen": best.agent_turns[0].completion.text,
            "rejected": worst.agent_turns[0].completion.text,
            "reward_chosen": best.reward,
            "reward_rejected": worst.reward,
        })
    return pairs


@dataclasses.dataclass
class _HFPolicy:
    """Local sampling via transformers generate (chat template applied by tokenizer)."""

    model: Any
    tokenizer: Any
    max_new_tokens: int
    temperature: float

    def complete(self, messages: List[Dict[str, str]], seed: int) -> Completion:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature or None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        return Completion(text=text, meta=None)


@dataclasses.dataclass
class LocalDPOTrainer:
    """Offline contrastive backend: update() accumulates DPO pairs; save() writes
    them to jsonl and (with fit_on_save, on a GPU box with the train_local extra)
    runs a trl DPO fit over everything accumulated so far."""

    name: str
    base_model: str
    checkpoint_path: Optional[str]
    lora_rank: int
    learning_rate: float
    beta: float
    min_reward_gap: float
    max_new_tokens: int
    temperature: float
    fit_on_save: bool

    def __post_init__(self):
        self.pairs: List[Dict[str, Any]] = []
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = self.checkpoint_path or self.base_model
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        return self._model, self._tokenizer

    def make_policy(self) -> Policy:
        model, tokenizer = self._load_model()
        return _HFPolicy(model, tokenizer, self.max_new_tokens, self.temperature)

    def update(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        new_pairs = build_dpo_pairs(trajectories, self.min_reward_gap)
        self.pairs.extend(new_pairs)
        gaps = [p["reward_chosen"] - p["reward_rejected"] for p in new_pairs]
        return {
            "n_new_pairs": len(new_pairs),
            "n_total_pairs": len(self.pairs),
            "mean_reward_gap": round(statistics.mean(gaps), 4) if gaps else None,
        }

    def save(self, results_dir: str, step: int) -> str:
        path = os.path.join(results_dir, "dpo_pairs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair) + "\n")
        logging.info("Wrote %d DPO pairs to %s", len(self.pairs), path)
        if self.fit_on_save and self.pairs:
            return self._fit(results_dir, step)
        return path

    def _fit(self, results_dir: str, step: int) -> str:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer

        model, tokenizer = self._load_model()
        rows = [
            {
                "prompt": tokenizer.apply_chat_template(
                    p["prompt_messages"], add_generation_prompt=True, tokenize=False
                ),
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
            for p in self.pairs
        ]
        output_dir = os.path.join(results_dir, f"dpo-step-{step:05d}")
        trainer = DPOTrainer(
            model=model,
            args=DPOConfig(
                output_dir=output_dir,
                beta=self.beta,
                learning_rate=self.learning_rate,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=10,
                report_to=[],
            ),
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
            peft_config=LoraConfig(r=self.lora_rank, lora_alpha=2 * self.lora_rank, task_type="CAUSAL_LM"),
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.checkpoint_path = output_dir
        self._model = None  # reload from the new checkpoint on next make_policy
        logging.info("DPO fit complete; checkpoint at %s", output_dir)
        return output_dir
