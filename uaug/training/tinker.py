import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Optional

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from uaug.utils.schema import Completion, Trajectory


def _renderer(base_model: str, tokenizer, thinking: bool) -> Any:
    name = model_info.get_recommended_renderer_name(base_model)
    if not thinking:
        name = f"{name}_disable_thinking"
    return renderers.get_renderer(name, tokenizer)


def _sampling_params(renderer, max_tokens: int):
    return tinker.types.SamplingParams(max_tokens=max_tokens, stop=renderer.get_stop_sequences())


@dataclasses.dataclass
class TinkerModel:
    """The model under test, served by Tinker: one completion per call."""

    sampling_client: Any
    renderer: Any
    sampling_params: Any

    def complete(self, messages: List[Dict[str, str]]) -> Completion:
        prompt = self.renderer.build_generation_prompt(messages)
        for _ in range(2):
            result = self.sampling_client.sample(
                prompt=prompt, sampling_params=self.sampling_params, num_samples=1
            ).result()
            sequence = result.sequences[0]
            # get_text_content strips any <think> block, so text is only the visible answer
            message, _ = self.renderer.parse_response(sequence.tokens)
            text = get_text_content(message).strip()
            if "<think>" not in text:
                break
            # A think block that hit max_tokens before closing leaks raw reasoning: resample once
            logging.warning("Unclosed <think> block in completion (%d tokens), resampling", len(sequence.tokens))
        text = text.split("</think>")[-1].split("<think>")[0].strip()
        # Token ids and logprobs for the training update
        meta = {
            "prompt_tokens": prompt.to_ints(),
            "tokens": list(sequence.tokens),
            "logprobs": list(sequence.logprobs),
        }
        return Completion(text=text, meta=meta)


def load_model(checkpoint: Optional[str], base_model: str, max_tokens: int, thinking: bool) -> TinkerModel:
    """Base weights (checkpoint null) or a trained sampler checkpoint as a TinkerModel."""
    service_client = tinker.ServiceClient()
    if checkpoint:
        sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    else:
        sampling_client = service_client.create_sampling_client(base_model=base_model)
    renderer = _renderer(base_model, get_tokenizer(base_model), thinking)
    return TinkerModel(sampling_client, renderer, _sampling_params(renderer, max_tokens))


@dataclasses.dataclass
class TinkerTrainer:
    """Importance-sampling policy gradient on Tinker-managed LoRA weights."""

    base_model: str
    lora_rank: int
    learning_rate: float
    max_tokens: int
    thinking: bool

    def __post_init__(self):
        service_client = tinker.ServiceClient()
        self.training_client = service_client.create_lora_training_client(
            base_model=self.base_model, rank=self.lora_rank
        )
        self.renderer = _renderer(self.base_model, self.training_client.get_tokenizer(), self.thinking)
        self.sampling_params = _sampling_params(self.renderer, self.max_tokens)

    def load_state(self, state_path: str) -> None:
        """Resume weights and Adam optimizer state from a saved state checkpoint."""
        self.training_client.load_state_with_optimizer(state_path).result()

    def make_model(self) -> TinkerModel:
        """A model sampling from the latest weights (refresh after every update)."""
        sampling_client = self.training_client.save_weights_and_get_sampling_client()
        return TinkerModel(sampling_client, self.renderer, self.sampling_params)

    def _make_datum(self, turn_meta: Dict[str, Any], advantage: float):
        """One agent turn as a next-token datum (the advantage covers the sampled tokens, prompt tokens get zero)."""
        prompt_tokens = turn_meta["prompt_tokens"]
        sampled_tokens = turn_meta["tokens"]
        sampled_logprobs = turn_meta["logprobs"]
        full = prompt_tokens + sampled_tokens
        n_prompt = len(prompt_tokens)
        return tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": full[1:],
                "logprobs": [0.0] * (n_prompt - 1) + sampled_logprobs,
                "advantages": [0.0] * (n_prompt - 1) + [advantage] * len(sampled_tokens),
            },
        )

    def update(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """One gradient step over every agent turn of the batch."""
        datums = [
            self._make_datum(turn.completion.meta, trajectory.advantage)
            for trajectory in trajectories
            for turn in trajectory.agent_turns
        ]
        fwd_bwd = self.training_client.forward_backward(datums, loss_fn="importance_sampling")
        optim = self.training_client.optim_step(
            tinker.types.AdamParams(learning_rate=self.learning_rate)
        )
        fwd_bwd_result, _ = fwd_bwd.result(), optim.result()
        # The scalar loss summaries live in metrics
        metrics = {k: round(float(v), 6) for k, v in (fwd_bwd_result.metrics or {}).items()}
        return {"n_datums": len(datums), **metrics}

    def save(self, results_dir: str, step: int) -> None:
        """Save sampler weights (for evaluation) and full training state (for resume)."""
        sampler = self.training_client.save_weights_for_sampler(name=f"step-{step:05d}")
        state = self.training_client.save_state(name=f"state-{step:05d}")
        sampler_path = str(sampler.result().path)
        state_path = str(state.result().path)
        with open(os.path.join(results_dir, "checkpoints.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "checkpoint": sampler_path, "state": state_path}) + "\n")
        logging.info("Saved Tinker checkpoint for step %d: sampler %s, state %s", step, sampler_path, state_path)
