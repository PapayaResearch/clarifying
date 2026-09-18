import dataclasses
import json
import logging
import os
from typing import Any, Dict, List

import tinker

from clarify.utils.schema import Trajectory
from clarify.utils.tinker import TinkerModel, make_renderer, make_sampling_params


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
        self.renderer = make_renderer(self.base_model, self.training_client.get_tokenizer(), self.thinking)
        self.sampling_params = make_sampling_params(self.renderer, self.max_tokens)

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
