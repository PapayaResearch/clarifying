import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from uaug.utils.schema import Completion


def make_renderer(base_model: str, tokenizer, thinking: bool) -> Any:
    name = model_info.get_recommended_renderer_name(base_model)
    if not thinking:
        name = f"{name}_disable_thinking"
    return renderers.get_renderer(name, tokenizer)


def make_sampling_params(renderer, max_tokens: int):
    return tinker.types.SamplingParams(max_tokens=max_tokens, stop=renderer.get_stop_sequences())


def split_thinking(message: Any) -> Tuple[str, str, bool]:
    """The visible text, the reasoning, and whether the response was well formed, for either shape the renderer returns."""
    content = message["content"]
    if isinstance(content, list):
        text = "".join(p["text"] for p in content if p["type"] == "text")
        thinking = "\n".join(p["thinking"] for p in content if p["type"] == "thinking")
    else:
        text, thinking = content, ""
    well_formed = "<think>" not in text and "</think>" not in text
    if "<think>" in text:
        before, _, rest = text.partition("<think>")
        block, _, after = rest.partition("</think>")
        thinking = "\n".join(x for x in (thinking, block) if x)
        text = before + after
    if "</think>" in text:
        before, _, text = text.rpartition("</think>")
        thinking = "\n".join(x for x in (thinking, before) if x)
    return text.strip(), thinking.strip(), well_formed


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
            # The renderer splits a clean response into its reasoning and its visible text
            message, termination = self.renderer.parse_response(sequence.tokens)
            text, thinking, well_formed = split_thinking(message)
            if termination.is_clean and well_formed:
                break
            # A completion that hit max_tokens or left a think tag in its text loses its answer: resample once
            logging.warning("%s completion (%d tokens), resampling",
                            "Truncated" if not termination.is_clean else "Malformed", len(sequence.tokens))
        # Token ids and logprobs for the training update
        meta = {
            "prompt_tokens": prompt.to_ints(),
            "tokens": list(sequence.tokens),
            "logprobs": list(sequence.logprobs),
        }
        return Completion(
            text=text, meta=meta, thinking=thinking or None,
            n_tokens=len(sequence.tokens), truncated=not termination.is_clean,
        )


def load_model(checkpoint: Optional[str], base_model: str, max_tokens: int, thinking: bool) -> TinkerModel:
    """Base weights (checkpoint null) or a trained sampler checkpoint as a TinkerModel."""
    service_client = tinker.ServiceClient()
    if checkpoint:
        sampling_client = service_client.create_sampling_client(model_path=checkpoint)
    else:
        sampling_client = service_client.create_sampling_client(base_model=base_model)
    renderer = make_renderer(base_model, get_tokenizer(base_model), thinking)
    return TinkerModel(sampling_client, renderer, make_sampling_params(renderer, max_tokens))
