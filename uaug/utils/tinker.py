import dataclasses
import logging
from typing import Any, Dict, List, Optional

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from uaug.utils.schema import Completion


def make_renderer(base_model: str, tokenizer, thinking: bool) -> Any:
    name = model_info.get_recommended_renderer_name(base_model)
    if not thinking:
        name = f"{name}_disable_thinking"
    return renderers.get_renderer(name, tokenizer)


def make_sampling_params(renderer, max_tokens: int):
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
    renderer = make_renderer(base_model, get_tokenizer(base_model), thinking)
    return TinkerModel(sampling_client, renderer, make_sampling_params(renderer, max_tokens))
