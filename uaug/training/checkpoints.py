"""Load a trained checkpoint as a sampling Policy for evaluation (PolicyAgent).

Both backends return the same Policy the trainers use for rollouts, so a
checkpoint is evaluated by the exact machinery that trained it. Deferred imports:
only the selected backend's SDK needs to be installed.
"""

from uaug.training.policy import Policy


def load_policy(
    backend: str,
    checkpoint: str,
    base_model: str,
    renderer_name: str,
    max_tokens: int,
    temperature: float,
) -> Policy:
    if backend == "tinker":
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from uaug.training.trainers import _TinkerPolicy

        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=checkpoint)
        tokenizer = get_tokenizer(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=renderer.get_stop_sequences()
        )
        return _TinkerPolicy(sampling_client, renderer, sampling_params)

    if backend == "local":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from uaug.training.trainers import _HFPolicy

        path = checkpoint or base_model
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
        return _HFPolicy(model, tokenizer, max_tokens, temperature)

    raise ValueError(f"Unknown policy backend: {backend!r} (expected 'tinker' or 'local')")
