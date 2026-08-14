"""The conversation-model seam: the model under test as one object.

conf/model/* instantiates a ChatModel - the open-weights model that holds the
conversation (asking questions, ask/stop decisions, composing the final
description) for prompted baselines, PolicyAgent, and the trainer alike.
conf/llm/* remains the orchestration model for environment roles (synthesis,
user simulator, judge), so the environment stays fixed while the model under
test varies and agent-side results remain directly comparable.

ChatModel implements both contracts a model is consumed through, so ${model}
can be passed wherever a model is an input:
- api_call (LanguageModel wrappers): chat_model(messages=..., response_format=...)
- Policy (PolicyAgent, rollouts): chat_model.complete(messages, seed)

It samples through the same Policy backends training uses - a prompted baseline
on provider=tinker samples from the exact serving stack RL training starts from.
"""

import dataclasses
import json
import logging
import os
import threading
from typing import Dict, List, Optional

from uaug.training.policy import Completion, Policy
from uaug.utils.api import _cache_key, _to_response

# Backend policies are memoized so every ChatModel instance with the same config
# (hydra instantiates one per ${model} reference) shares one sampling client /
# loaded model
_POLICIES = {}
_LOCK = threading.Lock()


def _shared_policy(
    provider: str,
    model_id: str,
    checkpoint: Optional[str],
    renderer_name: Optional[str],
    max_tokens: int,
    temperature: float,
) -> Policy:
    from uaug.training.checkpoints import load_policy  # deferred: backend SDKs

    key = (provider, model_id, checkpoint, renderer_name, max_tokens, temperature)
    with _LOCK:
        if key not in _POLICIES:
            _POLICIES[key] = load_policy(
                backend=provider,
                checkpoint=checkpoint,
                base_model=model_id,
                renderer_name=renderer_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return _POLICIES[key]


def _json_system_suffix(response_format) -> str:
    """Schema instruction appended to the system message (open-weights sampling
    has no native structured-output enforcement)."""
    schema = json.dumps(response_format.model_json_schema(), indent=2)
    return (
        "\n\nRespond with a single JSON object only - no code fences, no commentary. "
        f"It must validate against this JSON schema:\n{schema}"
    )


def _extract_json(text: str) -> str:
    """Best-effort extraction of the outermost JSON object from a sampled reply."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def _validates(content: str, response_format) -> bool:
    try:
        response_format(**json.loads(content))
        return True
    except Exception:
        return False


@dataclasses.dataclass
class ChatModel:
    """The model under test: name, weights, and how to sample it.

    provider "tinker" (managed sampling + LoRA RL) or "local" (HF transformers);
    checkpoint null = base weights. Structured outputs are requested by appending
    the JSON schema to the system message and parsed best-effort - the
    LanguageModel wrapper's retry loop and None-fallback absorb misses, and
    invalid outputs are never disk-cached so a retry can resample.
    """

    name: str
    provider: str
    model_id: str
    checkpoint: Optional[str]
    renderer_name: Optional[str]
    max_tokens: int
    temperature: float
    cache_dir: Optional[str] = None

    def __post_init__(self):
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._cache_params = {
            "provider": self.provider,
            "checkpoint": self.checkpoint,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @property
    def policy(self) -> Policy:
        """The backend sampling policy (built lazily, shared across instances)."""
        return _shared_policy(
            self.provider, self.model_id, self.checkpoint,
            self.renderer_name, self.max_tokens, self.temperature,
        )

    def complete(self, messages: List[Dict[str, str]], seed: int = 0) -> Completion:
        """Raw text completion - the Policy contract."""
        return self.policy.complete(messages, seed)

    def __call__(self, messages, tools=None, response_format=None):
        """Structured chat completion - the api_call contract."""
        if tools is not None:
            raise NotImplementedError(f"tool calls are not supported by the {self.provider} chat backend")

        prompt_messages = [dict(m) for m in messages]
        if response_format is not None:
            suffix = _json_system_suffix(response_format)
            if prompt_messages and prompt_messages[0].get("role") == "system":
                prompt_messages[0]["content"] += suffix
            else:
                prompt_messages.insert(0, {"role": "system", "content": suffix.strip()})

        cache_path = None
        if self.cache_dir is not None:
            key = _cache_key(self.model_id, prompt_messages, self._cache_params, response_format)
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                return _to_response(cached["content"], cached["usage"])

        completion = self.complete(prompt_messages)
        content = completion.text
        if response_format is not None:
            content = _extract_json(content)

        meta = completion.meta or {}
        n_prompt = len(meta.get("prompt_tokens") or [])
        n_sampled = len(meta.get("tokens") or [])
        usage = {
            "prompt_tokens": n_prompt,
            "completion_tokens": n_sampled,
            "total_tokens": n_prompt + n_sampled,
        }

        # Only cache results the caller will accept, so wrapper retries can resample
        if cache_path is not None and (response_format is None or _validates(content, response_format)):
            cached = {"content": content, "usage": usage, "model": self.model_id}
            tmp_path = cache_path + f".{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            os.replace(tmp_path, cache_path)

        logging.debug("Chat call to %s (%s): %d prompt + %d sampled tokens", self.model_id, self.provider, n_prompt, n_sampled)
        return _to_response(content, usage)
