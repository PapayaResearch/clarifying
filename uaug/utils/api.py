import hashlib
import json
import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Callable, Optional

import litellm

# Suppress litellm info messages
litellm.suppress_debug_info = True


class CostMeter:
    """Thread-safe accumulator of API spend across all calls in the process."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_usd = 0.0

    def add(self, amount_usd: float, max_usd: Optional[float]):
        with self._lock:
            self.total_usd += amount_usd
            if max_usd is not None and self.total_usd > max_usd:
                raise RuntimeError(
                    f"Cost meter exceeded budget: ${self.total_usd:.2f} > ${max_usd:.2f}. Aborting run."
                )


COST_METER = CostMeter()


def _cache_key(model: str, messages, params: dict, response_format) -> str:
    schema = None
    if response_format is not None:
        # Pydantic classes hash by name + JSON schema so prompt-identical calls with different schemas miss
        schema = getattr(response_format, "__name__", str(response_format))
        if hasattr(response_format, "model_json_schema"):
            schema = [schema, response_format.model_json_schema()]
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params, "schema": schema},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_response(content: str, usage: dict) -> SimpleNamespace:
    """Minimal object with the fields LanguageModel reads (choices[0].message.content, usage)."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(**usage),
    )


def create_api_call(
    model: str,
    delay: float,
    params: dict,
    cache_dir: Optional[str] = None,
    max_usd: Optional[float] = None,
) -> Callable:
    """Factory for the api_call function, with disk caching and cost metering."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def api_call(messages, tools=None, response_format=None):
        cache_path = None
        if cache_dir is not None and tools is None:
            key = _cache_key(model, messages, params, response_format)
            cache_path = os.path.join(cache_dir, f"{key}.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                return _to_response(cached["content"], cached["usage"])

        time.sleep(delay)
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            **params,
        )

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        COST_METER.add(cost, max_usd)

        if cache_path is not None:
            usage = response.usage
            cached = {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
                "model": model,
                "cost_usd": cost,
            }
            tmp_path = cache_path + f".{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            os.replace(tmp_path, cache_path)

        logging.debug("API call to %s cost $%.5f (total $%.4f)", model, cost, COST_METER.total_usd)
        return response

    return api_call
