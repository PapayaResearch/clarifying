import litellm
import time
from typing import Callable

# Suppress litellm info messages
litellm.suppress_debug_info = True


def create_api_call(model: str, delay: float, params: dict) -> Callable:
    """
    Factory for creating the api_call function.
    """
    def api_call(messages, tools=None, response_format=None):
        time.sleep(delay)
        return litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            **params
        )
    return api_call
