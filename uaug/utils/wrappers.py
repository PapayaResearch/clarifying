import json
import logging
from typing import List, Type

import litellm
from pydantic import BaseModel

# Suppress litellm info messages
litellm.suppress_debug_info = True


class IOSchema(BaseModel):
    """Base class for all input and output schemas. Provides automatic formatting
    for message construction and logging."""

    @classmethod
    def to_description_string(cls) -> str:
        """Lists all fields and their descriptions (goes into the system prompt)."""
        parts = []
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description or "No description provided"
            parts.append(f"{field_name.upper()}:\n{description.strip()}\n\n")
        return "".join(parts).strip()

    def to_formatted_string(self) -> str:
        """Formats field values for the user message (lists/dicts as indented JSON)."""
        parts = []
        for field_name, field_value in self.model_dump().items():
            if isinstance(field_value, (list, dict)):
                field_value = json.dumps(field_value, indent=2, default=str)
            parts.append(f"{field_name.upper()}:\n{str(field_value).strip()}\n\n")
        return "".join(parts).strip()

    def __str__(self) -> str:
        return self.to_formatted_string()


class LanguageModel:
    """Generalized wrapper for LLM calls with Pydantic-validated structured outputs."""

    def __init__(
        self,
        system_prompt: str,
        input_schema: Type[IOSchema],
        output_schema: Type[IOSchema],
        api_call: callable,
        max_retries: int = 3,
        enable_json_schema_validation: bool = True,
    ):
        input_description = input_schema.to_description_string()
        full_system_prompt = f"{system_prompt}\n\nExpect the following inputs:\n\n{input_description}"

        self.system_prompt = full_system_prompt
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.api_call = api_call
        self.max_retries = max_retries

        # Enable JSON schema validation for models that don't natively support it
        if enable_json_schema_validation:
            litellm.enable_json_schema_validation = True

    def _build_messages(self, inputs: IOSchema) -> List[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": inputs.to_formatted_string()},
        ]

    def get_response(self, **kwargs):
        """Calls the LLM with structured output. Returns a validated output_schema
        instance, or None after all retries fail."""
        validated_inputs = self.input_schema(**kwargs)
        messages = self._build_messages(validated_inputs)

        result = None
        for attempt in range(self.max_retries):
            try:
                response = self.api_call(
                    messages=messages,
                    response_format=self.output_schema,
                )
                content = response.choices[0].message.content
                parsed_json = json.loads(content)
                result = self.output_schema(**parsed_json)
                break
            except Exception as e:
                logging.warning(
                    "Language Model API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                result = None

        if result is None:
            logging.error("Language Model API call failed: no valid response after all retries.")
        return result
