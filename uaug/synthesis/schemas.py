from typing import List, Type

from pydantic import BaseModel, Field

from uaug.utils.wrappers import IOSchema


class DecomposeInput(IOSchema):
    """Input schema for description decomposition."""
    description: str = Field(description="The complete, detailed image description to decompose.")


class DetailItem(BaseModel):
    """One atomic, checkable detail extracted from the description."""
    text: str = Field(description="The detail as one self-contained factual sentence naming its entities explicitly.")
    category: str = Field(description="One of: entity, attribute, relation, style, count, layout.")
    importance: float = Field(ge=0.0, le=1.0, description="How much the image would change if this detail were different, 0 to 1.")
    check: str = Field(description="Self-contained yes/no question verifying that a text states this detail.")


class DecomposeOutput(IOSchema):
    """Output schema for description decomposition."""
    details: List[DetailItem] = Field(description="Every concrete visual fact in the description, one atomic detail each.")


class ErodeInput(IOSchema):
    """Input schema for prompt erosion."""
    details: List[str] = Field(description="The image details the user's request should convey - and nothing else.")


class ErodeOutput(IOSchema):
    """Output schema for prompt erosion."""
    request: str = Field(description="Natural 1-3 sentence user request containing only the given details.")


# Zero-arg factories so Hydra can instantiate schema *classes* via _target_
def decompose_input_schema() -> Type[IOSchema]:
    return DecomposeInput


def decompose_output_schema() -> Type[IOSchema]:
    return DecomposeOutput


def erode_input_schema() -> Type[IOSchema]:
    return ErodeInput


def erode_output_schema() -> Type[IOSchema]:
    return ErodeOutput
