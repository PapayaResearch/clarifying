from typing import List, Literal, Type

from pydantic import BaseModel, Field

from uaug.utils.wrappers import IOSchema


def decompose_input_schema(text_description: str) -> Type[IOSchema]:
    class DecomposeInput(IOSchema):
        """Input schema for decomposition."""
        text: str = Field(description=text_description)

    return DecomposeInput


def decompose_output_schema(
    categories: List[str],
    details_description: str,
    text_description: str,
    category_description: str,
    importance_description: str,
    check_description: str,
) -> Type[IOSchema]:
    class DetailItem(BaseModel):
        """One atomic, checkable detail extracted from the text."""
        text: str = Field(description=text_description)
        category: Literal[*categories] = Field(description=category_description)  # type: ignore
        importance: float = Field(ge=0.0, le=1.0, description=importance_description)
        check: str = Field(description=check_description)

    class DecomposeOutput(IOSchema):
        """Output schema for decomposition."""
        details: List[DetailItem] = Field(description=details_description)

    return DecomposeOutput


def erode_input_schema(details_description: str) -> Type[IOSchema]:
    class ErodeInput(IOSchema):
        """Input schema for prompt erosion."""
        details: List[str] = Field(description=details_description)

    return ErodeInput


def erode_output_schema(request_description: str) -> Type[IOSchema]:
    class ErodeOutput(IOSchema):
        """Output schema for prompt erosion."""
        request: str = Field(description=request_description)

    return ErodeOutput
