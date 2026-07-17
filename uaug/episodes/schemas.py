from typing import Type

from pydantic import Field

from uaug.utils.wrappers import IOSchema


class UserAnswerInput(IOSchema):
    """Input schema for the simulated user's answer."""
    intent: str = Field(description="The complete ground-truth description of the image the user wants.")
    conversation: str = Field(description="The conversation so far: the user's original request and any Q&A turns.")
    question: str = Field(description="The assistant's clarification question to answer.")


class UserAnswerOutput(IOSchema):
    """Output schema for the simulated user's answer."""
    answer: str = Field(description="The user's short, casual, truthful answer to the question.")


class AskerInput(IOSchema):
    """Input schema for the ask/stop decision."""
    request: str = Field(description="The user's original image request.")
    conversation: str = Field(description="Clarification Q&A so far (empty if none).")
    questions_left: int = Field(description="How many more questions may be asked.")


class AskerOutput(IOSchema):
    """Output schema for the ask/stop decision."""
    reasoning: str = Field(description="One or two sentences on the most important unresolved aspects.")
    ask: bool = Field(description="True to ask a clarification question, false to stop and compose the final description.")
    question: str = Field(description="The single concise question to ask (empty string when ask is false).")


class ComposeInput(IOSchema):
    """Input schema for final description composition."""
    request: str = Field(description="The user's original image request.")
    conversation: str = Field(description="Clarification Q&A so far (empty if none).")


class ComposeOutput(IOSchema):
    """Output schema for final description composition."""
    description: str = Field(description="The complete, detailed image description.")


# Zero-arg factories so Hydra can instantiate schema *classes* via _target_
def user_answer_input_schema() -> Type[IOSchema]:
    return UserAnswerInput


def user_answer_output_schema() -> Type[IOSchema]:
    return UserAnswerOutput


def asker_input_schema() -> Type[IOSchema]:
    return AskerInput


def asker_output_schema() -> Type[IOSchema]:
    return AskerOutput


def compose_input_schema() -> Type[IOSchema]:
    return ComposeInput


def compose_output_schema() -> Type[IOSchema]:
    return ComposeOutput
