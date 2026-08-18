import dataclasses
from typing import Type

from pydantic import Field

from uaug.utils.schema import Episode, JudgeResult, TaskInstance
from uaug.utils.wrappers import IOSchema, LanguageModel


def check_input_schema(check_description: str, text_description: str) -> Type[IOSchema]:
    class CheckInput(IOSchema):
        """Input schema for one rubric check."""
        check: str = Field(description=check_description)
        text: str = Field(description=text_description)

    return CheckInput


def check_output_schema(satisfied_description: str) -> Type[IOSchema]:
    class CheckOutput(IOSchema):
        """Output schema for one rubric check."""
        satisfied: bool = Field(description=satisfied_description)

    return CheckOutput


@dataclasses.dataclass
class RubricJudge:
    """Scores an episode: per redacted detail, was it elicited (transcript) and delivered (final output)?"""

    checker: LanguageModel

    def _satisfied(self, check: str, text: str) -> bool:
        response = self.checker.get_response(check=check, text=text)
        return bool(response.satisfied) if response is not None else False

    def judge(self, instance: TaskInstance, episode: Episode) -> JudgeResult:
        answers = "\n".join(t.text for t in episode.turns if t.role == "user_answer")

        per_detail = {}
        for detail in instance.redacted:
            # Elicited: the user's answers revealed it (the agent's own words don't count as information)
            elicited = bool(answers) and self._satisfied(detail.check, answers)
            delivered = self._satisfied(detail.check, episode.final_output)
            per_detail[detail.id] = {"elicited": elicited, "delivered": delivered}

        weights = {d.id: max(d.importance, 0.05) for d in instance.redacted}
        total = sum(weights.values())
        u_rec = sum(w * per_detail[i]["delivered"] for i, w in weights.items()) / total
        eps_c = sum(w * (not per_detail[i]["elicited"]) for i, w in weights.items()) / total
        eps_e = sum(w * (per_detail[i]["elicited"] and not per_detail[i]["delivered"]) for i, w in weights.items()) / total

        return JudgeResult(
            instance_id=episode.instance_id,
            agent_name=episode.agent_name,
            seed=episode.seed,
            per_detail=per_detail,
            u_rec=u_rec,
            eps_c_hat=eps_c,
            eps_e_hat=eps_e,
        )
