"""The structured answer contract — `AnswerV1`.

This replaces the hand-written JSON schema of the prototype notebook. One model
now serves three jobs at once, so they cannot drift apart:

  1. it constrains generation (exported as a JSON schema to Ollama `format=`);
  2. it validates what came back;
  3. it is the response body of the API (phase 8).

Field names follow the example in the project brief (§7) exactly, so a reader
of the brief can diff the two.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maintidoc.contracts.common import ElementRef
from maintidoc.contracts.tickets import TicketDraft


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Citation(BaseModel):
    """Pointer to a retrieved element. Must resolve inside the evidence bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1)

    @property
    def ref(self) -> str:
        return ElementRef(
            document_id=self.document_id, page=self.page, element_id=self.element_id
        ).ref


class ProposedAction(BaseModel):
    """A sensitive action the assistant proposes but may never perform itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["create_ticket"]
    ticket: TicketDraft


class AnswerV1(BaseModel):
    """Structured answer, version 1.

    Coherence rules are enforced here rather than in a prompt, because a prompt
    cannot guarantee them: an affirmative answer without a source, or a request
    for clarification without a question, is rejected as malformed and must be
    regenerated or downgraded.
    """

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    answer: str
    citations: list[Citation]
    uncertainties: list[str]
    clarification_question: str | None
    proposed_action: ProposedAction | None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.status is AnswerStatus.ANSWERED and not self.citations:
            raise ValueError("Une réponse affirmative doit citer au moins une source")
        if self.status is AnswerStatus.NEEDS_CLARIFICATION and not (
            self.clarification_question and self.clarification_question.strip()
        ):
            raise ValueError("needs_clarification exige une clarification_question")
        if self.proposed_action is not None and self.status is not AnswerStatus.ANSWERED:
            raise ValueError(
                "Aucune action ne peut être proposée sur une réponse non étayée"
            )
        if not self.answer.strip():
            raise ValueError("Le champ answer ne peut pas être vide")
        return self

    @property
    def cited_refs(self) -> tuple[str, ...]:
        return tuple(c.ref for c in self.citations)


def answer_json_schema() -> dict[str, Any]:
    """JSON schema for constrained decoding (Ollama `format=`, vLLM guided JSON).

    `extra="forbid"` yields `additionalProperties: false`, and every field is
    declared without a default so all of them land in `required` — the two
    properties that made the prototype's hand-written schema work.
    """
    return AnswerV1.model_json_schema()
