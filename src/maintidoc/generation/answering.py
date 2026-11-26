"""Grounded answering: generate, then prove the citations before showing them.

    dossier de preuves
      -> generation contrainte par AnswerV1
      -> verification structurelle  (la citation a-t-elle ete transmise ?)
      -> verification semantique    (le passage soutient-il l'affirmation ?)
      -> retrogradation si rien ne tient

The two checks answer different questions and neither subsumes the other. A
fabricated reference fails the first. A real reference attached to a claim it
does not support passes the first and fails the second. The brief requires
both to be caught, so both run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from maintidoc.contracts import AnswerStatus, AnswerV1, EvidenceBundle, RunVersions
from maintidoc.generation.client import GenerationError, GenerationResponse, VlmClient
from maintidoc.generation.grounding import (
    GroundingChecker,
    GroundingReport,
    check_grounding,
    downgrade_if_unsupported,
)
from maintidoc.generation.prompting import prepare_answer_prompt
from maintidoc.ingestion.store import ContentStore
from maintidoc.verification import VerificationReport, verify_structural


class MalformedAnswer(GenerationError):
    """The model returned something that is not a valid `AnswerV1`."""


@dataclass
class AnsweringResult:
    answer: AnswerV1
    bundle: EvidenceBundle
    structural: VerificationReport
    grounding: GroundingReport | None = None
    downgraded: bool = False
    raw: str = ""
    model: str = ""
    prompt_version: str = ""
    images_sent: int = 0
    elapsed_ms: float = 0.0
    versions: RunVersions | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_sourced(self) -> bool:
        """True only when both checks ran and both were satisfied.

        Deliberately strict: an answer whose semantic check never ran is not
        a sourced answer, it is an unverified one.
        """
        return (
            self.answer.status is AnswerStatus.ANSWERED
            and self.structural.ok
            and self.grounding is not None
            and self.grounding.any_support
            and not self.grounding.contradicted
        )

    @property
    def verified_refs(self) -> tuple[str, ...]:
        return self.structural.verified_refs


def parse_answer(raw: str) -> AnswerV1:
    """Parse and validate, tolerating a fenced code block.

    Constrained decoding should make fences impossible, but a model served
    without it still wraps JSON in ```json, and failing on that would be a
    pointless loss.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise MalformedAnswer(
            "Reponse vide : verifier que le mode raisonnement est desactive "
            "et que num_predict est suffisant."
        )
    try:
        return AnswerV1.model_validate_json(text)
    except ValidationError as exc:
        raise MalformedAnswer(f"Reponse non conforme a AnswerV1 : {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MalformedAnswer(f"Reponse non JSON : {exc}") from exc


class Answerer:
    def __init__(
        self,
        client: VlmClient,
        *,
        checker: GroundingChecker | None = None,
        content: ContentStore | None = None,
        prompts_dir=None,
        num_ctx: int = 4096,
        num_predict: int = 900,
        versions: RunVersions | None = None,
    ) -> None:
        self.client = client
        self.checker = checker
        self.content = content
        self.prompts_dir = prompts_dir
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.versions = versions

    def answer(self, bundle: EvidenceBundle) -> AnsweringResult:
        started = time.perf_counter()
        warnings: list[str] = []

        if not bundle.items:
            # Nothing was retrieved; there is nothing for a model to be
            # faithful to, so no call is made.
            empty = AnswerV1(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer="Aucune source accessible ne permet de repondre a cette question.",
                citations=[],
                uncertainties=["Aucun element pertinent n'a ete recupere."],
                clarification_question=None,
                proposed_action=None,
            )
            return AnsweringResult(
                answer=empty,
                bundle=bundle,
                structural=verify_structural(empty, bundle),
                model=self.client.name,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                versions=self.versions,
                warnings=["Dossier de preuves vide : aucun appel au modele."],
            )

        prepared = prepare_answer_prompt(
            bundle,
            content=self.content,
            prompts_dir=self.prompts_dir,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        response: GenerationResponse = self.client.generate(prepared.request)
        if response.looks_truncated:
            warnings.append(
                "Generation interrompue par la limite de tokens : augmenter num_predict."
            )

        answer = parse_answer(response.text)

        # 1. Structural: was this reference actually transmitted?
        structural = verify_structural(answer, bundle)
        if not structural.ok:
            warnings.extend(structural.issues)

        # 2. Semantic: does the passage support the claim?
        grounding: GroundingReport | None = None
        downgraded = False
        if self.checker is not None:
            grounding = check_grounding(answer, bundle, self.checker)
            adjusted = downgrade_if_unsupported(answer, grounding)
            downgraded = adjusted is not answer
            answer = adjusted
            if downgraded:
                warnings.append(
                    "Reponse retrogradee en insufficient_evidence : "
                    "aucune citation ne soutient l'affirmation."
                )
            structural = verify_structural(answer, bundle)
        else:
            warnings.append(
                "Aucun verificateur semantique : le soutien des citations n'est pas etabli."
            )

        return AnsweringResult(
            answer=answer,
            bundle=bundle,
            structural=structural,
            grounding=grounding,
            downgraded=downgraded,
            raw=response.text,
            model=response.model,
            prompt_version=f"{prepared.prompt.name}/{prepared.prompt.version}",
            images_sent=prepared.images_sent,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            versions=self.versions,
            warnings=warnings,
        )
