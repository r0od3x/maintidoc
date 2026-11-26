"""Semantic verification: does the cited passage actually support the claim?

The structural check written in phase 1 proves a citation points at something
really retrieved. It says nothing about whether that passage bears out the
sentence attached to it, and the brief targets ">= 90 % of citations supporting
their claim" — a semantic property.

Two checkers:

* `LlmGroundingChecker` reads claim and passage together with the
  `evidence_check_v1` prompt. This is the one that counts.
* `LexicalGroundingChecker` compares content words. It is a weak proxy, kept
  because it runs with no model at all: it reliably catches a citation with
  nothing in common with the claim, and it lets the pipeline be exercised and
  tested offline. It is never presented as equivalent to the LLM judge, and it
  abstains rather than guesses in the middle ground.

Neither may report "supported" by default. An unavailable checker leaves
`semantic_grounding_checked` false, and the answer is not presented as sourced.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from maintidoc.contracts import AnswerStatus, AnswerV1, EvidenceBundle
from maintidoc.generation.client import GenerationError, VlmClient
from maintidoc.generation.prompting import prepare_evidence_check_prompt

WORD = re.compile(r"[a-z0-9]{3,}")

# Words too common in this domain to carry evidence of support.
STOPWORDS = frozenset(
    """les des une und pour avec dans sur par que qui est sont etre cette ces
    son sa ses aux lui leur plus pas ne de la le du et en un au il elle nous
    vous ils elles doit peut tout tous toute toutes selon lors meme entre
    document documentaire page figure tableau voir""".split()
)


class GroundingVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY = "partially"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"  # checker abstained or was unavailable

    @property
    def is_adequate(self) -> bool:
        return self in (GroundingVerdict.SUPPORTED, GroundingVerdict.PARTIALLY)


@dataclass(frozen=True, slots=True)
class GroundingCheck:
    ref: str
    verdict: GroundingVerdict
    justification: str = ""
    quote: str | None = None
    overlap: float | None = None


@runtime_checkable
class GroundingChecker(Protocol):
    name: str

    def check(self, claim: str, ref: str, passage: str) -> GroundingCheck: ...


def _content_words(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    return {w for w in WORD.findall(ascii_text) if w not in STOPWORDS}


class LexicalGroundingChecker:
    """Content-word overlap. A floor, not a judge.

    Above `supported_at` the passage shares most of the claim's substance;
    below `unsupported_below` it shares almost none, which is enough to flag a
    citation pointing somewhere unrelated. Between the two it returns UNKNOWN
    rather than inventing a verdict it cannot justify.
    """

    name = "lexical"

    def __init__(self, supported_at: float = 0.60, unsupported_below: float = 0.20) -> None:
        self.supported_at = supported_at
        self.unsupported_below = unsupported_below

    def check(self, claim: str, ref: str, passage: str) -> GroundingCheck:
        claim_words = _content_words(claim)
        passage_words = _content_words(passage)
        if not claim_words or not passage_words:
            return GroundingCheck(ref, GroundingVerdict.UNKNOWN, "texte insuffisant")

        overlap = len(claim_words & passage_words) / len(claim_words)
        if overlap >= self.supported_at:
            verdict = GroundingVerdict.SUPPORTED
            why = f"recouvrement lexical eleve ({overlap:.0%})"
        elif overlap < self.unsupported_below:
            verdict = GroundingVerdict.UNSUPPORTED
            why = f"recouvrement lexical quasi nul ({overlap:.0%})"
        else:
            verdict = GroundingVerdict.UNKNOWN
            why = f"recouvrement intermediaire ({overlap:.0%}), verdict non tranche"
        return GroundingCheck(ref, verdict, why, overlap=overlap)


class LlmGroundingChecker:
    """Reads claim and passage together, via `evidence_check_v1`."""

    name = "llm"

    def __init__(self, client: VlmClient, *, prompts_dir=None, num_ctx: int = 4096) -> None:
        self.client = client
        self.prompts_dir = prompts_dir
        self.num_ctx = num_ctx
        self.name = f"llm:{client.name}"

    def check(self, claim: str, ref: str, passage: str) -> GroundingCheck:
        request, _template = prepare_evidence_check_prompt(
            claim, ref, passage, prompts_dir=self.prompts_dir, num_ctx=self.num_ctx
        )
        try:
            response = self.client.generate(request)
            payload = json.loads(response.text)
        except (GenerationError, json.JSONDecodeError, TypeError) as exc:
            # An unreachable or incoherent judge must not be read as approval.
            return GroundingCheck(
                ref, GroundingVerdict.UNKNOWN, f"verification indisponible : {exc}"
            )
        raw = str(payload.get("verdict", "")).lower()
        try:
            verdict = GroundingVerdict(raw)
        except ValueError:
            verdict = GroundingVerdict.UNKNOWN
        return GroundingCheck(
            ref,
            verdict,
            str(payload.get("justification", ""))[:400],
            quote=payload.get("quote") or None,
        )


@dataclass(frozen=True, slots=True)
class GroundingReport:
    checks: tuple[GroundingCheck, ...]
    checker: str

    @property
    def inadequate(self) -> tuple[GroundingCheck, ...]:
        return tuple(c for c in self.checks if not c.verdict.is_adequate)

    @property
    def adequate(self) -> tuple[GroundingCheck, ...]:
        return tuple(c for c in self.checks if c.verdict.is_adequate)

    @property
    def contradicted(self) -> tuple[GroundingCheck, ...]:
        return tuple(c for c in self.checks if c.verdict is GroundingVerdict.CONTRADICTED)

    @property
    def any_support(self) -> bool:
        return bool(self.adequate)

    @property
    def support_rate(self) -> float:
        """The brief's headline metric, per answer."""
        return len(self.adequate) / len(self.checks) if self.checks else 0.0


def check_grounding(
    answer: AnswerV1, bundle: EvidenceBundle, checker: GroundingChecker
) -> GroundingReport:
    """Check every citation of an answer against the passage it names.

    `AnswerV1` keeps citations as a flat list rather than inline markers, so a
    citation cannot be tied to one sentence. Each is therefore checked against
    the whole answer — a looser question ("does this passage support what was
    said?") but one that can actually be asked of the data available.
    """
    checks: list[GroundingCheck] = []
    for citation in answer.citations:
        evidence = bundle.get(citation.ref)
        if evidence is None:
            # Structural verification already rejects this; recorded, not judged.
            checks.append(
                GroundingCheck(
                    citation.ref, GroundingVerdict.UNSUPPORTED, "reference hors dossier"
                )
            )
            continue
        passage = (evidence.excerpt or "").strip()
        if not passage:
            checks.append(
                GroundingCheck(
                    citation.ref,
                    GroundingVerdict.UNKNOWN,
                    "preuve visuelle : soutien non verifiable par le texte",
                )
            )
            continue
        checks.append(checker.check(answer.answer, citation.ref, passage))
    return GroundingReport(tuple(checks), checker.name)


def downgrade_if_unsupported(answer: AnswerV1, report: GroundingReport) -> AnswerV1:
    """Demote an affirmative answer that no citation actually supports.

    This is the point of the whole phase: a confident answer whose sources do
    not bear it out is worse than an admission that the evidence is thin,
    because it looks sourced.
    """
    if answer.status is not AnswerStatus.ANSWERED:
        return answer
    if report.any_support and not report.contradicted:
        return answer

    reasons = [
        f"{check.ref} : {check.verdict.value}" for check in report.inadequate
    ] or ["aucune citation verifiee"]
    return AnswerV1(
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        answer=answer.answer,
        citations=answer.citations,
        uncertainties=[
            *answer.uncertainties,
            "Les sources citees ne soutiennent pas l'affirmation : " + " ; ".join(reasons),
        ],
        clarification_question=answer.clarification_question,
        proposed_action=None,  # no action may ride on an unsupported answer
    )
