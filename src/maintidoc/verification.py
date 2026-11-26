"""Structural verification of citations.

This is the deterministic half of the brief's citation requirement, lifted out
of the prototype notebook's `check_answer` and given a single home.

WHAT THIS GUARANTEES
    A cited element was actually retrieved and actually transmitted to the
    model for this request. Fabricated references and references to documents
    the user may not read are rejected here.

WHAT THIS DOES NOT GUARANTEE
    That the cited passage *supports* the sentence it is attached to. The brief
    targets ">= 90 % of citations supporting their claim", which is a semantic
    property; it needs the grounding check built in phase 5. Never present an
    answer as "sourced" on the strength of this module alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from maintidoc.contracts import AnswerStatus, AnswerV1, EvidenceBundle


@dataclass(frozen=True, slots=True)
class VerificationReport:
    issues: tuple[str, ...]
    verified_refs: tuple[str, ...]
    semantic_grounding_checked: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues


def verify_structural(answer: AnswerV1, bundle: EvidenceBundle) -> VerificationReport:
    """Check every citation against the evidence actually sent to the model."""
    allowed = bundle.allowed_refs
    issues: list[str] = []
    verified: list[str] = []

    for citation in answer.citations:
        ref = citation.ref
        if ref not in allowed:
            issues.append(f"Citation non récupérée ou inventée : {ref}")
            continue
        evidence = bundle.get(ref)
        assert evidence is not None  # guaranteed by the membership test above
        if evidence.page != citation.page:
            issues.append(f"Page incohérente pour {ref}")
            continue
        verified.append(ref)

    if answer.status is AnswerStatus.ANSWERED and not verified:
        issues.append("Réponse affirmative sans aucune citation vérifiable")

    return VerificationReport(
        issues=tuple(issues),
        verified_refs=tuple(verified),
        semantic_grounding_checked=False,
    )
