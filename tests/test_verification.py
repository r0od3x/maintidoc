"""Structural citation checks — provenance only, never semantic support."""

from maintidoc.contracts import AnswerStatus, AnswerV1, Citation, EvidenceBundle
from maintidoc.verification import verify_structural


def _answer(citations: list[Citation]) -> AnswerV1:
    return AnswerV1(
        status=AnswerStatus.ANSWERED,
        answer="Le voyant orange signale un avertissement.",
        citations=citations,
        uncertainties=[],
        clarification_question=None,
        proposed_action=None,
    )


def test_retrieved_citation_is_verified(bundle: EvidenceBundle):
    report = verify_structural(
        _answer([Citation(document_id="SAN-MAN-001", page=4, element_id="p04_text")]),
        bundle,
    )
    assert report.ok
    assert report.verified_refs == ("SAN-MAN-001:p4:p04_text",)


def test_fabricated_citation_is_caught(bundle: EvidenceBundle):
    report = verify_structural(
        _answer([Citation(document_id="SAN-MAN-001", page=9, element_id="p09_text")]),
        bundle,
    )
    assert not report.ok
    assert "inventée" in report.issues[0]


def test_real_element_cited_at_the_wrong_page_is_caught(bundle: EvidenceBundle):
    """The element exists, but not on the page the model claims."""
    report = verify_structural(
        _answer([Citation(document_id="SAN-MAN-001", page=7, element_id="p04_text")]),
        bundle,
    )
    assert not report.ok


def test_semantic_grounding_is_never_claimed(bundle: EvidenceBundle):
    report = verify_structural(
        _answer([Citation(document_id="SAN-MAN-001", page=4, element_id="p04_text")]),
        bundle,
    )
    assert report.semantic_grounding_checked is False
