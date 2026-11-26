"""Contract behaviour — the rules a prompt cannot guarantee."""

import pytest
from pydantic import ValidationError

from maintidoc.contracts import (
    AnswerStatus,
    AnswerV1,
    Element,
    ElementRef,
    Modality,
    answer_json_schema,
)

SHA = "a" * 64


def test_ref_round_trips():
    ref = ElementRef(document_id="SAN-MAN-001", page=4, element_id="p04_text")
    assert ref.ref == "SAN-MAN-001:p4:p04_text"
    assert ElementRef.parse(ref.ref) == ref


def test_malformed_ref_is_rejected():
    with pytest.raises(ValueError, match="malformée"):
        ElementRef.parse("SAN-MAN-001/page4")


def test_element_without_acl_group_is_rejected(text_element: Element):
    """Fail closed: no group means unreachable, never public."""
    payload = text_element.model_dump() | {"acl_groups": []}
    with pytest.raises(ValidationError):
        Element.model_validate(payload)


def test_visual_element_requires_an_image():
    with pytest.raises(ValidationError, match="image_uri"):
        Element(
            document_id="d", document_version="1", document_sha256=SHA, page=1,
            element_id="e", modality=Modality.SCHEMA, equipment_family="san",
            acl_groups=("maintenance",), text="pas d'image",
        )


def test_readable_by_denies_by_default(text_element: Element):
    assert text_element.readable_by(frozenset({"maintenance"}))
    assert not text_element.readable_by(frozenset({"autre_groupe"}))


def test_answered_without_citation_is_rejected():
    with pytest.raises(ValidationError, match="citer au moins une source"):
        AnswerV1(
            status=AnswerStatus.ANSWERED, answer="Le voyant signale un avertissement.",
            citations=[], uncertainties=[], clarification_question=None,
            proposed_action=None,
        )


def test_clarification_without_question_is_rejected():
    with pytest.raises(ValidationError, match="clarification_question"):
        AnswerV1(
            status=AnswerStatus.NEEDS_CLARIFICATION, answer="Modèle à confirmer.",
            citations=[], uncertainties=[], clarification_question=None,
            proposed_action=None,
        )


def test_action_cannot_ride_on_an_unsupported_answer():
    with pytest.raises(ValidationError, match="non étayée"):
        AnswerV1.model_validate(
            {
                "status": "needs_clarification",
                "answer": "Modèle à confirmer.",
                "citations": [],
                "uncertainties": [],
                "clarification_question": "Pouvez-vous photographier la plaque ?",
                "proposed_action": {
                    "kind": "create_ticket",
                    "ticket": {
                        "equipment_id": "SAN-LAB-2000",
                        "title": "Voyant orange",
                        "description": "Avertissement constaté sur le panneau.",
                    },
                },
            }
        )


def test_brief_section_7_example_validates():
    """The example printed in the project brief must parse unchanged."""
    answer = AnswerV1.model_validate(
        {
            "status": "needs_clarification",
            "answer": "Le modèle de l'équipement doit être confirmé.",
            "citations": [
                {"document_id": "manuel_pompe_a", "page": 18, "element_id": "figure_18_2"}
            ],
            "uncertainties": ["La plaque signalétique est illisible."],
            "clarification_question": "Pouvez-vous photographier la plaque signalétique ?",
            "proposed_action": None,
        }
    )
    assert answer.cited_refs == ("manuel_pompe_a:p18:figure_18_2",)


def test_schema_is_strict_enough_for_constrained_decoding():
    schema = answer_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "status", "answer", "citations", "uncertainties",
        "clarification_question", "proposed_action",
    }
