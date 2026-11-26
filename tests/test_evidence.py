"""The evidence bundle is the reference for what the model was allowed to cite."""

import pytest
from pydantic import ValidationError

from maintidoc.contracts import Evidence, EvidenceBundle, RetrievalBranch


def test_allowed_refs_cover_every_item(bundle: EvidenceBundle):
    assert bundle.allowed_refs == {
        "SAN-MAN-001:p4:p04_text",
        "SAN-MAN-001:p2:p02_image_01",
    }


def test_text_and_image_items_are_separable(bundle: EvidenceBundle):
    assert len(bundle.text_items) == 1
    assert len(bundle.image_items) == 1
    assert bundle.image_items[0].image_uri


def test_duplicate_evidence_is_rejected(text_element):
    item = Evidence.from_element(text_element, branch=RetrievalBranch.TEXT, rank=1)
    with pytest.raises(ValidationError, match="dupliquées"):
        EvidenceBundle(question="q", items=(item, item))


def test_budget_is_enforced(bundle: EvidenceBundle):
    bundle.enforce_budget(max_chars=8500, max_images=1)
    with pytest.raises(ValueError, match="Budget images"):
        bundle.enforce_budget(max_chars=8500, max_images=0)
    with pytest.raises(ValueError, match="Budget texte"):
        bundle.enforce_budget(max_chars=5, max_images=1)
