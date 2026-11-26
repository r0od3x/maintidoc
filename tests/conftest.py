from pathlib import Path

import pytest

from maintidoc.contracts import (
    Element,
    Evidence,
    EvidenceBundle,
    Modality,
    RetrievalBranch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


@pytest.fixture
def prompts_dir() -> Path:
    return PROJECT_ROOT / "prompts"


@pytest.fixture
def text_element() -> Element:
    return Element(
        document_id="SAN-MAN-001",
        document_version="1.1",
        document_sha256=SHA,
        page=4,
        element_id="p04_text",
        modality=Modality.TEXT,
        equipment_family="san",
        acl_groups=("maintenance",),
        text="Le voyant orange signale un avertissement sur le controleur A.",
    )


@pytest.fixture
def schema_element() -> Element:
    return Element(
        document_id="SAN-MAN-001",
        document_version="1.1",
        document_sha256=SHA,
        page=2,
        element_id="p02_image_01",
        modality=Modality.SCHEMA,
        equipment_family="san",
        acl_groups=("maintenance",),
        image_uri="s3://maintidoc/san/p02_image_01.png",
        figure_id="FIG-01",
    )


@pytest.fixture
def bundle(text_element: Element, schema_element: Element) -> EvidenceBundle:
    return EvidenceBundle(
        question="Que signifie le voyant orange sur le SAN-LAB 2000 ?",
        items=(
            Evidence.from_element(text_element, branch=RetrievalBranch.TEXT, rank=1),
            Evidence.from_element(schema_element, branch=RetrievalBranch.VISUAL, rank=1),
        ),
    )
