"""Acceptance tests for phase 2, on the real SAN-LAB 2000 manual.

The synthetic tests prove each mechanism in isolation; these prove the pipeline
produces the right answer on the document the project actually ships with.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maintidoc.contracts import Modality
from maintidoc.ingestion import CorpusManifest, CorpusStore, IngestionPipeline

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
MANIFEST = CORPUS_ROOT / "manifest.yaml"
DOCUMENT_ID = "SAN-MAN-001"


@pytest.fixture(scope="module")
def elements(tmp_path_factory: pytest.TempPathFactory):
    """Ingest the real manual into a throwaway derived corpus."""
    manifest = CorpusManifest.from_file(MANIFEST)
    source = manifest.get(DOCUMENT_ID)
    if not source.resolve(CORPUS_ROOT).is_file():
        pytest.skip("Manuel SAN absent du depot")
    store = CorpusStore(tmp_path_factory.mktemp("derived"))
    pipeline = IngestionPipeline(store, corpus_root=CORPUS_ROOT)
    result = pipeline.ingest_document(source, manifest.corpus_version)
    assert result.status.value == "ingested", result.error
    return store.read_elements(DOCUMENT_ID)


def test_the_five_figures_are_found_by_caption(elements):
    """FIG-01..FIG-05, located by their captions rather than by page number."""
    figures = {e.figure_id: e.page for e in elements if e.modality is Modality.SCHEMA}
    assert figures == {
        "FIG-01": 2,
        "FIG-02": 3,
        "FIG-03": 4,
        "FIG-04": 5,
        "FIG-05": 6,
    }


def test_the_cover_image_is_not_a_schematic(elements):
    """The prototype's rule `2 <= page <= 6` happened to classify the cover
    correctly. Here it is an illustration because no caption refers to it."""
    cover = [e for e in elements if e.page == 1 and e.modality.is_visual]
    assert len(cover) == 1
    assert cover[0].modality is Modality.PHOTO
    assert cover[0].figure_id is None


def test_every_figure_keeps_its_caption_whole(elements):
    """A truncated caption loses the warning the document attaches to a figure."""
    captions = {
        e.figure_id: (e.text or "") for e in elements if e.modality is Modality.SCHEMA
    }
    assert captions["FIG-01"].endswith("constructeur reel.")
    assert captions["FIG-03"].endswith("prescrit.")
    assert all(text.rstrip().endswith(".") for text in captions.values())


def test_tables_are_extracted(elements):
    tables = [e for e in elements if e.modality is Modality.TABLE]
    assert len(tables) >= 8
    assert all("|" in (e.text or "") for e in tables)


def test_table_content_is_not_duplicated_as_body_text(elements):
    """Indexing a cell twice inflates whatever a table mentions."""
    page3_text = " ".join(
        e.text or ""
        for e in elements
        if e.page == 3 and e.modality is Modality.TEXT
    )
    page3_tables = " ".join(
        e.text or ""
        for e in elements
        if e.page == 3 and e.modality is Modality.TABLE
    )
    assert "WWPN" in page3_tables
    assert "Carte de connexion FC" not in page3_text


def test_running_footer_is_absent_from_every_element(elements):
    assert not any("Manuel pedagogique fictif" in (e.text or "") for e in elements)


def test_caption_fragments_do_not_leak_into_body_text(elements):
    body = [e.text or "" for e in elements if e.modality is Modality.TEXT]
    for fragment in ("constructeur reel.", "n’est prescrit.", "un SAN reel."):
        assert not any(text.strip().endswith(fragment) for text in body)


def test_provenance_is_complete(elements):
    """Without these fields a citation cannot be verified later."""
    assert elements
    for element in elements:
        assert element.document_id == DOCUMENT_ID
        assert element.document_version == "1.1"
        assert len(element.document_sha256) == 64
        assert element.acl_groups == ("maintenance", "maintenance_san")
        assert element.equipment_family == "san"
        assert element.page >= 1


def test_element_references_are_unique(elements):
    refs = [e.ref.ref for e in elements]
    assert len(refs) == len(set(refs))
