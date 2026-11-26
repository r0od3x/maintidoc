"""Ingestion unit tests, on PDFs built for the purpose.

Synthetic documents are used so each behaviour can be isolated: a figure with
an embedded image, a figure drawn in vector only, a footer carrying a page
number. None of that can be exercised reliably against a single real manual.
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image
from pydantic import ValidationError

from maintidoc.contracts import ExtractionMethod, Modality
from maintidoc.ingestion import (
    CorpusManifest,
    CorpusStore,
    DocumentSource,
    IngestionPipeline,
    IngestionStatus,
    LocalContentStore,
)
from maintidoc.ingestion.figures import (
    Caption,
    find_captions,
    match_caption,
    pick_image_for_caption,
)
from maintidoc.ingestion.pdf import boilerplate_key, chunk_blocks, rows_to_markdown

FOOTER_Y = 800.0


def _png(width: int = 300, height: int = 200, colour=(180, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _source(**overrides) -> DocumentSource:
    defaults = dict(
        path=Path("doc.pdf"),
        document_id="DOC-TEST",
        document_version="1.0",
        equipment_family="banc",
        acl_groups=("maintenance",),
        title="Document de test",
    )
    return DocumentSource(**(defaults | overrides))


def _write_pdf(path: Path, *, embedded_figure: bool, vector_figure: bool) -> Path:
    doc = pymupdf.open()
    for page_number in (1, 2):
        page = doc.new_page()
        page.insert_text((55, 80), f"{page_number}. Section de test", fontsize=13)
        page.insert_text(
            (55, 105),
            "Le voyant orange signale un avertissement sur le controleur A.",
            fontsize=10,
        )
        if page_number == 1 and embedded_figure:
            page.insert_image(pymupdf.Rect(60, 140, 360, 340), stream=_png())
            page.insert_text((60, 360), "FIG-01 : panneau avant et voyants", fontsize=9)
        if page_number == 2 and vector_figure:
            page.draw_rect(pymupdf.Rect(70, 150, 330, 330), color=(0, 0, 0), width=1.5)
            page.draw_line(pymupdf.Point(70, 240), pymupdf.Point(330, 240))
            page.insert_text((60, 360), "Figure 2 : chemins redondants", fontsize=9)
        # Footer differs only by its page number.
        page.insert_text(
            (49, FOOTER_Y), f"DOC-TEST | Manuel de test | Page {page_number}", fontsize=8
        )
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    return _write_pdf(tmp_path / "doc.pdf", embedded_figure=True, vector_figure=True)


@pytest.fixture
def pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(CorpusStore(tmp_path / "derived"), corpus_root=tmp_path)


# --- captions -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FIG-01 : schema logique", "FIG-01"),
        ("FIG 7 - panneau", "FIG-07"),
        ("Figure 3 : vue eclatee", "FIG-03"),
        ("Fig. 12 — detail", "FIG-12"),
        ("Schema 4 : cablage", "FIG-04"),
        ("Schéma 4 : cablage", "FIG-04"),
    ],
)
def test_caption_labels_are_normalised(text: str, expected: str):
    found = match_caption(text)
    assert found is not None
    assert found[0] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Le figurant est arrive",
        "Figurez-vous que le voyant clignote",
        "1. Presentation et perimetre",
        "",
    ],
)
def test_ordinary_prose_is_not_a_caption(text: str):
    assert match_caption(text) is None


def test_caption_absorbs_its_continuation_line():
    """A wrapped caption must be one caption, and its tail must not be indexed
    as body text — that was leaking fragments like 'constructeur reel.'."""
    blocks = [
        (55, 600, 500, 619, "FIG-01 : schema logique illustrant un cablage", 0, 0),
        (55, 624, 140, 633, "constructeur reel.", 1, 0),
        (55, 700, 400, 712, "Paragraphe suivant, sans rapport.", 2, 0),
    ]
    captions = find_captions(blocks)
    assert len(captions) == 1
    assert captions[0].text.endswith("constructeur reel.")
    assert captions[0].block_indices == (0, 1)


def test_caption_matches_the_image_above_it():
    caption = Caption("FIG-01", "FIG-01 : panneau", (55, 610, 500, 620), (0,))
    far = pymupdf.Rect(55, 100, 500, 200)
    near = pymupdf.Rect(55, 385, 500, 600)
    assert pick_image_for_caption(caption, [far, near]) == near


def test_caption_ignores_an_image_in_another_column():
    caption = Caption("FIG-01", "FIG-01 : panneau", (55, 610, 250, 620), (0,))
    elsewhere = pymupdf.Rect(400, 540, 560, 600)
    assert pick_image_for_caption(caption, [elsewhere]) is None


# --- page furniture and chunking -----------------------------------------


def test_boilerplate_key_collapses_page_numbers():
    assert boilerplate_key("DOC | Page 2") == boilerplate_key("DOC | Page 17")


def test_rows_to_markdown_pads_ragged_rows():
    markdown = rows_to_markdown([["Champ", "Valeur"], ["Equipement"]])
    assert markdown.splitlines()[0] == "| Champ | Valeur |"
    assert markdown.splitlines()[-1] == "| Equipement |  |"


def test_chunks_respect_the_character_budget():
    blocks = [(0, i * 10, 100, i * 10 + 8, "phrase de test. " * 12, i, 0) for i in range(6)]
    chunks = chunk_blocks(blocks, max_chars=300)
    assert chunks
    assert all(len(text) <= 330 for text, _ in chunks)


def test_oversized_block_is_split_on_sentences():
    long_text = "Une phrase complete. " * 60
    chunks = chunk_blocks([(0, 0, 100, 10, long_text, 0, 0)], max_chars=200)
    assert len(chunks) > 1
    assert all(len(text) <= 220 for text, _ in chunks)


# --- extraction end to end ------------------------------------------------


def test_embedded_figure_is_tied_to_its_caption(pipeline: IngestionPipeline, pdf_path: Path):
    pipeline.ingest_document(_source(), "test")
    elements = pipeline.store.read_elements("DOC-TEST")
    figure = next(e for e in elements if e.figure_id == "FIG-01")
    assert figure.modality is Modality.SCHEMA
    assert figure.extraction is ExtractionMethod.NATIVE
    assert figure.page == 1
    assert pipeline.store.content.exists(figure.image_uri)


def test_vector_figure_is_rendered(pipeline: IngestionPipeline, pdf_path: Path):
    """Line art is absent from get_images(); the prototype could not see it."""
    pipeline.ingest_document(_source(), "test")
    elements = pipeline.store.read_elements("DOC-TEST")
    figure = next(e for e in elements if e.figure_id == "FIG-02")
    assert figure.modality is Modality.SCHEMA
    assert figure.extraction is ExtractionMethod.RENDER
    assert figure.page == 2
    assert pipeline.store.content.exists(figure.image_uri)


def test_running_footer_is_not_indexed(pipeline: IngestionPipeline, pdf_path: Path):
    pipeline.ingest_document(_source(), "test")
    body = [
        e.text or "" for e in pipeline.store.read_elements("DOC-TEST")
        if e.modality is Modality.TEXT
    ]
    assert body, "le corps de page doit rester indexe"
    assert not any("Manuel de test" in text for text in body)


def test_access_policy_comes_from_the_manifest(pipeline: IngestionPipeline, pdf_path: Path):
    pipeline.ingest_document(_source(acl_groups=("groupe_a", "groupe_b")), "test")
    elements = pipeline.store.read_elements("DOC-TEST")
    assert elements
    assert all(e.acl_groups == ("groupe_a", "groupe_b") for e in elements)
    assert all(e.equipment_family == "banc" for e in elements)


def test_missing_file_fails_without_raising(pipeline: IngestionPipeline):
    result = pipeline.ingest_document(_source(path=Path("absent.pdf")), "test")
    assert result.status is IngestionStatus.FAILED
    assert "introuvable" in (result.error or "")


# --- idempotency and storage ---------------------------------------------


def test_unchanged_document_is_skipped(pipeline: IngestionPipeline, pdf_path: Path):
    first = pipeline.ingest_document(_source(), "test")
    second = pipeline.ingest_document(_source(), "test")
    assert first.status is IngestionStatus.INGESTED
    assert second.status is IngestionStatus.SKIPPED
    assert first.sha256 == second.sha256


def test_element_ids_are_stable_across_runs(pipeline: IngestionPipeline, pdf_path: Path):
    """Renumbering elements would invalidate every citation already recorded."""
    pipeline.ingest_document(_source(), "test")
    before = [e.ref.ref for e in pipeline.store.read_elements("DOC-TEST")]
    pipeline.ingest_document(_source(), "test", force=True)
    assert [e.ref.ref for e in pipeline.store.read_elements("DOC-TEST")] == before


def test_modified_document_is_replaced(pipeline: IngestionPipeline, pdf_path: Path):
    pipeline.ingest_document(_source(), "test")
    _write_pdf(pdf_path, embedded_figure=False, vector_figure=True)
    result = pipeline.ingest_document(_source(), "test")
    assert result.status is IngestionStatus.REPLACED


def test_two_manifest_entries_on_one_file_are_flagged(
    pipeline: IngestionPipeline, pdf_path: Path
):
    pipeline.ingest_document(_source(), "test")
    result = pipeline.ingest_document(_source(document_id="DOC-COPIE"), "test")
    assert any("DOC-TEST" in w for w in result.warnings)


def test_identical_bytes_are_stored_once(tmp_path: Path):
    store = LocalContentStore(tmp_path)
    first = store.put(_png(), extension="png")
    second = store.put(_png(), extension="png")
    assert first == second
    assert len(list((tmp_path / "objects").rglob("*.png"))) == 1


def test_content_store_rejects_a_foreign_uri(tmp_path: Path):
    with pytest.raises(ValueError, match="URI"):
        LocalContentStore(tmp_path).get("s3://bucket/objet.png")


# --- manifest -------------------------------------------------------------


def test_manifest_rejects_duplicate_document_ids():
    entry = _source().model_dump()
    with pytest.raises(ValidationError, match="double"):
        CorpusManifest(corpus_version="v1", documents=(entry, entry))


def test_manifest_requires_an_access_group():
    with pytest.raises(ValidationError):
        _source(acl_groups=())


# --- OCR wiring -----------------------------------------------------------
#
# Tesseract is not installed in every environment, so the engine is replaced by
# a stub. This proves the plumbing — that OCR output becomes its own citable
# element, tagged as OCR — without depending on a binary being present.


class StubOcr:
    name = "stub"

    def __init__(self, text: str = "Controleur A  Alarme temperature") -> None:
        self.text = text
        self.calls: list[int] = []

    def available(self) -> bool:
        return True

    def read(self, image: bytes, *, language: str = "fra") -> str:
        self.calls.append(len(image))
        return self.text


def test_text_inside_a_figure_becomes_its_own_element(tmp_path: Path, pdf_path: Path):
    ocr = StubOcr()
    pipeline = IngestionPipeline(
        CorpusStore(tmp_path / "derived"), corpus_root=tmp_path, ocr=ocr
    )
    pipeline.ingest_document(_source(), "test")
    elements = pipeline.store.read_elements("DOC-TEST")

    read_from_figure = [e for e in elements if e.extraction is ExtractionMethod.OCR]
    assert read_from_figure, "le texte lu sur une figure doit etre indexe"
    assert all(e.modality is Modality.TEXT for e in read_from_figure)
    assert any("Alarme temperature" in (e.text or "") for e in read_from_figure)
    assert ocr.calls


def test_a_page_declared_scanned_is_read_by_ocr(tmp_path: Path, pdf_path: Path):
    ocr = StubOcr("Texte reconnu sur une page scannee")
    pipeline = IngestionPipeline(
        CorpusStore(tmp_path / "derived"), corpus_root=tmp_path, ocr=ocr
    )
    pipeline.ingest_document(_source(scanned=True), "test")
    elements = pipeline.store.read_elements("DOC-TEST")
    assert any(e.element_id.endswith("ocr_01") for e in elements)


def test_ocr_output_is_distinguishable_from_a_text_layer(tmp_path: Path, pdf_path: Path):
    """An evaluation must be able to tell recognised text from real text."""
    pipeline = IngestionPipeline(
        CorpusStore(tmp_path / "derived"), corpus_root=tmp_path, ocr=StubOcr()
    )
    pipeline.ingest_document(_source(), "test")
    methods = {e.extraction for e in pipeline.store.read_elements("DOC-TEST")}
    assert ExtractionMethod.OCR in methods
    assert ExtractionMethod.NATIVE in methods


def test_without_an_engine_no_ocr_element_is_produced(
    pipeline: IngestionPipeline, pdf_path: Path
):
    pipeline.ingest_document(_source(), "test")
    elements = pipeline.store.read_elements("DOC-TEST")
    assert all(e.extraction is not ExtractionMethod.OCR for e in elements)
