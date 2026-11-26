"""Indexing and vector-store tests.

Everything here runs offline: Qdrant in process, deterministic embeddings. The
point is to prove the plumbing — payloads, filters, idempotency, versioning —
not retrieval quality, which only real models can show.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from maintidoc.contracts import Element, Modality
from maintidoc.ingestion.store import LocalContentStore
from maintidoc.retrieval import (
    HashEmbedder,
    Indexer,
    IndexJournal,
    Space,
    VectorStore,
    access_filter,
    element_payload,
)

SHA = "b" * 64


def _png(colour=(10, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _element(
    element_id: str,
    *,
    page: int = 1,
    modality: Modality = Modality.TEXT,
    text: str | None = "texte de reference",
    image_uri: str | None = None,
    acl_groups: tuple[str, ...] = ("maintenance",),
    family: str = "san",
    document_id: str = "DOC-1",
    figure_id: str | None = None,
) -> Element:
    return Element(
        document_id=document_id,
        document_version="1.0",
        document_sha256=SHA,
        page=page,
        element_id=element_id,
        modality=modality,
        equipment_family=family,
        acl_groups=acl_groups,
        text=text,
        image_uri=image_uri,
        figure_id=figure_id,
    )


@pytest.fixture
def content(tmp_path: Path) -> LocalContentStore:
    return LocalContentStore(tmp_path / "objects")


@pytest.fixture
def store() -> VectorStore:
    vector_store = VectorStore.memory()
    yield vector_store
    vector_store.close()


@pytest.fixture
def corpus(content: LocalContentStore) -> list[Element]:
    """Two documents, two families, two access groups, text and images."""
    schema_uri = content.put(_png(), extension="png")
    photo_uri = content.put(_png((200, 40, 40)), extension="png")
    return [
        _element("p01_text_01", text="Le voyant orange fixe signale un avertissement thermique."),
        _element("p02_table_01", modality=Modality.TABLE, page=2, text="| orange | fixe |"),
        _element(
            "p02_fig_01", modality=Modality.SCHEMA, page=2,
            text="FIG-01 : panneau avant et voyants", image_uri=schema_uri,
            figure_id="FIG-01",
        ),
        _element(
            "p01_img_01", modality=Modality.PHOTO, text=None,
            image_uri=photo_uri,
        ),
        _element(
            "p01_text_01", document_id="DOC-2", family="onduleur",
            acl_groups=("maintenance_energie",),
            text="Sur cet onduleur, le voyant orange fixe signale un passage sur batterie.",
        ),
    ]


@pytest.fixture
def indexer(store: VectorStore, content: LocalContentStore) -> Indexer:
    return Indexer(
        store,
        text_embedder=HashEmbedder(name="hash-text"),
        visual_encoder=HashEmbedder(name="hash-visual"),
        content=content,
    )


# --- what goes into which index -------------------------------------------


def test_a_captioned_figure_lands_in_both_indexes(corpus, indexer: Indexer):
    """Findable by its caption and by its content, under one reference."""
    text_refs = {e.ref.ref for e in indexer.text_candidates(corpus)}
    visual_refs = {e.ref.ref for e in indexer.visual_candidates(corpus)}
    assert "DOC-1:p2:p02_fig_01" in text_refs
    assert "DOC-1:p2:p02_fig_01" in visual_refs


def test_an_uncaptioned_image_is_only_visual(corpus, indexer: Indexer):
    text_refs = {e.ref.ref for e in indexer.text_candidates(corpus)}
    visual_refs = {e.ref.ref for e in indexer.visual_candidates(corpus)}
    assert "DOC-1:p1:p01_img_01" not in text_refs
    assert "DOC-1:p1:p01_img_01" in visual_refs


def test_build_populates_both_collections(corpus, indexer: Indexer, store: VectorStore):
    run = indexer.build(corpus, index_version="v1", corpus_version="c1")
    assert run.text_points == 4
    assert run.visual_points == 2
    assert store.count(Space.TEXT) == 4
    assert store.count(Space.VISUAL) == 2
    assert run.documents == 2


# --- payload and provenance ----------------------------------------------


def test_payload_carries_everything_a_citation_needs():
    element = _element("p03_fig_01", modality=Modality.SCHEMA, page=3,
                       text="FIG-02 : vue", image_uri="local://abc.png", figure_id="FIG-02")
    payload = element_payload(element, index_version="v2", corpus_version="2026.11")
    assert payload["ref"] == "DOC-1:p3:p03_fig_01"
    assert payload["acl_groups"] == ["maintenance"]
    assert payload["index_version"] == "v2"
    assert payload["corpus_version"] == "2026.11"
    assert payload["figure_id"] == "FIG-02"


def test_reindexing_overwrites_rather_than_duplicating(corpus, indexer: Indexer, store):
    indexer.build(corpus, index_version="v1", corpus_version="c1")
    indexer.build(corpus, index_version="v2", corpus_version="c1")
    assert store.count(Space.TEXT) == 4
    assert store.count(Space.VISUAL) == 2

    hits = store.search(Space.TEXT, HashEmbedder().encode_query("voyant orange"), limit=5)
    assert hits
    assert len({hit.ref for hit in hits}) == len(hits)


def test_changing_model_dimension_is_refused(store: VectorStore):
    store.ensure_collection(Space.TEXT, 64)
    with pytest.raises(ValueError, match="reindexation"):
        store.ensure_collection(Space.TEXT, 128)
    store.ensure_collection(Space.TEXT, 128, recreate=True)


def test_a_missing_image_does_not_abort_the_index(corpus, store: VectorStore, content):
    broken = _element("p09_fig_01", modality=Modality.SCHEMA, page=9,
                      text="FIG-09 : absente", image_uri="local://" + "f" * 64 + ".png")
    indexer = Indexer(
        store,
        text_embedder=HashEmbedder(),
        visual_encoder=HashEmbedder(),
        content=content,
    )
    run = indexer.build([*corpus, broken], index_version="v1", corpus_version="c1")
    assert "DOC-1:p9:p09_fig_01" in run.skipped_images
    assert run.visual_points == 2
    assert run.text_points == 5


# --- access control, applied before scoring -------------------------------


def test_filter_restricts_results_to_permitted_groups(corpus, indexer: Indexer, store):
    indexer.build(corpus, index_version="v1", corpus_version="c1")
    query = HashEmbedder().encode_query("voyant orange fixe")

    energy = store.search(
        Space.TEXT, query, limit=10, query_filter=access_filter(["maintenance_energie"])
    )
    assert energy
    assert {hit.document_id for hit in energy} == {"DOC-2"}


def test_a_user_without_groups_sees_nothing(corpus, indexer: Indexer, store):
    """Deny by default. An empty group list must not mean 'everything'."""
    indexer.build(corpus, index_version="v1", corpus_version="c1")
    hits = store.search(
        Space.TEXT,
        HashEmbedder().encode_query("voyant"),
        limit=10,
        query_filter=access_filter([]),
    )
    assert hits == []


def test_no_filter_at_all_returns_everything(corpus, indexer: Indexer, store):
    """`None` means 'no restriction asked for' and is distinct from 'no groups'."""
    indexer.build(corpus, index_version="v1", corpus_version="c1")
    hits = store.search(Space.TEXT, HashEmbedder().encode_query("voyant"), limit=10)
    assert {hit.document_id for hit in hits} == {"DOC-1", "DOC-2"}


def test_family_filter_narrows_the_corpus(corpus, indexer: Indexer, store):
    indexer.build(corpus, index_version="v1", corpus_version="c1")
    hits = store.search(
        Space.TEXT,
        HashEmbedder().encode_query("voyant orange"),
        limit=10,
        query_filter=access_filter(None, families=["onduleur"]),
    )
    assert {hit.equipment_family for hit in hits} == {"onduleur"}


def test_deleting_a_document_removes_its_points(corpus, indexer: Indexer, store):
    indexer.build(corpus, index_version="v1", corpus_version="c1")
    store.delete_document(Space.TEXT, "DOC-2")
    hits = store.search(Space.TEXT, HashEmbedder().encode_query("voyant"), limit=10)
    assert "DOC-2" not in {hit.document_id for hit in hits}


# --- the index journal ----------------------------------------------------


def test_journal_records_models_and_versions(corpus, indexer: Indexer, tmp_path: Path):
    journal = IndexJournal(tmp_path / "index_runs.jsonl")
    journal.append(indexer.build(corpus, index_version="v1", corpus_version="c1"))
    journal.append(indexer.build(corpus, index_version="v2", corpus_version="c2"))

    runs = journal.runs()
    assert len(runs) == 2
    assert journal.latest().index_version == "v2"
    assert runs[0].text_model == "hash-text"
    assert runs[0].visual_model == "hash-visual"
    assert runs[0].duration_seconds >= 0


def test_journal_is_empty_before_any_run(tmp_path: Path):
    journal = IndexJournal(tmp_path / "absent.jsonl")
    assert journal.runs() == []
    assert journal.latest() is None


# --- embedding contracts --------------------------------------------------


def test_hash_embedder_is_deterministic_and_normalised():
    embedder = HashEmbedder()
    first = embedder.encode_passages(["voyant orange", "autre texte"])
    second = embedder.encode_passages(["voyant orange", "autre texte"])
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_hash_embedder_handles_empty_text():
    assert np.linalg.norm(HashEmbedder().encode_query("")) == pytest.approx(1.0)


def test_text_only_index_is_allowed(corpus, store: VectorStore):
    """Useful when no visual model is available."""
    indexer = Indexer(store, text_embedder=HashEmbedder())
    run = indexer.build(corpus, index_version="v1", corpus_version="c1")
    assert run.visual_points == 0
    assert run.visual_model is None
    assert store.count(Space.VISUAL) == 0


def test_local_mode_persists_across_reopen(tmp_path: Path, corpus, content):
    """The on-disk local store must survive a restart, unlike the prototype's
    in-memory index rebuilt on every launch."""
    path = tmp_path / "qdrant"
    first = VectorStore.local(path)
    Indexer(first, text_embedder=HashEmbedder(), visual_encoder=HashEmbedder(),
            content=content).build(corpus, index_version="v1", corpus_version="c1")
    assert first.count(Space.TEXT) == 4
    first.close()

    second = VectorStore.local(path)
    assert second.count(Space.TEXT) == 4
    second.close()
