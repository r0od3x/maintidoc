"""Tests for the synthetic corpus generator.

Generation is scoped to a couple of models so the suite stays fast; the
properties under test do not depend on the corpus size.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maintidoc.contracts import ExtractionMethod, Modality
from maintidoc.ingestion import CorpusManifest, CorpusStore, IngestionPipeline
from maintidoc.synth import ALL_MODELS, generate_corpus, write_facts, write_manifest
from maintidoc.synth.catalog import SAN_MODELS, UPS_MODELS

SAMPLE = (SAN_MODELS[0], UPS_MODELS[0])


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("corpus")
    documents = generate_corpus(root / "raw", seed=42, models=SAMPLE)
    return root, documents


# --- the point of the whole corpus ---------------------------------------


def test_the_orange_indicator_means_something_different_on_every_model():
    """The corpus is useless unless retrieving the right document is the hard
    part. Ten models, one phrase, ten answers — a system that skips model
    identification cannot be right by luck.

    This is the brief's own fine-tuning example (§5): telling a temperature
    alarm apart from a similar indicator belonging to other equipment.
    """
    meanings = {}
    for model in ALL_MODELS:
        steady_orange = [
            led for led in model.leds if led.colour == "orange" and led.state == "fixe"
        ]
        assert steady_orange, f"{model.label} doit porter un voyant orange fixe"
        meanings[model.label] = steady_orange[0].meaning

    assert len(meanings) == len(ALL_MODELS)
    assert len(set(meanings.values())) == len(ALL_MODELS), "significations non distinctes"


def test_both_families_are_represented():
    families = {model.family for model in ALL_MODELS}
    assert families == {"san", "onduleur"}
    assert len([m for m in ALL_MODELS if m.family == "san"]) >= 4
    assert len([m for m in ALL_MODELS if m.family == "onduleur"]) >= 4


def test_access_groups_are_not_uniform():
    """Phase 4 needs documents a given user cannot read."""
    groups = {group for model in ALL_MODELS for group in model.acl_groups}
    assert "responsable_maintenance" in groups
    restricted = [m for m in ALL_MODELS if "maintenance" not in m.acl_groups]
    assert restricted, "au moins un modele doit etre d'acces restreint"


# --- determinism ----------------------------------------------------------


def test_same_seed_yields_identical_bytes(tmp_path: Path):
    """Shifting output would change element ids and invalidate past citations."""
    first = generate_corpus(tmp_path / "a", seed=7, models=SAMPLE[:1])
    second = generate_corpus(tmp_path / "b", seed=7, models=SAMPLE[:1])
    for left, right in zip(first, second, strict=True):
        assert left.path.read_bytes() == right.path.read_bytes()


def test_a_different_seed_changes_the_corpus(tmp_path: Path):
    first = generate_corpus(tmp_path / "a", seed=1, models=SAMPLE[:1])
    second = generate_corpus(tmp_path / "b", seed=2, models=SAMPLE[:1])
    assert any(
        left.path.read_bytes() != right.path.read_bytes()
        for left, right in zip(first, second, strict=True)
    )


def test_a_document_is_seeded_independently(tmp_path: Path):
    """Adding a model must not perturb the documents of the others."""
    alone = generate_corpus(tmp_path / "a", seed=5, models=SAMPLE[:1])
    together = generate_corpus(tmp_path / "b", seed=5, models=SAMPLE)
    by_id = {d.document_id: d for d in together}
    for document in alone:
        assert document.path.read_bytes() == by_id[document.document_id].path.read_bytes()


# --- manifest and ground truth -------------------------------------------


def test_manifest_is_valid_and_preserves_hand_written_entries(generated, tmp_path: Path):
    root, documents = generated
    kept = [
        {
            "path": "raw/san_lab_2000/manuel.pdf",
            "document_id": "SAN-MAN-001",
            "document_version": "1.1",
            "equipment_family": "san",
            "acl_groups": ["maintenance"],
            "title": "Manuel d'origine",
            "language": "fr",
        }
    ]
    path = tmp_path / "manifest.yaml"
    write_manifest(documents, path, root, corpus_version="test", keep=kept)

    manifest = CorpusManifest.from_file(path)
    assert manifest.corpus_version == "test"
    assert manifest.get("SAN-MAN-001").title == "Manuel d'origine"
    assert len(manifest.documents) == len(documents) + 1
    assert manifest.families == {"san", "onduleur"}


def test_facts_record_where_each_answer_was_written(generated, tmp_path: Path):
    _root, documents = generated
    path = tmp_path / "facts.json"
    count = write_facts(documents, path)
    facts = json.loads(path.read_text(encoding="utf-8"))

    assert count == len(facts) > 0
    for fact in facts:
        assert fact["page"] >= 1
        assert fact["expected"]
        assert fact["model"]
    subjects = {fact["subject"] for fact in facts}
    assert any("orange fixe" in subject for subject in subjects)


# --- the generated documents survive ingestion ---------------------------


def test_generated_documents_ingest_with_their_figures(generated, tmp_path: Path):
    root, documents = generated
    write_manifest(documents, root / "manifest.yaml", root, corpus_version="test")
    manifest = CorpusManifest.from_file(root / "manifest.yaml")
    pipeline = IngestionPipeline(CorpusStore(tmp_path / "derived"), corpus_root=root)
    results = pipeline.ingest_corpus(manifest)

    assert all(r.status.value == "ingested" for r in results), [r.error for r in results]
    elements = list(pipeline.store.iter_all_elements())
    assert elements

    figures = [e for e in elements if e.modality is Modality.SCHEMA]
    assert figures, "les legendes generees doivent produire des figures"
    assert all(figure.figure_id for figure in figures)


def test_both_figure_paths_are_exercised(generated, tmp_path: Path):
    """Raster front panels and vector schematics, so neither path rots."""
    root, documents = generated
    write_manifest(documents, root / "manifest.yaml", root, corpus_version="test")
    manifest = CorpusManifest.from_file(root / "manifest.yaml")
    pipeline = IngestionPipeline(CorpusStore(tmp_path / "derived"), corpus_root=root)
    pipeline.ingest_corpus(manifest)

    methods = {
        e.extraction for e in pipeline.store.iter_all_elements() if e.modality.is_visual
    }
    assert ExtractionMethod.NATIVE in methods
    assert ExtractionMethod.RENDER in methods


def test_tables_are_detected_in_generated_documents(generated, tmp_path: Path):
    """The ruled grid is what makes a generated table detectable."""
    root, documents = generated
    write_manifest(documents, root / "manifest.yaml", root, corpus_version="test")
    manifest = CorpusManifest.from_file(root / "manifest.yaml")
    pipeline = IngestionPipeline(CorpusStore(tmp_path / "derived"), corpus_root=root)
    pipeline.ingest_corpus(manifest)

    tables = [
        e for e in pipeline.store.iter_all_elements() if e.modality is Modality.TABLE
    ]
    assert len(tables) >= len(documents)
    assert any("orange" in (t.text or "") for t in tables)


def test_generated_footers_are_not_indexed(generated, tmp_path: Path):
    root, documents = generated
    write_manifest(documents, root / "manifest.yaml", root, corpus_version="test")
    manifest = CorpusManifest.from_file(root / "manifest.yaml")
    pipeline = IngestionPipeline(CorpusStore(tmp_path / "derived"), corpus_root=root)
    pipeline.ingest_corpus(manifest)

    assert not any(
        "MaintiDoc  |" in (e.text or "") for e in pipeline.store.iter_all_elements()
    )
