"""Phase 4 against the real index and the real reranker.

Skipped unless both are available, rather than quietly passing on stand-ins.

Models are loaded **once per module and shared**. Building a retriever per test
reloaded BGE-M3 each time, and BGE-M3 plus the reranker plus both CLIP towers
is roughly 1.4 billion parameters in fp32 — enough to exhaust the Windows
paging file and take the process down with os error 1455.

The same constraint applies to the service built in phase 8: models must be
loaded once at startup and shared across requests, never per request.

The visual branch is left out here. These assertions are about reranking and
access control, neither of which needs it, and omitting it keeps two large
models out of the process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maintidoc.config import get_settings
from maintidoc.contracts import Principal
from maintidoc.ingestion.store import CorpusStore
from maintidoc.retrieval import (
    Bm25Index,
    IndexJournal,
    NoReranker,
    Retriever,
    Space,
    VectorStore,
)

pytestmark = pytest.mark.models

DERIVED = Path(__file__).resolve().parents[1] / "corpus" / "derived"
FAKE = {"hash-text", "hash-visual", "hash-test"}
TECH = Principal(user_id="tech-san", groups=frozenset({"maintenance", "maintenance_san"}))
QUESTION = "Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?"


def _require_real_index() -> None:
    latest = IndexJournal(DERIVED / "index_runs.jsonl").latest()
    if latest is None or latest.text_model in FAKE:
        pytest.skip("Index reel absent")


@pytest.fixture(scope="module")
def environment():
    """Index, corpus and the two models, loaded once for the whole module."""
    _require_real_index()
    settings = get_settings()
    store = VectorStore.local(DERIVED / "qdrant")
    if store.count(Space.TEXT) == 0:
        store.close()
        pytest.skip("Index vide")

    elements = list(CorpusStore(DERIVED).iter_all_elements())
    # BGE-M3 and the reranker together raise MemoryError on this machine, so
    # these tests drive the lexical branch into the reranker. That is enough
    # to exercise reranking and access control; the dense branch has its own
    # coverage in the phase 3 tests.
    try:
        from maintidoc.retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            settings.reranker_model, cache_dir=settings.model_cache_dir
        )
    except Exception as exc:
        store.close()
        pytest.skip(f"Reranker indisponible : {type(exc).__name__}: {exc}")

    lexical = Bm25Index.build(elements)

    def make(use_reranker: bool) -> Retriever:
        return Retriever(
            store,
            text_embedder=None,
            visual_encoder=None,
            lexical=lexical,
            reranker=reranker if use_reranker else NoReranker(),
            max_evidence_chars=settings.max_evidence_chars,
            max_images=settings.max_images,
            rerank_limit=20,
        )

    yield settings, make
    store.close()


def _first_correct(result) -> int:
    for index, item in enumerate(result.bundle.items, start=1):
        if "SAN-2000" in item.document_id:
            return index
    return 99


def test_reranking_puts_the_right_model_first(environment):
    """Ten models share this phrase; only the named one is correct.

    Fusion alone ranks a SAN-RACK 240 note first, with the SAN-LAB 2000
    document third. Closing that gap is what the reranker is for, and what
    phase 10's fine-tuning will be measured against.
    """
    _settings, make = environment
    result = make(True).retrieve(QUESTION, TECH, top_k=5)
    assert result.bundle.items
    assert "SAN-2000" in result.bundle.items[0].document_id, (
        f"premier resultat {result.bundle.items[0].document_id}"
    )


def test_reranking_improves_on_fusion_alone(environment):
    _settings, make = environment
    baseline = make(False).retrieve(QUESTION, TECH, top_k=5)
    reranked = make(True).retrieve(QUESTION, TECH, top_k=5)
    assert _first_correct(reranked) <= _first_correct(baseline)


def test_access_control_holds_on_the_real_corpus(environment):
    """The phase 4 acceptance criterion, on 41 real documents."""
    _settings, make = environment
    energy = Principal(user_id="nrj", groups=frozenset({"maintenance_energie"}))
    result = make(True).retrieve(QUESTION, energy, top_k=8)
    documents = {item.document_id for item in result.bundle.items}
    assert documents
    assert all(not name.startswith("SAN-") for name in documents), documents


def test_budgets_are_respected_on_the_real_corpus(environment):
    settings, make = environment
    result = make(True).retrieve(QUESTION, TECH, top_k=8)
    assert result.bundle.total_excerpt_chars <= settings.max_evidence_chars
    assert len(result.bundle.image_items) <= settings.max_images
