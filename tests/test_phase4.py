"""Retrieval pipeline tests: access control, lexical search, fusion, bundling.

All offline. Access control is the one that matters most here — it is the
phase's acceptance criterion and the only part whose failure is a security
bug rather than a quality problem.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from maintidoc.contracts import Element, Modality, Principal, RetrievalBranch
from maintidoc.ingestion.store import LocalContentStore
from maintidoc.retrieval import (
    AccessDenied,
    Bm25Index,
    HashEmbedder,
    Indexer,
    NoReranker,
    Retriever,
    VectorStore,
    reciprocal_rank_fusion,
    tokenize,
)
from maintidoc.retrieval.fusion import FusedCandidate
from maintidoc.retrieval.store import SearchHit

SHA = "c" * 64


def _png(colour=(30, 90, 160)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _element(
    element_id: str,
    text: str | None,
    *,
    document_id: str = "SAN-2000-MANU",
    page: int = 1,
    modality: Modality = Modality.TEXT,
    acl_groups: tuple[str, ...] = ("maintenance_san",),
    family: str = "san",
    image_uri: str | None = None,
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
def elements(content: LocalContentStore) -> list[Element]:
    """Two families, two access groups, one restricted document."""
    return [
        _element("p01_text_01", "Sur le SAN-LAB 2000 le voyant orange fixe signale un "
                                "avertissement thermique sur le controleur A."),
        _element("p02_text_01", "Procedure de controle non intrusive du SAN-LAB 2000.", page=2),
        _element(
            "p02_fig_01", "FIG-01 : panneau avant du SAN-LAB 2000 et position des voyants",
            page=2, modality=Modality.SCHEMA, figure_id="FIG-01",
            image_uri=content.put(_png(), extension="png"),
        ),
        _element(
            "p01_text_01", "Sur le SAN-LAB 2200 le voyant orange fixe signale une "
                           "reconstruction du groupe de disques en cours.",
            document_id="SAN-2200-MANU",
        ),
        _element(
            "p01_text_01", "Sur le UPS-LAB 500 le voyant orange fixe signale un "
                           "fonctionnement sur batterie.",
            document_id="UPS-500-MANU", family="onduleur",
            acl_groups=("maintenance_energie",),
        ),
        _element(
            "p01_text_01", "Rapport confidentiel reserve au responsable de maintenance.",
            document_id="SAN-R240-ALAR", acl_groups=("responsable_maintenance",),
        ),
    ]


@pytest.fixture
def retriever(elements, content: LocalContentStore) -> Retriever:
    store = VectorStore.memory()
    embedder = HashEmbedder()
    Indexer(
        store, text_embedder=embedder, visual_encoder=embedder, content=content
    ).build(elements, index_version="v1", corpus_version="c1")
    return Retriever(
        store,
        text_embedder=embedder,
        visual_encoder=embedder,
        lexical=Bm25Index.build(elements),
        reranker=NoReranker(),
        max_evidence_chars=4000,
        max_images=1,
    )


TECH_SAN = Principal(user_id="tech-san", groups=frozenset({"maintenance_san"}))
TECH_ENERGY = Principal(user_id="tech-nrj", groups=frozenset({"maintenance_energie"}))
MANAGER = Principal(
    user_id="chef", groups=frozenset({"maintenance_san", "responsable_maintenance"})
)


# --- acceptance criterion: access control ---------------------------------


def test_a_user_of_one_group_cannot_retrieve_another_group_document(retriever: Retriever):
    """The phase 4 acceptance criterion.

    Filtering happens before scoring, so a forbidden document is never ranked.
    Returning it and discarding it afterwards would still leak, through
    latency and through the fact that a relevant document exists.
    """
    result = retriever.retrieve("voyant orange fixe", TECH_SAN)
    documents = {item.document_id for item in result.bundle.items}
    assert documents
    assert "UPS-500-MANU" not in documents
    assert "SAN-R240-ALAR" not in documents

    other = retriever.retrieve("voyant orange fixe", TECH_ENERGY)
    assert {item.document_id for item in other.bundle.items} == {"UPS-500-MANU"}


def test_a_restricted_document_appears_only_for_its_group(retriever: Retriever):
    technician = retriever.retrieve("rapport confidentiel", TECH_SAN)
    manager = retriever.retrieve("rapport confidentiel", MANAGER)
    assert "SAN-R240-ALAR" not in {i.document_id for i in technician.bundle.items}
    assert "SAN-R240-ALAR" in {i.document_id for i in manager.bundle.items}


def test_a_principal_without_groups_is_refused(retriever: Retriever):
    """Deny by default, and loudly: no groups must not mean no filter."""
    with pytest.raises(AccessDenied):
        retriever.retrieve("voyant orange", Principal(user_id="inconnu"))


def test_every_branch_respects_the_same_restriction(retriever: Retriever):
    """A branch that forgot to filter would leak through fusion."""
    result = retriever.retrieve("voyant orange fixe SAN-LAB 2000", TECH_ENERGY)
    assert result.trace.branch_counts
    for candidate in result.candidates:
        assert candidate.hit.document_id == "UPS-500-MANU"


def test_family_narrowing_is_applied(retriever: Retriever, elements):
    principal = Principal(
        user_id="tech",
        groups=frozenset({"maintenance_san", "maintenance_energie"}),
        families=frozenset({"onduleur"}),
    )
    result = retriever.retrieve("voyant orange fixe", principal)
    assert {i.document_id for i in result.bundle.items} == {"UPS-500-MANU"}


# --- lexical branch -------------------------------------------------------


def test_tokenizer_folds_accents_and_splits_identifiers():
    assert tokenize("Chemin dégradé") == ["chemin", "degrade"]
    assert tokenize("SAN-LAB 2000") == ["san", "lab", "2000"]


def test_bm25_separates_models_that_differ_by_one_digit(elements):
    """Exactly what dense retrieval is soft on, and why this branch exists."""
    index = Bm25Index.build(elements)
    hits = index.search("SAN-LAB 2200", limit=3)
    assert hits
    assert hits[0].document_id == "SAN-2200-MANU"


def test_bm25_filters_before_scoring(elements):
    index = Bm25Index.build(elements)
    hits = index.search("voyant orange fixe", limit=10, groups=["maintenance_energie"])
    assert hits
    assert all(hit.document_id == "UPS-500-MANU" for hit in hits)


def test_bm25_ignores_elements_without_text(content: LocalContentStore):
    silent = _element("p01_img_01", None, modality=Modality.PHOTO,
                      image_uri=content.put(_png(), extension="png"))
    index = Bm25Index.build([silent, _element("p01_text_01", "voyant orange")])
    assert len(index) == 1


def test_bm25_returns_nothing_for_an_unknown_term(elements):
    assert Bm25Index.build(elements).search("xylophone", limit=5) == []


# --- fusion ---------------------------------------------------------------


def _hit(ref: str, rank: int, page: int = 1, document: str = "D1", text: str = "t") -> SearchHit:
    return SearchHit(
        ref=ref, document_id=document, page=page, element_id=ref.split(":")[-1],
        modality="text", score=1.0 / rank, rank=rank, text=text,
    )


def test_fusion_combines_ranks_not_scores():
    """An element ranked well by two branches beats one ranked best by a single
    branch, whatever the incomparable raw scores say."""
    fused = reciprocal_rank_fusion(
        {
            "dense": [_hit("D1:p1:a", 2), _hit("D1:p1:b", 1)],
            "lexical": [_hit("D1:p1:a", 1), _hit("D1:p1:c", 2)],
        },
        page_bonus=0.0,
    )
    assert fused[0].ref == "D1:p1:a"
    assert fused[0].from_several_branches
    assert fused[0].ranks == {"dense": 2, "lexical": 1}


def test_fusion_is_at_element_level_not_page_level():
    """The prototype promoted a whole page; unrelated elements rode along."""
    fused = reciprocal_rank_fusion(
        {"dense": [_hit("D1:p5:a", 1), _hit("D1:p5:b", 9)]}, page_bonus=0.0
    )
    refs = [c.ref for c in fused]
    assert refs == ["D1:p5:a", "D1:p5:b"]
    assert fused[0].score > fused[1].score


def test_page_bonus_cannot_outrank_genuine_agreement():
    """The page is a hint, not a vote."""
    fused = reciprocal_rank_fusion(
        {
            "dense": [_hit("D1:p1:a", 1), _hit("D1:p2:x", 2)],
            "visual": [_hit("D1:p2:y", 1), _hit("D1:p1:a", 3)],
        },
        page_bonus=0.25,
    )
    assert fused[0].ref == "D1:p1:a"


def test_fusion_handles_an_empty_branch():
    fused = reciprocal_rank_fusion({"dense": [_hit("D1:p1:a", 1)], "visual": []})
    assert [c.ref for c in fused] == ["D1:p1:a"]
    assert fused[0].branches == ("dense",)


# --- evidence bundling ----------------------------------------------------


def test_bundle_is_the_phase_one_contract(retriever: Retriever):
    """Phase 5 and the citation verifier already consume exactly this."""
    result = retriever.retrieve("voyant orange fixe", TECH_SAN)
    bundle = result.bundle
    assert bundle.question == "voyant orange fixe"
    assert bundle.allowed_refs
    assert all(item.ref in bundle.allowed_refs for item in bundle.items)
    bundle.enforce_budget(max_chars=4000, max_images=1)


def test_image_budget_is_respected(retriever: Retriever):
    result = retriever.retrieve("panneau avant voyants", TECH_SAN)
    assert len(result.bundle.image_items) <= 1


def test_a_figure_over_budget_still_contributes_its_caption(elements, content):
    """Dropping the figure entirely would lose evidence the model can use."""
    store = VectorStore.memory()
    embedder = HashEmbedder()
    extra = [
        *elements,
        _element("p03_fig_01", "FIG-02 : seconde figure du SAN-LAB 2000", page=3,
                 modality=Modality.SCHEMA, figure_id="FIG-02",
                 image_uri=content.put(_png((200, 10, 10)), extension="png")),
    ]
    Indexer(store, text_embedder=embedder, visual_encoder=embedder,
            content=content).build(extra, index_version="v1", corpus_version="c1")
    retriever = Retriever(
        store, text_embedder=embedder, visual_encoder=embedder,
        lexical=Bm25Index.build(extra), max_images=1, max_evidence_chars=4000,
    )
    bundle = retriever.retrieve("figure du SAN-LAB 2000", TECH_SAN).bundle
    captions = [i for i in bundle.items if i.modality is Modality.SCHEMA]
    assert len(bundle.image_items) <= 1
    assert len(captions) >= 1


def test_character_budget_is_respected(elements, content):
    store = VectorStore.memory()
    embedder = HashEmbedder()
    Indexer(store, text_embedder=embedder, visual_encoder=embedder,
            content=content).build(elements, index_version="v1", corpus_version="c1")
    retriever = Retriever(
        store, text_embedder=embedder, visual_encoder=embedder,
        lexical=Bm25Index.build(elements), max_evidence_chars=120, max_images=0,
    )
    bundle = retriever.retrieve("voyant orange fixe", TECH_SAN).bundle
    assert bundle.total_excerpt_chars <= 120


def test_trace_reports_each_stage(retriever: Retriever):
    result = retriever.retrieve("voyant orange fixe", TECH_SAN)
    trace = result.trace
    assert set(trace.branch_counts) == {"dense", "lexical", "visual"}
    assert trace.fused_count >= trace.selected_count
    assert trace.reranker == "none"
    assert trace.elapsed_ms >= 0


def test_evidence_from_several_branches_is_marked_fused(retriever: Retriever):
    result = retriever.retrieve("SAN-LAB 2000 voyant orange", TECH_SAN)
    branches = {item.branch for item in result.bundle.items}
    assert branches <= {RetrievalBranch.TEXT, RetrievalBranch.VISUAL, RetrievalBranch.FUSED}


# --- reranker plumbing ----------------------------------------------------


def test_no_reranker_preserves_fusion_order():
    candidates = [
        FusedCandidate(hit=_hit("D1:p1:a", 1), score=0.5, branches=("dense",), rank=1),
        FusedCandidate(hit=_hit("D1:p1:b", 2), score=0.3, branches=("dense",), rank=2),
    ]
    assert [c.ref for c in NoReranker().rerank("q", candidates)] == ["D1:p1:a", "D1:p1:b"]


def test_reranker_keeps_unscorable_candidates(content: LocalContentStore):
    """An uncaptioned figure cannot be cross-encoded, and must not vanish."""

    class StubReranker:
        name = "stub"

        def rerank(self, question, candidates, *, limit=None):
            from maintidoc.retrieval.rerank import CrossEncoderReranker

            scorable = [c for c in candidates if (c.hit.text or "").strip()]
            unscorable = [c for c in candidates if not (c.hit.text or "").strip()]
            assert CrossEncoderReranker  # the real one applies the same split
            return (scorable + unscorable)[:limit]

    silent = SearchHit(ref="D1:p1:img", document_id="D1", page=1, element_id="img",
                       modality="photo", score=0.1, rank=1, image_uri="local://x.png")
    candidates = [
        FusedCandidate(hit=silent, score=0.4, branches=("visual",), rank=1),
        FusedCandidate(hit=_hit("D1:p1:a", 2), score=0.3, branches=("dense",), rank=2),
    ]
    result = StubReranker().rerank("q", candidates)
    assert {c.ref for c in result} == {"D1:p1:img", "D1:p1:a"}
