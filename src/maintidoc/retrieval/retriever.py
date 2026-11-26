"""The retrieval pipeline, in the order the brief fixes (§4.2).

    identite et permissions
      -> filtrage du corpus autorise
      -> recherche texte + lexicale + visuelle
      -> fusion RRF
      -> reranking
      -> dossier de preuves borne

The output is an `EvidenceBundle` — the phase 1 contract. Phase 5 consumes
exactly that, and the citation verifier already written checks against it, so
the two halves meet without an adapter.

The access decision is taken once, at the top, and pushed down into every
branch as a filter. No branch may widen it, and a principal with no group
retrieves nothing rather than everything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from maintidoc.contracts import (
    Evidence,
    EvidenceBundle,
    Principal,
    RetrievalBranch,
)
from maintidoc.retrieval.embeddings import TextEmbedder, VisualEncoder
from maintidoc.retrieval.fusion import FusedCandidate, reciprocal_rank_fusion
from maintidoc.retrieval.lexical import Bm25Index
from maintidoc.retrieval.rerank import NoReranker, Reranker
from maintidoc.retrieval.store import Space, VectorStore, access_filter

BRANCH_DENSE = "dense"
BRANCH_LEXICAL = "lexical"
BRANCH_VISUAL = "visual"

_BRANCH_TO_CONTRACT = {
    BRANCH_DENSE: RetrievalBranch.TEXT,
    BRANCH_LEXICAL: RetrievalBranch.TEXT,
    BRANCH_VISUAL: RetrievalBranch.VISUAL,
}


def _branch_of(candidate: FusedCandidate) -> RetrievalBranch:
    """One branch keeps its identity; several become FUSED."""
    if len(candidate.branches) != 1:
        return RetrievalBranch.FUSED
    return _BRANCH_TO_CONTRACT.get(candidate.branches[0], RetrievalBranch.FUSED)


class AccessDenied(Exception):
    """Raised when a principal holds no group at all."""


@dataclass
class RetrievalTrace:
    """What each stage did. Phase 9 reads this; a human debugging reads it too."""

    branch_counts: dict[str, int] = field(default_factory=dict)
    fused_count: int = 0
    reranked_count: int = 0
    selected_count: int = 0
    dropped_for_budget: int = 0
    reranker: str = "none"
    elapsed_ms: float = 0.0


@dataclass
class RetrievalResult:
    bundle: EvidenceBundle
    candidates: list[FusedCandidate]
    trace: RetrievalTrace

    @property
    def refs(self) -> list[str]:
        return [candidate.ref for candidate in self.candidates]


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        *,
        text_embedder: TextEmbedder | None = None,
        visual_encoder: VisualEncoder | None = None,
        lexical: Bm25Index | None = None,
        reranker: Reranker | None = None,
        max_evidence_chars: int = 8500,
        max_images: int = 1,
        branch_limit: int = 20,
        rerank_limit: int = 30,
    ) -> None:
        self.store = store
        self.text_embedder = text_embedder
        self.visual_encoder = visual_encoder
        self.lexical = lexical
        self.reranker = reranker or NoReranker()
        self.max_evidence_chars = max_evidence_chars
        self.max_images = max_images
        self.branch_limit = branch_limit
        self.rerank_limit = rerank_limit

    # --- the pipeline -----------------------------------------------------

    def retrieve(
        self,
        question: str,
        principal: Principal,
        *,
        visual_query: str | None = None,
        top_k: int = 8,
    ) -> RetrievalResult:
        started = time.perf_counter()

        # 1. Identity and permissions, before anything is searched.
        if not principal.can_read_anything:
            raise AccessDenied(
                f"{principal.user_id} n'appartient a aucun groupe : aucun document accessible"
            )
        groups = sorted(principal.groups)
        families = sorted(principal.families) if principal.families else None
        query_filter = access_filter(groups, families=families)

        # 2. Branches, each restricted to the authorised corpus.
        # Every branch is optional. Beyond flexibility this is a memory
        # concession: BGE-M3 and the reranker together exceed what some
        # machines can hold, and a lexical-plus-reranker configuration still
        # answers while fitting.
        branches: dict[str, list] = {}
        if self.text_embedder is not None:
            branches[BRANCH_DENSE] = self.store.search(
                Space.TEXT,
                self.text_embedder.encode_query(question),
                limit=self.branch_limit,
                query_filter=query_filter,
            )
        if self.lexical is not None:
            branches[BRANCH_LEXICAL] = self.lexical.search(
                question, limit=self.branch_limit, groups=groups, families=families
            )
        if not branches and self.visual_encoder is None:
            raise ValueError("Aucune branche de recherche active")
        if self.visual_encoder is not None:
            branches[BRANCH_VISUAL] = self.store.search(
                Space.VISUAL,
                self.visual_encoder.encode_query(visual_query or question),
                limit=self.branch_limit,
                query_filter=query_filter,
            )

        trace = RetrievalTrace(
            branch_counts={name: len(hits) for name, hits in branches.items()},
            reranker=self.reranker.name,
        )

        # 3. Fuse ranks, not scores.
        fused = reciprocal_rank_fusion(branches, limit=self.rerank_limit)
        trace.fused_count = len(fused)

        # 4. Rerank the survivors.
        reranked = self.reranker.rerank(question, fused, limit=top_k)
        trace.reranked_count = len(reranked)

        # 5. Bound the evidence and hand phase 5 a contract object.
        bundle, dropped = self._build_bundle(question, visual_query, reranked)
        trace.selected_count = len(bundle.items)
        trace.dropped_for_budget = dropped
        trace.elapsed_ms = (time.perf_counter() - started) * 1000

        return RetrievalResult(bundle=bundle, candidates=reranked, trace=trace)

    # --- evidence ---------------------------------------------------------

    def _build_bundle(
        self,
        question: str,
        visual_query: str | None,
        candidates: list[FusedCandidate],
    ) -> tuple[EvidenceBundle, int]:
        items: list[Evidence] = []
        used_chars = 0
        images = 0
        dropped = 0

        for candidate in candidates:
            hit = candidate.hit
            is_visual = bool(hit.image_uri)
            excerpt = hit.text or None

            if is_visual and images >= self.max_images:
                # Budget is on images sent to the model; the caption may still
                # travel as text, so the evidence is not lost entirely.
                if not excerpt:
                    dropped += 1
                    continue
                is_visual = False

            if excerpt is not None:
                remaining = self.max_evidence_chars - used_chars
                if remaining <= 0:
                    dropped += 1
                    continue
                excerpt = excerpt[:remaining]
                used_chars += len(excerpt)
            elif not is_visual:
                dropped += 1
                continue

            items.append(
                Evidence(
                    document_id=hit.document_id,
                    page=hit.page,
                    element_id=hit.element_id,
                    modality=hit.modality,
                    excerpt=excerpt,
                    image_uri=hit.image_uri if is_visual else None,
                    branch=_branch_of(candidate),
                    rank=len(items) + 1,
                    raw_score=hit.score,
                    rrf_score=candidate.score,
                    rerank_score=candidate.score if self.reranker.name != "none" else None,
                )
            )
            if is_visual:
                images += 1

        bundle = EvidenceBundle(
            question=question, visual_query=visual_query, items=tuple(items)
        )
        bundle.enforce_budget(
            max_chars=self.max_evidence_chars, max_images=self.max_images
        )
        return bundle, dropped
