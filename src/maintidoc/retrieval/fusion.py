"""Reciprocal Rank Fusion across the retrieval branches.

RRF combines *ranks*, not scores. That matters here: BGE-M3 cosines, CLIP
cosines and BM25 scores live on incompatible scales, and normalising them
against each other would invent a comparison that does not exist.

    RRF(d) = sum over branches of 1 / (k + rank(d))

Fusion is at **element** level. The prototype fused at page level, which was
reasonable for one nine-page manual — it pulled a figure and its explanation
together. Across 41 documents it is too coarse: a page holds several unrelated
elements, and promoting all of them because one matched floods the evidence
budget with noise.

The page is kept as a weaker signal instead. An element whose page is
independently supported by another branch gets a small bonus, which recovers
the figure-plus-caption effect without the collateral damage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from maintidoc.retrieval.store import SearchHit

DEFAULT_K = 60
DEFAULT_PAGE_BONUS = 0.25


@dataclass
class FusedCandidate:
    """One element, with where it came from and why it ranks where it does."""

    hit: SearchHit
    score: float
    branches: tuple[str, ...]
    ranks: dict[str, int] = field(default_factory=dict)
    page_bonus: float = 0.0
    rank: int = 0

    @property
    def ref(self) -> str:
        return self.hit.ref

    @property
    def from_several_branches(self) -> bool:
        return len(self.branches) > 1


def reciprocal_rank_fusion(
    branches: dict[str, list[SearchHit]],
    *,
    k: int = DEFAULT_K,
    page_bonus: float = DEFAULT_PAGE_BONUS,
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Fuse ranked lists into one ordering.

    `page_bonus` is expressed as a fraction of a single branch's contribution
    at rank 1, so it stays small relative to a genuine second-branch hit and
    cannot by itself outrank an element two branches agreed on.
    """
    contributions: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    best_hit: dict[str, SearchHit] = {}
    page_support: dict[tuple[str, int], set[str]] = defaultdict(set)

    for branch, hits in branches.items():
        for hit in hits:
            if not hit.ref:
                continue
            contributions[hit.ref] += 1.0 / (k + hit.rank)
            ranks[hit.ref][branch] = hit.rank
            page_support[(hit.document_id, hit.page)].add(branch)
            # Keep the richest payload seen for this element.
            current = best_hit.get(hit.ref)
            if current is None or (hit.text and not current.text):
                best_hit[hit.ref] = hit

    unit = 1.0 / (k + 1)
    candidates: list[FusedCandidate] = []
    for ref, base in contributions.items():
        hit = best_hit[ref]
        own_branches = set(ranks[ref])
        supporting = page_support[(hit.document_id, hit.page)] - own_branches
        bonus = page_bonus * unit * len(supporting)
        candidates.append(
            FusedCandidate(
                hit=hit,
                score=base + bonus,
                branches=tuple(sorted(own_branches)),
                ranks=dict(ranks[ref]),
                page_bonus=bonus,
            )
        )

    candidates.sort(key=lambda c: (-c.score, c.ref))
    for position, candidate in enumerate(candidates, start=1):
        candidate.rank = position
    return candidates[:limit] if limit else candidates
