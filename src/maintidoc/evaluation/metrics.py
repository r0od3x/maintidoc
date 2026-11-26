"""Retrieval and answer metrics.

Implemented rather than imported so the definitions are visible and arguable —
in a report, "nDCG@5" means whatever the code computed, and a reader should be
able to check it.

Graded relevance is deliberately not used: ground truth here is binary (the
element either holds the fact or it does not), so a graded gain would invent
distinctions the corpus does not contain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Share of relevant items found in the top k."""
    if not relevant:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    if k <= 0:
        return float("nan")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in set(relevant)) / len(top)


def mrr_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant item, 0 if none in the top k."""
    wanted = set(relevant)
    for position, item in enumerate(retrieved[:k], start=1):
        if item in wanted:
            return 1.0 / position
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Normalised discounted cumulative gain, binary relevance."""
    if not relevant:
        return float("nan")
    wanted = set(relevant)
    gains = [1.0 if item in wanted else 0.0 for item in retrieved[:k]]
    ideal = [1.0] * min(len(wanted), k)
    best = dcg(ideal)
    return dcg(gains) / best if best else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Returns NaN for an empty sample."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def mean(values: Sequence[float]) -> float:
    usable = [v for v in values if not math.isnan(v)]
    return sum(usable) / len(usable) if usable else float("nan")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_5: float
    ndcg_at_5: float
    mrr_at_10: float
    precision_at_5: float
    scenarios: int

    @classmethod
    def over(
        cls, pairs: Sequence[tuple[Sequence[str], Sequence[str]]]
    ) -> RetrievalMetrics:
        """`pairs` is (retrieved refs, relevant refs) per scenario."""
        usable = [(r, g) for r, g in pairs if g]
        return cls(
            recall_at_5=mean([recall_at_k(r, g, 5) for r, g in usable]),
            ndcg_at_5=mean([ndcg_at_k(r, g, 5) for r, g in usable]),
            mrr_at_10=mean([mrr_at_k(r, g, 10) for r, g in usable]),
            precision_at_5=mean([precision_at_k(r, g, 5) for r, g in usable]),
            scenarios=len(usable),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "recall@5": round(self.recall_at_5, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "mrr@10": round(self.mrr_at_10, 4),
            "precision@5": round(self.precision_at_5, 4),
            "scenarios": self.scenarios,
        }


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    p50_ms: float
    p95_ms: float
    max_ms: float
    samples: int

    @classmethod
    def over(cls, durations: Sequence[float]) -> LatencyMetrics:
        return cls(
            p50_ms=percentile(durations, 0.50),
            p95_ms=percentile(durations, 0.95),
            max_ms=max(durations) if durations else float("nan"),
            samples=len(durations),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "max_ms": round(self.max_ms, 1),
            "samples": self.samples,
        }
