"""Phase 9 — jeu de test, metriques et tests d'injection."""

from maintidoc.evaluation.dataset import (
    TARGETS,
    Category,
    Dataset,
    Injection,
    InjectionChannel,
    Scenario,
    build_dataset,
)
from maintidoc.evaluation.metrics import (
    LatencyMetrics,
    RetrievalMetrics,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)
from maintidoc.evaluation.runner import (
    EvaluationRunner,
    PoisonedService,
    Report,
    Verdict,
)

__all__ = [
    "TARGETS",
    "Category",
    "Dataset",
    "EvaluationRunner",
    "Injection",
    "InjectionChannel",
    "LatencyMetrics",
    "PoisonedService",
    "Report",
    "RetrievalMetrics",
    "Scenario",
    "Verdict",
    "build_dataset",
    "mrr_at_k",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
]
