"""Phase 10 — adaptation du reranker, suivi d'experiences, livraison."""

from maintidoc.training.compare import (
    ArmResult,
    Comparison,
    ComparisonHarness,
    format_comparison,
)
from maintidoc.training.tracking import (
    MlflowTracker,
    NullTracker,
    Tracker,
    sanitise_metric_name,
    tracking_run,
)
from maintidoc.training.train import (
    TrainingConfig,
    TrainingResult,
    to_training_pairs,
    train_reranker,
    write_manifest,
)
from maintidoc.training.triplets import (
    Triplet,
    TripletSplit,
    TripletStats,
    build_split,
    collect_passages,
    mine_triplets,
    split_by_model,
)

__all__ = [
    "ArmResult",
    "Comparison",
    "ComparisonHarness",
    "MlflowTracker",
    "NullTracker",
    "Tracker",
    "TrainingConfig",
    "TrainingResult",
    "Triplet",
    "TripletSplit",
    "TripletStats",
    "build_split",
    "collect_passages",
    "format_comparison",
    "mine_triplets",
    "split_by_model",
    "to_training_pairs",
    "tracking_run",
    "sanitise_metric_name",
    "train_reranker",
    "write_manifest",
]
