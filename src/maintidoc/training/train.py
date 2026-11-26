"""Fine-tuning the cross-encoder reranker.

Written to run, and deliberately not run here: `bge-reranker-v2-m3` is 568M
parameters and this machine already cannot hold it alongside BGE-M3 (phase 4).
Training it on CPU is not a question of patience.

`--base-model` exists so the pipeline can be exercised end to end on a small
model, which is what the tests do. The honest consequence is stated in the
README: the training code is tested, the training is not done, and no gain is
claimed.

Everything is logged to MLflow: parameters, dataset statistics, metrics, and
the split itself. A fine-tuning run whose data cannot be reconstructed is not
an experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from maintidoc.training.triplets import TripletSplit, TripletStats


@dataclass
class TrainingConfig:
    base_model: str = "BAAI/bge-reranker-v2-m3"
    output_dir: Path = Path("models/reranker-maintidoc")
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 512
    seed: int = 20261120
    device: str = "cpu"

    def as_params(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass
class TrainingResult:
    output_dir: Path
    config: TrainingConfig
    stats: TripletStats
    trained: bool
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def to_training_pairs(split: TripletSplit, part: str = "train") -> list[tuple[str, str, float]]:
    """Flatten triplets into labelled pairs for a cross-encoder.

    A cross-encoder scores one (query, passage) pair at a time, so each
    triplet becomes a positive pair and a negative pair.
    """
    pairs: list[tuple[str, str, float]] = []
    for triplet in getattr(split, part):
        pairs.append((triplet.query, triplet.positive, 1.0))
        pairs.append((triplet.query, triplet.negative, 0.0))
    return pairs


def train_reranker(
    split: TripletSplit, config: TrainingConfig, *, dry_run: bool = False
) -> TrainingResult:
    """Fine-tune, or explain precisely why it could not start."""
    stats = TripletStats.of(split)
    if stats.leaks:
        # Refuse rather than produce a number that cannot be trusted.
        raise ValueError(
            f"Fuite entre les splits pour les modeles {stats.leaks} : "
            "le gain mesure serait de la memorisation"
        )
    if not split.train:
        raise ValueError("Jeu d'entrainement vide")

    if dry_run:
        return TrainingResult(
            output_dir=config.output_dir,
            config=config,
            stats=stats,
            trained=False,
            detail="dry-run : donnees preparees, aucun entrainement lance",
        )

    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f"Dependances d'entrainement absentes : {exc}") from exc

    pairs = to_training_pairs(split, "train")
    examples = [InputExample(texts=[q, p], label=label) for q, p, label in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=config.batch_size)

    try:
        model = CrossEncoder(
            config.base_model,
            num_labels=1,
            max_length=config.max_length,
            device=config.device,
        )
        warmup = int(len(loader) * config.epochs * config.warmup_ratio)
        model.fit(
            train_dataloader=loader,
            epochs=config.epochs,
            warmup_steps=warmup,
            optimizer_params={"lr": config.learning_rate},
            output_path=str(config.output_dir),
            show_progress_bar=False,
        )
    except (MemoryError, OSError) as exc:
        # The known constraint on this machine, reported rather than crashed.
        return TrainingResult(
            output_dir=config.output_dir,
            config=config,
            stats=stats,
            trained=False,
            detail=(
                f"Entrainement impossible : {type(exc).__name__}. "
                f"{config.base_model} ne tient pas en memoire ici."
            ),
        )

    return TrainingResult(
        output_dir=config.output_dir,
        config=config,
        stats=stats,
        trained=True,
        detail=f"{len(examples)} paires, {config.epochs} epoques",
    )


def write_manifest(result: TrainingResult, path: Path) -> None:
    """Record what was trained, from what, so a run can be reproduced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trained": result.trained,
                "detail": result.detail,
                "finished_at": result.finished_at.isoformat(),
                "config": result.config.as_params(),
                "output_dir": str(result.output_dir),
                "dataset": result.stats.model_dump(mode="json"),
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
