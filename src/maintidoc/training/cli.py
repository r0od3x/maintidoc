"""Reranker adaptation pipeline.

    uv run python -m maintidoc.training mine
    uv run python -m maintidoc.training train --dry-run
    uv run python -m maintidoc.training compare --lean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maintidoc.config import CORPUS_DIR, get_settings
from maintidoc.evaluation.dataset import Dataset
from maintidoc.ingestion.store import CorpusStore
from maintidoc.training.compare import ComparisonHarness, format_comparison
from maintidoc.training.tracking import tracking_run
from maintidoc.training.train import TrainingConfig, train_reranker, write_manifest
from maintidoc.training.triplets import TripletSplit, TripletStats, build_split

TRAIN_DIR = Path("eval") / "training"
DEFAULT_FACTS = Path("eval") / "datasets" / "corpus_facts.json"
DEFAULT_SPLIT = TRAIN_DIR / "triplets.json"
DEFAULT_MANIFEST = TRAIN_DIR / "training_manifest.json"
DEFAULT_COMPARISON = Path("eval") / "reports" / "comparison.json"
DEFAULT_SCENARIOS = Path("eval") / "datasets" / "scenarios.json"


def cmd_mine(args: argparse.Namespace) -> int:
    if not args.facts.is_file():
        print(f"Verites de terrain absentes : {args.facts}")
        print("Lancer : uv run python -m maintidoc.synth generate")
        return 1
    elements = list(CorpusStore(args.derived).iter_all_elements())
    if not elements:
        print("Corpus derive vide : lancer `python -m maintidoc.ingestion ingest`.")
        return 1

    split = build_split(
        args.facts, elements, negatives_per_positive=args.negatives, seed=args.seed
    )
    split.save(args.split)
    stats = TripletStats.of(split)

    print(f"{stats.total} triplets ecrits dans {args.split}")
    print(f"  negatifs difficiles : {stats.hard} ({stats.hard_ratio:.0%})")
    print(f"  negatifs faciles    : {stats.easy}")
    print(f"  modeles / familles  : {stats.models} / {stats.families}")
    print(f"  repartition         : {stats.sizes}")
    if stats.leaks:
        print(f"  FUITE entre splits  : {stats.leaks}")
        return 1
    print("  aucune fuite entre splits")

    target = args.target
    if stats.total < target:
        print(
            f"\nAttention : {stats.total} triplets, le cahier des charges en "
            f"demande au moins {target}. Augmenter --negatives."
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if not args.split.is_file():
        print(f"Triplets absents : {args.split}. Lancer `mine` d'abord.")
        return 1
    split = TripletSplit.load(args.split)
    settings = get_settings()
    config = TrainingConfig(
        base_model=args.base_model or settings.reranker_model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    with tracking_run(f"train-{config.base_model}", enabled=not args.no_tracking) as tracker:
        tracker.log_params(config.as_params())
        stats = TripletStats.of(split)
        tracker.log_params({"triplets": stats.total, "hard_ratio": round(stats.hard_ratio, 3)})
        try:
            result = train_reranker(split, config, dry_run=args.dry_run)
        except ValueError as exc:
            print(f"Entrainement refuse : {exc}")
            return 1
        tracker.log_metrics({"triplets_train": len(split.train)})
        write_manifest(result, args.manifest)
        if not tracker.enabled:
            print(f"Suivi MLflow inactif : {getattr(tracker, 'reason', 'inconnu')}")

    print(f"Entraine : {result.trained}")
    print(f"Detail   : {result.detail}")
    print(f"Manifeste: {args.manifest}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if not args.scenarios.is_file():
        print(f"Jeu de test absent : {args.scenarios}. Lancer `evaluation build`.")
        return 1
    dataset = Dataset.load(args.scenarios)
    settings = get_settings()

    from maintidoc.api.deps import Runtime

    runtime = Runtime.lean(derived=args.derived) if args.lean else Runtime.full(
        derived=args.derived
    )

    base = None
    adapted = None
    if not args.lean:
        from maintidoc.retrieval.rerank import CrossEncoderReranker

        try:
            base = CrossEncoderReranker(settings.reranker_model)
        except (MemoryError, OSError, ImportError) as exc:
            print(f"Reranker de base indisponible : {type(exc).__name__}")
        if args.adapted and Path(args.adapted).exists():
            try:
                adapted = CrossEncoderReranker(str(args.adapted))
            except (MemoryError, OSError, ImportError) as exc:
                print(f"Reranker adapte indisponible : {type(exc).__name__}")
        else:
            print("Aucun reranker adapte fourni : bras 'adapted' non execute.")

    harness = ComparisonHarness(
        runtime.retriever, dataset, service=runtime.service, top_k=args.top_k
    )
    with tracking_run("comparison", enabled=not args.no_tracking) as tracker:
        comparison = harness.compare(base=base, adapted=adapted)
        for arm in comparison.arms:
            if arm.available:
                tracker.log_metrics(
                    {f"{arm.name}_{k}": v for k, v in arm.metrics.items()
                     if isinstance(v, (int, float))}
                )
        comparison.save(args.comparison)
        tracker.log_artifact(args.comparison)
    runtime.close()

    print(format_comparison(comparison))
    print(f"\nRapport : {args.comparison}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.training", description=__doc__)
    parser.add_argument("--derived", type=Path, default=CORPUS_DIR / "derived")
    parser.add_argument("--no-tracking", action="store_true", help="Desactiver MLflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser("mine", help="Constituer les triplets")
    mine.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    mine.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    mine.add_argument("--negatives", type=int, default=13)
    mine.add_argument("--target", type=int, default=800)
    mine.add_argument("--seed", type=int, default=20261120)
    mine.set_defaults(func=cmd_mine)

    train = subparsers.add_parser("train", help="Adapter le reranker")
    train.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    train.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    train.add_argument("--output", type=Path, default=Path("models/reranker-maintidoc"))
    train.add_argument("--base-model", help="Modele de depart (defaut : configuration)")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--dry-run", action="store_true", help="Preparer sans entrainer")
    train.set_defaults(func=cmd_train)

    compare = subparsers.add_parser("compare", help="Comparer les trois bras")
    compare.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    compare.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    compare.add_argument("--adapted", type=Path, help="Chemin du reranker adapte")
    compare.add_argument("--top-k", type=int, default=8)
    compare.add_argument("--lean", action="store_true")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
