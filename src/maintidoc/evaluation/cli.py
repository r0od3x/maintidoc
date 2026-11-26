"""Evaluation command line. Headless, for CI.

    uv run python -m maintidoc.evaluation build
    uv run python -m maintidoc.evaluation run --lean
    uv run python -m maintidoc.evaluation report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from maintidoc.config import CORPUS_DIR, get_settings
from maintidoc.evaluation.dataset import Dataset, build_dataset
from maintidoc.evaluation.runner import EvaluationRunner, Report

EVAL_DIR = Path("eval")
DEFAULT_FACTS = EVAL_DIR / "datasets" / "corpus_facts.json"
DEFAULT_DATASET = EVAL_DIR / "datasets" / "scenarios.json"
DEFAULT_REPORT = EVAL_DIR / "reports" / "latest.json"


def cmd_build(args: argparse.Namespace) -> int:
    if not args.facts.is_file():
        print(f"Verites de terrain absentes : {args.facts}")
        print("Lancer : uv run python -m maintidoc.synth generate")
        return 1
    dataset = build_dataset(
        args.facts, corpus_version=get_settings().corpus_version, seed=args.seed
    )
    dataset.save(args.dataset)
    print(f"{len(dataset.scenarios)} scenarios ecrits dans {args.dataset}")
    for category, count in sorted(dataset.counts.items()):
        print(f"  {category:<24} {count}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.dataset.is_file():
        print(f"Jeu de test absent : {args.dataset}. Lancer `build` d'abord.")
        return 1
    dataset = Dataset.load(args.dataset)

    from maintidoc.api.deps import Runtime

    runtime = Runtime.lean(derived=args.derived) if args.lean else Runtime.full(
        derived=args.derived
    )
    answerer = None if args.retrieval_only else runtime.answerer
    runner = EvaluationRunner(
        runtime.retriever,
        answerer=answerer,
        service=runtime.service,
        top_k=args.top_k,
    )
    report = runner.run(dataset)
    runtime.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"\nRapport : {args.report}")

    failures = sum(1 for v in report.verdicts if v.passed is False)
    return 1 if failures and args.strict else 0


def cmd_report(args: argparse.Namespace) -> int:
    if not args.report.is_file():
        print(f"Aucun rapport : {args.report}")
        return 1
    _print_summary(Report.model_validate_json(args.report.read_text(encoding="utf-8")))
    return 0


def _print_summary(report: Report) -> None:
    summary = report.summary
    print(f"Jeu de test : {report.dataset} | corpus {report.corpus_version}")
    print(f"Generateur  : {report.generator}")
    print(f"Scenarios   : {summary.get('scenarios', 0)}\n")

    print("Recherche :", json.dumps(summary.get("recherche", {}), ensure_ascii=False))
    print("Latence   :", json.dumps(summary.get("latence", {}), ensure_ascii=False))
    print()
    print(f"{'categorie':<26}{'total':>7}{'evalues':>9}{'reussis':>9}{'taux':>8}")
    for category, stats in summary.get("par_categorie", {}).items():
        rate = stats.get("taux")
        shown = f"{rate:.0%}" if isinstance(rate, (int, float)) else "n/a"
        print(
            f"{category:<26}{stats['total']:>7}{stats['evalues']:>9}"
            f"{stats['reussis']:>9}{shown:>8}"
        )
    print()
    print(f"Fuites de controle d'acces : {summary.get('fuites_acces')}")
    print(f"Injections confinees       : {summary.get('injections_confinees')}")
    print(f"Injections suivies         : {summary.get('injections_suivies')}")
    rate = summary.get("taux_citations_soutenues")
    print(
        "Taux de citations soutenues : "
        + (f"{rate:.0%}" if rate is not None else "non mesure (aucun generateur)")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.evaluation", description=__doc__)
    parser.add_argument("--derived", type=Path, default=CORPUS_DIR / "derived")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Construire le jeu de test")
    build.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    build.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    build.add_argument("--seed", type=int, default=20261120)
    build.set_defaults(func=cmd_build)

    run = subparsers.add_parser("run", help="Executer le jeu de test")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--lean", action="store_true", help="Sans modeles lourds")
    run.add_argument("--retrieval-only", action="store_true")
    run.add_argument("--strict", action="store_true", help="Code de sortie 1 si echec")
    run.set_defaults(func=cmd_run)

    show = subparsers.add_parser("report", help="Afficher un rapport existant")
    show.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    show.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
