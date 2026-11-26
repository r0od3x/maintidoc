"""Synthetic corpus generator.

    uv run python -m maintidoc.synth generate
    uv run python -m maintidoc.synth generate --seed 7 --no-manifest

Dev tool, not part of the serving runtime: it authors the corpus the pipeline
then ingests like any other document.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from maintidoc.config import CORPUS_DIR
from maintidoc.synth.catalog import ALL_MODELS
from maintidoc.synth.generate import generate_corpus, write_facts, write_manifest

HAND_WRITTEN = {"SAN-MAN-001"}


def _preserved_entries(manifest_path: Path) -> list[dict]:
    """Keep manifest entries that were not produced by this generator."""
    if not manifest_path.is_file():
        return []
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return [
        entry for entry in data.get("documents", []) if entry.get("document_id") in HAND_WRITTEN
    ]


def cmd_generate(args: argparse.Namespace) -> int:
    output = args.corpus / "raw" / "genere"
    documents = generate_corpus(output, seed=args.seed)

    by_family: dict[str, int] = {}
    pages = 0
    for document in documents:
        by_family[document.family] = by_family.get(document.family, 0) + 1
        pages += document.pages

    print(f"{len(documents)} documents generes, {pages} pages, graine {args.seed}")
    for family, count in sorted(by_family.items()):
        print(f"  {family:<12} {count} documents")

    if args.manifest:
        manifest_path = args.corpus / "manifest.yaml"
        write_manifest(
            documents,
            manifest_path,
            args.corpus,
            corpus_version=args.corpus_version,
            keep=_preserved_entries(manifest_path),
        )
        print(f"Manifeste ecrit : {manifest_path}")

    count = write_facts(documents, args.facts)
    print(f"Verites de terrain : {count} faits dans {args.facts}")
    print(f"Modeles couverts : {len(ALL_MODELS)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.synth", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generer le corpus synthetique")
    generate.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    generate.add_argument("--seed", type=int, default=20261119)
    generate.add_argument("--corpus-version", default="2026.11")
    generate.add_argument("--facts", type=Path, default=Path("eval/datasets/corpus_facts.json"))
    generate.add_argument(
        "--no-manifest",
        dest="manifest",
        action="store_false",
        help="Ne pas reecrire corpus/manifest.yaml",
    )
    generate.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
