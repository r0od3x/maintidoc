"""Ingestion command line.

    uv run python -m maintidoc.ingestion ingest
    uv run python -m maintidoc.ingestion ingest --document SAN-MAN-001 --force
    uv run python -m maintidoc.ingestion stats
    uv run python -m maintidoc.ingestion show SAN-MAN-001 --page 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maintidoc.config import CORPUS_DIR, get_settings
from maintidoc.ingestion.manifest import CorpusManifest
from maintidoc.ingestion.ocr import default_engine
from maintidoc.ingestion.pipeline import IngestionPipeline
from maintidoc.ingestion.store import CorpusStore

DEFAULT_MANIFEST = CORPUS_DIR / "manifest.yaml"
DEFAULT_DERIVED = CORPUS_DIR / "derived"


def _pipeline(args: argparse.Namespace) -> tuple[IngestionPipeline, CorpusManifest, CorpusStore]:
    manifest = CorpusManifest.from_file(args.manifest)
    store = CorpusStore(args.derived)
    settings = get_settings()
    ocr = default_engine(
        enabled=args.ocr,
        binary=args.tesseract or settings.tesseract_cmd,
        tessdata_dir=settings.tessdata_dir,
    )
    if args.ocr and not ocr.available():
        print(
            "OCR demande mais indisponible : binaire Tesseract introuvable ou "
            "non executable (installation incomplete ?). Poursuite sans OCR."
        )
    pipeline = IngestionPipeline(store, corpus_root=args.manifest.parent, ocr=ocr)
    return pipeline, manifest, store


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline, manifest, _ = _pipeline(args)
    results = pipeline.ingest_corpus(manifest, force=args.force, only=args.document)

    failures = 0
    for result in results:
        counts = ", ".join(f"{k}={v}" for k, v in result.element_counts.items()) or "-"
        print(f"[{result.status.value:>8}] {result.document_id:<20} {counts}")
        if result.error:
            failures += 1
            print(f"           erreur : {result.error}")
        for warning in result.warnings:
            print(f"           ! {warning}")

    total = sum(r.total_elements for r in results)
    print(
        f"\n{len(results)} document(s), {total} elements, "
        f"{len(manifest.families)} famille(s) d'equipement."
    )
    return 1 if failures else 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = CorpusStore(args.derived)
    records = store.records()
    if not records:
        print("Corpus vide : lancer `ingest` d'abord.")
        return 1

    totals: dict[str, int] = {}
    print(f"{'document':<20} {'pages':>5} {'elements':>9}  modalites")
    for record in sorted(records, key=lambda r: r.document_id):
        detail = ", ".join(f"{k}={v}" for k, v in record.element_counts.items())
        print(
            f"{record.document_id:<20} {record.page_count:>5} "
            f"{record.total_elements:>9}  {detail}"
        )
        for key, value in record.element_counts.items():
            totals[key] = totals.get(key, 0) + value

    print(f"\nTotal : {sum(totals.values())} elements — " +
          ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    settings = get_settings()
    print(f"Version de corpus configuree : {settings.corpus_version}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = CorpusStore(args.derived)
    elements = store.read_elements(args.document_id)
    if not elements:
        print(f"Aucun element pour {args.document_id}.")
        return 1
    for element in elements:
        if args.page and element.page != args.page:
            continue
        head = f"{element.ref.ref:<34} {element.modality.value:<8} {element.extraction.value:<7}"
        extra = f"figure={element.figure_id} " if element.figure_id else ""
        body = (element.text or "").replace("\n", " ")[: args.width]
        print(f"{head} {extra}{body}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.ingestion", description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingerer le corpus declare")
    ingest.add_argument("--document", help="Limiter a un document_id")
    ingest.add_argument("--force", action="store_true", help="Reingerer meme si inchange")
    ingest.add_argument("--ocr", action="store_true", help="Activer l'OCR local si disponible")
    ingest.add_argument("--tesseract", type=Path, help="Chemin du binaire tesseract")
    ingest.set_defaults(func=cmd_ingest)

    stats = subparsers.add_parser("stats", help="Inventaire du corpus derive")
    stats.set_defaults(func=cmd_stats)

    show = subparsers.add_parser("show", help="Lister les elements d'un document")
    show.add_argument("document_id")
    show.add_argument("--page", type=int)
    show.add_argument("--width", type=int, default=90)
    show.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
