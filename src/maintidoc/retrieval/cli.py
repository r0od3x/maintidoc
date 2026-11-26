"""Indexing and search command line.

    uv run python -m maintidoc.retrieval build
    uv run python -m maintidoc.retrieval build --fake          # hors ligne
    uv run python -m maintidoc.retrieval stats
    uv run python -m maintidoc.retrieval search "voyant orange sur SAN-LAB 2000"
    uv run python -m maintidoc.retrieval search "schema des chemins redondants" --space visual

`--fake` swaps in deterministic pseudo-embeddings. It exercises the plumbing
without downloading anything, and tells you nothing about retrieval quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maintidoc.config import CORPUS_DIR, get_settings
from maintidoc.contracts import Principal
from maintidoc.ingestion.store import CorpusStore
from maintidoc.retrieval.embeddings import build_text_embedder, build_visual_encoder
from maintidoc.retrieval.indexer import Indexer, IndexJournal
from maintidoc.retrieval.lexical import Bm25Index
from maintidoc.retrieval.rerank import NoReranker, build_reranker
from maintidoc.retrieval.retriever import AccessDenied, Retriever
from maintidoc.retrieval.store import Space, VectorStore, access_filter

DEFAULT_DERIVED = CORPUS_DIR / "derived"


def _vector_store(args: argparse.Namespace) -> VectorStore:
    settings = get_settings()
    if args.qdrant_url:
        return VectorStore.remote(args.qdrant_url)
    if settings.qdrant_local:
        return VectorStore.local(args.qdrant_path or (args.derived / "qdrant"))
    return VectorStore.remote(settings.qdrant_url)


def _journal(args: argparse.Namespace) -> IndexJournal:
    """The journal lives beside the index it describes.

    Pointing `--qdrant-path` elsewhere — a scratch index, an experiment — must
    not append to the journal of the real one, or a throwaway run would be
    mistaken for the index actually in use.
    """
    if args.qdrant_path:
        return IndexJournal(args.qdrant_path.parent / "index_runs.jsonl")
    return IndexJournal(args.derived / "index_runs.jsonl")


def cmd_build(args: argparse.Namespace) -> int:
    settings = get_settings()
    corpus = CorpusStore(args.derived)
    elements = list(corpus.iter_all_elements())
    if not elements:
        print("Corpus derive vide : lancer `python -m maintidoc.ingestion ingest`.")
        return 1

    print(f"Chargement des modeles ({'factices' if args.fake else 'reels'})...")
    text_embedder = build_text_embedder(settings, fake=args.fake)
    visual_encoder = None if args.no_visual else build_visual_encoder(settings, fake=args.fake)

    store = _vector_store(args)
    indexer = Indexer(
        store,
        text_embedder=text_embedder,
        visual_encoder=visual_encoder,
        content=corpus.content,
    )
    run = indexer.build(
        elements,
        index_version=args.index_version or settings.index_version,
        corpus_version=settings.corpus_version,
        recreate=args.recreate,
    )
    _journal(args).append(run)

    print(f"Index texte   : {run.text_points} points ({run.text_model}, {run.text_dimension}d)")
    if run.visual_model:
        print(
            f"Index visuel  : {run.visual_points} points "
            f"({run.visual_model}, {run.visual_dimension}d)"
        )
    print(f"Documents     : {run.documents}")
    print(f"Duree         : {run.duration_seconds:.1f} s")
    if run.skipped_images:
        print(f"Images ignorees : {len(run.skipped_images)}")
    store.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = _vector_store(args)
    print(f"Collection texte  : {store.count(Space.TEXT)} points")
    print(f"Collection visuel : {store.count(Space.VISUAL)} points")
    store.close()

    latest = _journal(args).latest()
    if latest is None:
        print("Aucune execution d'indexation enregistree.")
        return 0
    print(
        f"\nDerniere indexation : index {latest.index_version}, "
        f"corpus {latest.corpus_version}, {latest.finished_at:%Y-%m-%d %H:%M}"
    )
    print(f"  modele texte  : {latest.text_model} ({latest.text_dimension}d)")
    print(f"  modele visuel : {latest.visual_model or 'aucun'}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    space = Space.VISUAL if args.space == "visual" else Space.TEXT
    if space is Space.VISUAL:
        encoder = build_visual_encoder(settings, fake=args.fake)
    else:
        encoder = build_text_embedder(settings, fake=args.fake)

    store = _vector_store(args)
    hits = store.search(
        space,
        encoder.encode_query(args.query),
        limit=args.limit,
        query_filter=access_filter(args.groups, families=args.family),
    )
    if not hits:
        print("Aucun resultat (index vide, ou aucun document accessible a ces groupes).")
        store.close()
        return 1

    for hit in hits:
        figure = f" [{hit.figure_id}]" if hit.figure_id else ""
        print(f"{hit.rank:>2}. {hit.score:.4f}  {hit.ref}{figure}  ({hit.modality})")
        if hit.text:
            print(f"      {' '.join(hit.text.split())[:110]}")
    store.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Full pipeline: permissions, three branches, fusion, reranking, evidence."""
    settings = get_settings()
    corpus = CorpusStore(args.derived)
    elements = list(corpus.iter_all_elements())

    principal = Principal(
        user_id=args.user,
        groups=frozenset(args.groups or ()),
        families=frozenset(args.family) if args.family else None,
    )
    # Loading every model at once exceeds the memory of some machines. Each
    # piece is built separately so the failure names the component that did
    # not fit, instead of a traceback from somewhere inside torch.
    try:
        text_embedder = (
            None if args.no_dense else build_text_embedder(settings, fake=args.fake)
        )
        visual_encoder = (
            None if args.no_visual else build_visual_encoder(settings, fake=args.fake)
        )
        reranker = NoReranker() if args.no_rerank else build_reranker(settings)
    except (MemoryError, OSError) as exc:
        print(
            "Memoire insuffisante pour charger tous les modeles "
            f"({type(exc).__name__}). Relancer avec une configuration allegee :"
        )
        print("  --no-dense --no-visual     (lexical + reranker)")
        print("  --no-rerank                (dense + visuel, sans reranker)")
        return 3

    retriever = Retriever(
        _vector_store(args),
        text_embedder=text_embedder,
        visual_encoder=visual_encoder,
        lexical=None if args.no_lexical else Bm25Index.build(elements),
        reranker=reranker,
        max_evidence_chars=settings.max_evidence_chars,
        max_images=settings.max_images,
    )
    try:
        result = retriever.retrieve(args.question, principal, top_k=args.limit)
    except AccessDenied as exc:
        print(f"Acces refuse : {exc}")
        return 2

    trace = result.trace
    branches = ", ".join(f"{k}={v}" for k, v in sorted(trace.branch_counts.items()))
    print(f"Branches : {branches}")
    print(
        f"Fusion : {trace.fused_count} candidats -> reranking ({trace.reranker}) "
        f"-> {trace.selected_count} preuves, {trace.elapsed_ms:.0f} ms"
    )
    for item in result.bundle.items:
        figure = " [image]" if item.image_uri else ""
        print(f"{item.rank:>2}. {item.ref}{figure}  ({item.modality.value})")
        if item.excerpt:
            print(f"      {' '.join(item.excerpt.split())[:120]}")
    print()
    print(f"Budget : {result.bundle.total_excerpt_chars} caracteres, "
          f"{len(result.bundle.image_items)} image(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.retrieval", description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--qdrant-url", help="Utiliser un serveur Qdrant plutot que le mode local")
    parser.add_argument(
        "--qdrant-path", type=Path, help="Emplacement du stockage Qdrant local"
    )
    parser.add_argument("--fake", action="store_true", help="Embeddings deterministes hors ligne")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Construire les index")
    build.add_argument("--recreate", action="store_true", help="Recreer les collections")
    build.add_argument("--no-visual", action="store_true", help="Index texte uniquement")
    build.add_argument("--index-version", help="Etiquette de version d'index")
    build.set_defaults(func=cmd_build)

    stats = subparsers.add_parser("stats", help="Etat des collections")
    stats.set_defaults(func=cmd_stats)

    search = subparsers.add_parser("search", help="Rechercher dans un index")
    search.add_argument("query")
    search.add_argument("--space", choices=("text", "visual"), default="text")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--groups", nargs="*", help="Groupes d'acces de l'utilisateur")
    search.add_argument("--family", nargs="*", help="Restreindre a des familles")
    search.set_defaults(func=cmd_search)

    ask = subparsers.add_parser("ask", help="Pipeline complet de recuperation")
    ask.add_argument("question")
    ask.add_argument("--user", default="tech-demo")
    ask.add_argument("--groups", nargs="*", default=["maintenance", "maintenance_san"])
    ask.add_argument("--family", nargs="*")
    ask.add_argument("--limit", type=int, default=6)
    ask.add_argument("--no-rerank", action="store_true", help="Baseline sans reranker")
    ask.add_argument("--no-visual", action="store_true")
    ask.add_argument("--no-dense", action="store_true", help="Desactiver la branche BGE-M3")
    ask.add_argument("--no-lexical", action="store_true")
    ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
