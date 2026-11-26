"""End-to-end answering command.

    uv run python -m maintidoc.generation answer "Que signifie le voyant orange ... ?"
    uv run python -m maintidoc.generation answer "..." --no-dense --no-visual
    uv run python -m maintidoc.generation check          # etat du modele local

Runs retrieval (phase 4) then grounded generation (phase 5), and prints what
each verification concluded. An answer is labelled sourced only when both
checks ran and both passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maintidoc.config import CORPUS_DIR, get_settings
from maintidoc.contracts import Principal
from maintidoc.generation.answering import Answerer
from maintidoc.generation.client import OllamaVlmClient
from maintidoc.generation.grounding import LexicalGroundingChecker, LlmGroundingChecker
from maintidoc.ingestion.store import CorpusStore
from maintidoc.retrieval import (
    Bm25Index,
    NoReranker,
    Retriever,
    VectorStore,
    build_reranker,
    build_text_embedder,
    build_visual_encoder,
)
from maintidoc.retrieval.retriever import AccessDenied
from maintidoc.versions import capture_versions

DEFAULT_DERIVED = CORPUS_DIR / "derived"


def cmd_check(args: argparse.Namespace) -> int:
    settings = get_settings()
    client = OllamaVlmClient(settings.vlm_model, host=settings.ollama_host)
    installed = client.installed_models()
    if not installed:
        print(f"Ollama injoignable sur {settings.ollama_host}.")
        print("Installer Ollama, le demarrer, puis :")
        print(f"  ollama pull {settings.vlm_model}")
        return 1
    print(f"Modeles locaux : {', '.join(installed)}")
    if client.available():
        print(f"Modele configure disponible : {settings.vlm_model}")
        return 0
    print(f"Modele configure ABSENT : {settings.vlm_model}")
    print(f"  ollama pull {settings.vlm_model}")
    return 1


def cmd_answer(args: argparse.Namespace) -> int:
    settings = get_settings()
    corpus = CorpusStore(args.derived)
    elements = list(corpus.iter_all_elements())
    if not elements:
        print("Corpus derive vide : lancer `python -m maintidoc.ingestion ingest`.")
        return 1

    try:
        text_embedder = None if args.no_dense else build_text_embedder(settings)
        visual_encoder = None if args.no_visual else build_visual_encoder(settings)
        reranker = NoReranker() if args.no_rerank else build_reranker(settings)
    except (MemoryError, OSError) as exc:
        print(f"Memoire insuffisante pour charger les modeles ({type(exc).__name__}).")
        print("Relancer avec --no-dense --no-visual, ou --no-rerank.")
        return 3

    store = VectorStore.local(args.derived / "qdrant")
    retriever = Retriever(
        store,
        text_embedder=text_embedder,
        visual_encoder=visual_encoder,
        lexical=Bm25Index.build(elements),
        reranker=reranker,
        max_evidence_chars=settings.max_evidence_chars,
        max_images=settings.max_images,
    )
    principal = Principal(
        user_id=args.user,
        groups=frozenset(args.groups or ()),
        families=frozenset(args.family) if args.family else None,
    )
    try:
        retrieved = retriever.retrieve(args.question, principal, top_k=args.limit)
    except AccessDenied as exc:
        print(f"Acces refuse : {exc}")
        store.close()
        return 2

    client = OllamaVlmClient(settings.vlm_model, host=settings.ollama_host)
    if not client.available():
        print(f"Modele {settings.vlm_model} indisponible via Ollama ({settings.ollama_host}).")
        print("Verifier avec : python -m maintidoc.generation check")
        print(f"\nRecuperation seule : {len(retrieved.bundle.items)} preuves retenues.")
        for item in retrieved.bundle.items:
            print(f"  {item.rank:>2}. {item.ref}")
        store.close()
        return 4

    checker = (
        LexicalGroundingChecker()
        if args.lexical_check
        else LlmGroundingChecker(client)
    )
    result = Answerer(
        client,
        checker=checker,
        content=corpus.content,
        num_ctx=args.num_ctx,
        versions=capture_versions(settings),
    ).answer(retrieved.bundle)
    store.close()

    print(f"Statut : {result.answer.status.value}")
    print(f"\n{result.answer.answer}\n")

    print("Citations")
    verified = set(result.verified_refs)
    grounding = {c.ref: c for c in (result.grounding.checks if result.grounding else ())}
    for citation in result.answer.citations:
        structural = "structure OK" if citation.ref in verified else "STRUCTURE KO"
        check = grounding.get(citation.ref)
        semantic = check.verdict.value if check else "non verifie"
        print(f"  {citation.ref}  [{structural} | soutien : {semantic}]")
    if not result.answer.citations:
        print("  (aucune)")

    for uncertainty in result.answer.uncertainties:
        print(f"\nIncertitude : {uncertainty}")
    for warning in result.warnings:
        print(f"Avertissement : {warning}")

    if result.grounding:
        print(f"\nTaux de citations soutenues : {result.grounding.support_rate:.0%}")
    print(f"Reponse sourcee : {'oui' if result.is_sourced else 'non'}")
    print(
        f"Modele {result.model} | prompt {result.prompt_version} | "
        f"{result.images_sent} image(s) | {result.elapsed_ms:.0f} ms"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.generation", description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Verifier Ollama et le modele")
    check.set_defaults(func=cmd_check)

    answer = subparsers.add_parser("answer", help="Repondre a une question sourcee")
    answer.add_argument("question")
    answer.add_argument("--user", default="tech-demo")
    answer.add_argument("--groups", nargs="*", default=["maintenance", "maintenance_san"])
    answer.add_argument("--family", nargs="*")
    answer.add_argument("--limit", type=int, default=6)
    answer.add_argument("--num-ctx", type=int, default=4096)
    answer.add_argument("--no-dense", action="store_true")
    answer.add_argument("--no-visual", action="store_true")
    answer.add_argument("--no-rerank", action="store_true")
    answer.add_argument(
        "--lexical-check",
        action="store_true",
        help="Verificateur lexical au lieu du juge LLM (plus rapide, plus faible)",
    )
    answer.set_defaults(func=cmd_answer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
