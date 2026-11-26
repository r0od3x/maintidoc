"""Run the agent end to end, including the human approval step.

    uv run python -m maintidoc.agent ask "Que signifie le voyant orange ... ?" \
        --equipment SAN-LAB-2000 --ticket

The approval prompt is answered on the terminal. The run genuinely stops at
that point: nothing is created until a decision is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from maintidoc.agent.budget import BudgetTracker
from maintidoc.agent.graph import MaintiDocGraph
from maintidoc.agent.state import initial_state
from maintidoc.config import CORPUS_DIR, get_settings
from maintidoc.contracts import Principal
from maintidoc.generation.answering import Answerer
from maintidoc.generation.client import OllamaVlmClient
from maintidoc.generation.grounding import LexicalGroundingChecker, LlmGroundingChecker
from maintidoc.ingestion.store import CorpusStore
from maintidoc.mcp_server import (
    ApprovalSigner,
    MaintenanceRepository,
    MaintenanceService,
    seed,
)
from maintidoc.retrieval import (
    Bm25Index,
    NoReranker,
    Retriever,
    VectorStore,
    build_reranker,
    build_text_embedder,
    build_visual_encoder,
)

DEFAULT_DERIVED = CORPUS_DIR / "derived"
DEFAULT_DB = CORPUS_DIR / "derived" / "maintenance.sqlite3"


def _service(args: argparse.Namespace) -> MaintenanceService:
    settings = get_settings()
    repository = MaintenanceRepository(args.database)
    if args.seed:
        seed(repository)
    secret = settings.jwt_secret.get_secret_value() or "secret-de-developpement"
    return MaintenanceService(
        repository,
        ApprovalSigner(secret, ttl_seconds=settings.approval_token_ttl_seconds),
    )


def cmd_ask(args: argparse.Namespace) -> int:
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
        print(f"Memoire insuffisante ({type(exc).__name__}). Essayer --no-dense --no-visual.")
        return 3

    client = OllamaVlmClient(settings.vlm_model, host=settings.ollama_host)
    if not client.available():
        print(f"Modele {settings.vlm_model} indisponible via Ollama.")
        print("Verifier avec : python -m maintidoc.generation check")
        return 4

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
    answerer = Answerer(
        client,
        checker=LexicalGroundingChecker() if args.lexical_check else LlmGroundingChecker(client),
        content=corpus.content,
    )
    graph = MaintiDocGraph(retriever, answerer, service=_service(args))

    state = initial_state(
        args.question,
        Principal(user_id=args.user, groups=frozenset(args.groups or ())),
        BudgetTracker.from_settings(settings),
        equipment_id=args.equipment,
        wants_ticket=args.ticket,
    )
    result = graph.run(state, thread_id=args.thread)

    pending = graph.pending_approval(thread_id=args.thread)
    if pending is not None:
        print("\n--- VALIDATION HUMAINE REQUISE ---")
        print(json.dumps(pending["ticket"], ensure_ascii=False, indent=2))
        answer = input("Creer ce ticket ? [o/N] ").strip().lower()
        result = graph.resume(
            {"approved": answer in {"o", "oui", "y", "yes"}}, thread_id=args.thread
        )

    store.close()
    _report(result)
    return 0


def _report(result: dict) -> None:
    answer = result.get("answer")
    if answer is not None:
        print(f"\nStatut : {answer.status.value}")
        print(f"\n{answer.answer}\n")
        for citation in answer.citations:
            print(f"  source : {citation.ref}")
        for uncertainty in answer.uncertainties:
            print(f"  incertitude : {uncertainty}")

    outcome = result.get("outcome") or {}
    print(f"\nIssue : {outcome.get('status', 'inconnue')} {outcome.get('detail', '')}")
    if result.get("ticket_id"):
        print(f"Ticket cree : {result['ticket_id']}")
    for warning in result.get("warnings", []):
        print(f"Avertissement : {warning}")
    budget = result.get("budget")
    if budget is not None:
        print(f"Budgets : {budget.snapshot()}")
    print(f"Parcours : {' -> '.join(result.get('trace', []))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintidoc.agent", description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Poser une question au workflow complet")
    ask.add_argument("question")
    ask.add_argument("--user", default="tech-san")
    ask.add_argument("--groups", nargs="*", default=["maintenance", "maintenance_san"])
    ask.add_argument("--equipment", help="Identifiant d'equipement pour l'historique MCP")
    ask.add_argument("--ticket", action="store_true", help="Preparer un ticket")
    ask.add_argument("--thread", default="cli")
    ask.add_argument("--seed", action="store_true", help="Semer la base de maintenance")
    ask.add_argument("--no-dense", action="store_true")
    ask.add_argument("--no-visual", action="store_true")
    ask.add_argument("--no-rerank", action="store_true")
    ask.add_argument("--lexical-check", action="store_true")
    ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
