"""Shared runtime for the API.

One container, built at startup, holding the retriever, the answerer, the
maintenance service and the agent graph.

Loading models **once and sharing them** is not an optimisation here, it is a
requirement: BGE-M3 and the reranker together already exhaust memory on the
development machine, and a per-request load would fail on the second request.
That failure was found in phase 4 and this is where the lesson lands.

`Runtime.lean()` builds a working stack with no models at all — stub generator,
lexical retrieval, lexical grounding. The tests use it, and so can a demo on a
machine that cannot hold the real ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maintidoc.agent.budget import BudgetTracker
from maintidoc.agent.graph import MaintiDocGraph
from maintidoc.api.security import TokenService, UserDirectory
from maintidoc.config import CORPUS_DIR, Settings, get_settings
from maintidoc.generation.answering import Answerer
from maintidoc.generation.client import StubVlmClient, VlmClient
from maintidoc.generation.grounding import GroundingChecker, LexicalGroundingChecker
from maintidoc.ingestion.store import CorpusStore
from maintidoc.mcp_server import (
    ApprovalSigner,
    MaintenanceRepository,
    MaintenanceService,
    seed,
)
from maintidoc.retrieval import (
    Bm25Index,
    HashEmbedder,
    Indexer,
    NoReranker,
    Retriever,
    VectorStore,
)


@dataclass
class Runtime:
    settings: Settings
    corpus: CorpusStore
    store: VectorStore
    retriever: Retriever
    answerer: Answerer
    service: MaintenanceService
    graph: MaintiDocGraph
    tokens: TokenService
    directory: UserDirectory
    model_name: str = "stub"
    degraded: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.degraded is None:
            self.degraded = []

    def new_budget(self) -> BudgetTracker:
        return BudgetTracker.from_settings(self.settings)

    def close(self) -> None:
        self.store.close()
        self.service.repository.close()

    # --- builders ---------------------------------------------------------

    @classmethod
    def lean(
        cls,
        *,
        settings: Settings | None = None,
        derived: Path | None = None,
        vlm: VlmClient | None = None,
        checker: GroundingChecker | None = None,
        elements: list | None = None,
    ) -> Runtime:
        """A complete stack with no large model loaded.

        Retrieval is lexical only and the index is in memory, so this starts
        in milliseconds and needs nothing downloaded.
        """
        cfg = settings or get_settings()
        root = derived or (CORPUS_DIR / "derived")
        corpus = CorpusStore(root)
        items = elements if elements is not None else list(corpus.iter_all_elements())

        store = VectorStore.memory()
        if items:
            embedder = HashEmbedder()
            Indexer(store, text_embedder=embedder).build(
                items, index_version="lean", corpus_version=cfg.corpus_version
            )

        retriever = Retriever(
            store,
            text_embedder=None,
            visual_encoder=None,
            lexical=Bm25Index.build(items),
            reranker=NoReranker(),
            max_evidence_chars=cfg.max_evidence_chars,
            max_images=cfg.max_images,
        )
        client = vlm or StubVlmClient(responses=[])
        answerer = Answerer(
            client, checker=checker or LexicalGroundingChecker(), content=corpus.content
        )
        service = _service(cfg, ":memory:", seed_data=True)
        return cls(
            settings=cfg,
            corpus=corpus,
            store=store,
            retriever=retriever,
            answerer=answerer,
            service=service,
            graph=MaintiDocGraph(retriever, answerer, service=service),
            tokens=_token_service(cfg),
            directory=UserDirectory.demo(),
            model_name=getattr(client, "name", "stub"),
            degraded=["modeles non charges (mode allege)"],
        )

    @classmethod
    def full(cls, *, settings: Settings | None = None, derived: Path | None = None) -> Runtime:
        """The real stack. Degrades component by component and says which."""
        from maintidoc.generation.client import OllamaVlmClient
        from maintidoc.generation.grounding import LlmGroundingChecker
        from maintidoc.retrieval import (
            build_reranker,
            build_text_embedder,
            build_visual_encoder,
        )

        cfg = settings or get_settings()
        root = derived or (CORPUS_DIR / "derived")
        corpus = CorpusStore(root)
        items = list(corpus.iter_all_elements())
        degraded: list[str] = []

        def _try(label: str, build):
            try:
                return build()
            except (MemoryError, OSError, ImportError, ValueError) as exc:
                degraded.append(f"{label} indisponible ({type(exc).__name__})")
                return None

        text_embedder = _try("embeddings texte", lambda: build_text_embedder(cfg))
        visual_encoder = _try("encodeur visuel", lambda: build_visual_encoder(cfg))
        reranker = _try("reranker", lambda: build_reranker(cfg)) or NoReranker()

        store = VectorStore.local(root / "qdrant")
        retriever = Retriever(
            store,
            text_embedder=text_embedder,
            visual_encoder=visual_encoder,
            lexical=Bm25Index.build(items),
            reranker=reranker,
            max_evidence_chars=cfg.max_evidence_chars,
            max_images=cfg.max_images,
        )

        client = OllamaVlmClient(cfg.vlm_model, host=cfg.ollama_host)
        if not client.available():
            degraded.append(f"modele {cfg.vlm_model} indisponible via Ollama")
        answerer = Answerer(
            client,
            checker=LlmGroundingChecker(client),
            content=corpus.content,
            versions=None,
        )
        service = _service(cfg, root / "maintenance.sqlite3", seed_data=True)
        return cls(
            settings=cfg,
            corpus=corpus,
            store=store,
            retriever=retriever,
            answerer=answerer,
            service=service,
            graph=MaintiDocGraph(retriever, answerer, service=service),
            tokens=_token_service(cfg),
            directory=UserDirectory.demo(),
            model_name=cfg.vlm_model,
            degraded=degraded,
        )


def _token_service(settings: Settings) -> TokenService:
    secret = settings.jwt_secret.get_secret_value()
    if not secret:
        # Development convenience, stated loudly rather than hidden: a fixed
        # fallback secret must never reach a deployment.
        secret = "secret-de-developpement-a-remplacer"
    return TokenService(secret)


def _service(settings: Settings, path: Any, *, seed_data: bool) -> MaintenanceService:
    repository = MaintenanceRepository(path)
    if seed_data and not repository.equipment("SAN-LAB-2000"):
        seed(repository)
    secret = settings.jwt_secret.get_secret_value() or "secret-de-developpement"
    return MaintenanceService(
        repository,
        ApprovalSigner(secret, ttl_seconds=settings.approval_token_ttl_seconds),
    )
