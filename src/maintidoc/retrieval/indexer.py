"""Building the two indexes, and recording what was built.

A reindex is a tracked job, not a side effect. Every run appends an `IndexRun`
describing the models, dimensions, versions and counts involved. Without that
record, two evaluation results are not comparable: a change in recall could
come from the reranker, the corpus or a quietly swapped embedding model, and
nothing would say which.

An element can legitimately appear in both indexes. A captioned figure is
findable by its caption in the text index and by its content in the visual one;
the reference is the same in both, so the fusion of phase 4 merges them rather
than counting the figure twice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from maintidoc.contracts import Element
from maintidoc.ingestion.store import ContentStore
from maintidoc.retrieval.embeddings import TextEmbedder, VisualEncoder
from maintidoc.retrieval.store import Space, VectorStore, element_payload

DEFAULT_BATCH = 16


class IndexRun(BaseModel):
    """Provenance of one indexing job."""

    model_config = ConfigDict(extra="forbid")

    index_version: str
    corpus_version: str
    text_model: str
    text_dimension: int
    visual_model: str | None = None
    visual_dimension: int | None = None
    text_points: int = 0
    visual_points: int = 0
    documents: int = 0
    skipped_images: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def _batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Indexer:
    def __init__(
        self,
        store: VectorStore,
        *,
        text_embedder: TextEmbedder,
        visual_encoder: VisualEncoder | None = None,
        content: ContentStore | None = None,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.store = store
        self.text_embedder = text_embedder
        self.visual_encoder = visual_encoder
        self.content = content
        self.batch_size = batch_size

    # --- selection --------------------------------------------------------

    @staticmethod
    def text_candidates(elements: Iterable[Element]) -> list[Element]:
        """Anything carrying words, figure captions included."""
        return [e for e in elements if (e.text or "").strip()]

    @staticmethod
    def visual_candidates(elements: Iterable[Element]) -> list[Element]:
        return [e for e in elements if e.image_uri]

    # --- building ---------------------------------------------------------

    def build(
        self,
        elements: Sequence[Element],
        *,
        index_version: str,
        corpus_version: str,
        recreate: bool = False,
    ) -> IndexRun:
        started = datetime.now(UTC)
        skipped: list[str] = []

        texts = self.text_candidates(elements)
        self.store.ensure_collection(
            Space.TEXT, self.text_embedder.dimension, recreate=recreate
        )
        text_points = 0
        for batch in _batched(texts, self.batch_size):
            vectors = self.text_embedder.encode_passages([e.text or "" for e in batch])
            payloads = [
                element_payload(e, index_version=index_version, corpus_version=corpus_version)
                for e in batch
            ]
            text_points += self.store.upsert(Space.TEXT, vectors, payloads)

        visual_points = 0
        visuals: list[Element] = []
        if self.visual_encoder is not None and self.content is not None:
            visuals = self.visual_candidates(elements)
            self.store.ensure_collection(
                Space.VISUAL, self.visual_encoder.dimension, recreate=recreate
            )
            for batch in _batched(visuals, self.batch_size):
                loaded: list[bytes] = []
                kept: list[Element] = []
                for element in batch:
                    try:
                        loaded.append(self.content.get(element.image_uri or ""))
                        kept.append(element)
                    except (OSError, ValueError):
                        # A missing binary must not abort the whole index.
                        skipped.append(element.ref.ref)
                if not kept:
                    continue
                vectors = self.visual_encoder.encode_images(loaded)
                payloads = [
                    element_payload(
                        e, index_version=index_version, corpus_version=corpus_version
                    )
                    for e in kept
                ]
                visual_points += self.store.upsert(Space.VISUAL, vectors, payloads)

        return IndexRun(
            index_version=index_version,
            corpus_version=corpus_version,
            text_model=self.text_embedder.name,
            text_dimension=self.text_embedder.dimension,
            visual_model=self.visual_encoder.name if self.visual_encoder else None,
            visual_dimension=self.visual_encoder.dimension if self.visual_encoder else None,
            text_points=text_points,
            visual_points=visual_points,
            documents=len({e.document_id for e in elements}),
            skipped_images=skipped,
            started_at=started,
            finished_at=datetime.now(UTC),
        )


class IndexJournal:
    """Append-only log of index runs, beside the derived corpus."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, run: IndexRun) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def runs(self) -> list[IndexRun]:
        if not self.path.is_file():
            return []
        return [
            IndexRun.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def latest(self) -> IndexRun | None:
        runs = self.runs()
        return runs[-1] if runs else None
