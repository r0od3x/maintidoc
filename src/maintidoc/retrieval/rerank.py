"""Cross-encoder reranking.

Fusion orders candidates without ever having compared the question to a
passage directly — it only knows ranks. A cross-encoder reads both together
and is far better at the distinction this corpus is built on: which of ten
near-identical passages is about *this* model.

This is also the component phase 10 fine-tunes, which is why it sits behind a
protocol. Swapping the base model for an adapted one must not touch the
retriever.

An element with no text cannot be scored by a text cross-encoder. Those keep
their fusion rank and are merged back in, rather than being silently dropped —
an uncaptioned figure is still evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from maintidoc.retrieval.fusion import FusedCandidate

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, question: str, candidates: list[FusedCandidate], *, limit: int | None = None
    ) -> list[FusedCandidate]: ...


class NoReranker:
    """Keeps the fusion order. The baseline phase 10 measures against."""

    name = "none"

    def rerank(
        self, question: str, candidates: list[FusedCandidate], *, limit: int | None = None
    ) -> list[FusedCandidate]:
        ordered = list(candidates)
        return ordered[:limit] if limit else ordered


class CrossEncoderReranker:
    """`bge-reranker-v2-m3` by default, multilingual and French-capable."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            cache_folder=str(cache_dir) if cache_dir else None,
        )

    def rerank(
        self, question: str, candidates: list[FusedCandidate], *, limit: int | None = None
    ) -> list[FusedCandidate]:
        scorable = [c for c in candidates if (c.hit.text or "").strip()]
        unscorable = [c for c in candidates if not (c.hit.text or "").strip()]
        if not scorable:
            return list(candidates)[:limit] if limit else list(candidates)

        scores = self._model.predict(
            [(question, c.hit.text or "") for c in scorable],
            show_progress_bar=False,
        )
        for candidate, score in zip(scorable, scores, strict=True):
            candidate.score = float(score)
        scorable.sort(key=lambda c: (-c.score, c.ref))

        # Unscorable candidates keep their relative fusion order, appended after
        # the reranked ones rather than discarded.
        unscorable.sort(key=lambda c: c.rank)
        ordered = scorable + unscorable
        for position, candidate in enumerate(ordered, start=1):
            candidate.rank = position
        return ordered[:limit] if limit else ordered


def build_reranker(settings, *, enabled: bool = True) -> Reranker:
    if not enabled:
        return NoReranker()
    return CrossEncoderReranker(
        settings.reranker_model, cache_dir=settings.model_cache_dir
    )
