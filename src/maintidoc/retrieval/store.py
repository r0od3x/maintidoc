"""Qdrant-backed vector store.

Two collections, because the text and visual spaces have different dimensions
and different models: `maintidoc_text` and `maintidoc_visual`.

Qdrant runs here in local mode — same engine, same API, no server. The URL form
is used in phase 10 under Docker. Keeping one client type means the code that
indexes and searches is identical in tests, on a laptop and in deployment.

Every point carries its access groups in the payload. That is what lets phase 4
filter *before* retrieval rather than fetching and then discarding: a result
the user may not read must never be scored, because its mere rank leaks the
existence and relevance of a document.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from maintidoc.contracts import Element

TEXT_COLLECTION = "maintidoc_text"
VISUAL_COLLECTION = "maintidoc_visual"


class Space(StrEnum):
    TEXT = "text"
    VISUAL = "visual"

    @property
    def collection(self) -> str:
        return TEXT_COLLECTION if self is Space.TEXT else VISUAL_COLLECTION


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One scored candidate, with everything needed to cite it."""

    ref: str
    document_id: str
    page: int
    element_id: str
    modality: str
    score: float
    rank: int
    text: str | None = None
    image_uri: str | None = None
    figure_id: str | None = None
    equipment_family: str | None = None


def element_payload(element: Element, *, index_version: str, corpus_version: str) -> dict:
    """Payload stored beside a vector.

    `acl_groups` and `equipment_family` are filter keys. The rest is what a
    citation needs, so a hit can be rendered without a second round trip to
    the element store.
    """
    return {
        "ref": element.ref.ref,
        "document_id": element.document_id,
        "document_version": element.document_version,
        "page": element.page,
        "element_id": element.element_id,
        "modality": element.modality.value,
        "extraction": element.extraction.value,
        "equipment_family": element.equipment_family,
        "acl_groups": list(element.acl_groups),
        "figure_id": element.figure_id,
        "image_uri": element.image_uri,
        "text": element.text,
        "index_version": index_version,
        "corpus_version": corpus_version,
    }


def point_id(ref: str, space: Space) -> str:
    """Stable UUID derived from the element reference.

    Re-indexing the same element overwrites its point instead of adding a
    duplicate, so a reindex is idempotent just as ingestion is.
    """
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"maintidoc/{space.value}/{ref}"))


def access_filter(
    groups: Iterable[str] | None = None,
    *,
    families: Iterable[str] | None = None,
    document_ids: Iterable[str] | None = None,
) -> models.Filter | None:
    """Pre-retrieval filter. Deny by default: no groups means no results."""
    conditions: list[models.Condition] = []
    if groups is not None:
        allowed = list(groups)
        if not allowed:
            # Match nothing rather than everything.
            return models.Filter(
                must=[models.HasIdCondition(has_id=[])]  # type: ignore[list-item]
            )
        conditions.append(
            models.FieldCondition(
                key="acl_groups", match=models.MatchAny(any=allowed)
            )
        )
    if families:
        conditions.append(
            models.FieldCondition(
                key="equipment_family", match=models.MatchAny(any=list(families))
            )
        )
    if document_ids:
        conditions.append(
            models.FieldCondition(
                key="document_id", match=models.MatchAny(any=list(document_ids))
            )
        )
    return models.Filter(must=conditions) if conditions else None


class VectorStore:
    """Thin wrapper over Qdrant for the two MaintiDoc collections."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    @classmethod
    def local(cls, path: Path) -> VectorStore:
        """On-disk local Qdrant. No server, same engine."""
        path.mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)))

    @classmethod
    def memory(cls) -> VectorStore:
        """In-process Qdrant, for tests."""
        return cls(QdrantClient(":memory:"))

    @classmethod
    def remote(cls, url: str, api_key: str | None = None) -> VectorStore:
        return cls(QdrantClient(url=url, api_key=api_key))

    # --- collections ------------------------------------------------------

    def ensure_collection(self, space: Space, dimension: int, *, recreate: bool = False) -> None:
        name = space.collection
        exists = self.client.collection_exists(name)
        if exists and recreate:
            self.client.delete_collection(name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
            # Indexing the filter keys keeps pre-filtering cheap as the corpus
            # grows. Local mode ignores them and says so on every call; the
            # indexes still matter for the server deployment of phase 10, so
            # they are declared here and the local-mode notice is silenced.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
                for field, schema in (
                    ("acl_groups", models.PayloadSchemaType.KEYWORD),
                    ("equipment_family", models.PayloadSchemaType.KEYWORD),
                    ("document_id", models.PayloadSchemaType.KEYWORD),
                    ("modality", models.PayloadSchemaType.KEYWORD),
                ):
                    self.client.create_payload_index(
                        collection_name=name, field_name=field, field_schema=schema
                    )
            return

        info = self.client.get_collection(name)
        current = info.config.params.vectors.size  # type: ignore[union-attr]
        if current != dimension:
            raise ValueError(
                f"Collection {name} en {current} dimensions, modele en {dimension}. "
                f"Changer de modele impose une reindexation : relancer avec --recreate."
            )

    def count(self, space: Space) -> int:
        if not self.client.collection_exists(space.collection):
            return 0
        return self.client.count(space.collection, exact=True).count

    # --- writing ----------------------------------------------------------

    def upsert(
        self, space: Space, vectors: np.ndarray, payloads: Sequence[dict]
    ) -> int:
        if len(payloads) == 0:
            return 0
        points = [
            models.PointStruct(
                id=point_id(payload["ref"], space),
                vector=[float(value) for value in vector],
                payload=dict(payload),
            )
            for vector, payload in zip(vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=space.collection, points=points, wait=True)
        return len(points)

    def delete_document(self, space: Space, document_id: str) -> None:
        if not self.client.collection_exists(space.collection):
            return
        self.client.delete(
            collection_name=space.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    # --- reading ----------------------------------------------------------

    def search(
        self,
        space: Space,
        vector: np.ndarray,
        *,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[SearchHit]:
        if not self.client.collection_exists(space.collection):
            return []
        found = self.client.query_points(
            collection_name=space.collection,
            query=[float(value) for value in vector],
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points
        hits: list[SearchHit] = []
        for rank, point in enumerate(found, start=1):
            payload: dict[str, Any] = point.payload or {}
            hits.append(
                SearchHit(
                    ref=payload.get("ref", ""),
                    document_id=payload.get("document_id", ""),
                    page=int(payload.get("page", 0)),
                    element_id=payload.get("element_id", ""),
                    modality=payload.get("modality", ""),
                    score=float(point.score),
                    rank=rank,
                    text=payload.get("text"),
                    image_uri=payload.get("image_uri"),
                    figure_id=payload.get("figure_id"),
                    equipment_family=payload.get("equipment_family"),
                )
            )
        return hits

    def close(self) -> None:
        self.client.close()
