"""Evidence contracts: what was actually retrieved and actually sent to the model.

The bundle is the single source of truth for citation verification. A citation
is structurally valid only if it names an element present in *this* bundle —
not merely an element that exists somewhere in the corpus.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maintidoc.contracts.common import Element, ElementRef, Modality


class RetrievalBranch(StrEnum):
    """Which search produced the candidate, kept for the RRF audit trail."""

    TEXT = "text"
    VISUAL = "visual"
    FUSED = "fused"


class Evidence(BaseModel):
    """One retrieved element, bounded to what is transmitted to the model."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1)
    modality: Modality

    excerpt: str | None = None
    image_uri: str | None = None

    branch: RetrievalBranch
    rank: int = Field(ge=1)
    raw_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @model_validator(mode="after")
    def _payload_present(self) -> Self:
        # A figure may travel as its caption alone when the image budget is
        # already spent: the evidence is weaker, not absent. What it may never
        # be is empty.
        if self.modality.is_visual and not (self.image_uri or self.excerpt):
            raise ValueError("Une preuve visuelle doit porter image_uri ou sa legende")
        if not self.modality.is_visual and self.excerpt is None:
            raise ValueError("Une preuve textuelle doit porter un excerpt")
        return self

    @property
    def ref(self) -> str:
        return ElementRef(
            document_id=self.document_id, page=self.page, element_id=self.element_id
        ).ref

    @classmethod
    def from_element(
        cls,
        element: Element,
        *,
        branch: RetrievalBranch,
        rank: int,
        excerpt: str | None = None,
        **scores: float | None,
    ) -> Self:
        return cls(
            document_id=element.document_id,
            page=element.page,
            element_id=element.element_id,
            modality=element.modality,
            # A figure keeps its caption as text alongside the image.
            excerpt=excerpt or element.text,
            image_uri=element.image_uri,
            branch=branch,
            rank=rank,
            **scores,
        )


class EvidenceBundle(BaseModel):
    """The bounded evidence set handed to the VLM for one request."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    visual_query: str | None = None
    items: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _refs_are_unique(self) -> Self:
        refs = [item.ref for item in self.items]
        if len(refs) != len(set(refs)):
            raise ValueError("Preuves dupliquées dans le dossier")
        return self

    @property
    def allowed_refs(self) -> frozenset[str]:
        """The only citation targets the model may legitimately produce."""
        return frozenset(item.ref for item in self.items)

    @property
    def text_items(self) -> tuple[Evidence, ...]:
        return tuple(i for i in self.items if i.excerpt)

    @property
    def image_items(self) -> tuple[Evidence, ...]:
        """Evidence actually carrying an image — what the budget counts.

        A figure reduced to its caption is not an image for this purpose, so it
        does not consume the image budget.
        """
        return tuple(i for i in self.items if i.image_uri)

    @property
    def total_excerpt_chars(self) -> int:
        return sum(len(i.excerpt or "") for i in self.items)

    def get(self, ref: str) -> Evidence | None:
        return next((i for i in self.items if i.ref == ref), None)

    def enforce_budget(self, *, max_chars: int, max_images: int) -> None:
        """Raise if the bundle exceeds the context budget. Called by the builder
        in phase 4; kept here so the limit travels with the data it bounds."""
        if self.total_excerpt_chars > max_chars:
            raise ValueError(
                f"Budget texte dépassé : {self.total_excerpt_chars} > {max_chars}"
            )
        if len(self.image_items) > max_images:
            raise ValueError(
                f"Budget images dépassé : {len(self.image_items)} > {max_images}"
            )
