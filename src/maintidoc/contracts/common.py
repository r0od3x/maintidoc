"""Identity and element contracts.

Every retrievable unit of the corpus is an `Element`. Its identity triple
(document_id, page, element_id) is what a citation points at, so those fields
are frozen at ingestion and carried unchanged through retrieval, generation and
verification. If this triple is not stable, no citation can be verified.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REF_RE = re.compile(r"^(?P<document_id>[^:]+):p(?P<page>\d+):(?P<element_id>[^:]+)$")


class Modality(StrEnum):
    """Nature of an element. Drives which index it lands in and whether the
    original bytes are sent to the vision-language model."""

    TEXT = "text"
    TABLE = "table"
    SCHEMA = "schema"  # figure / diagramme technique (FIG-xx)
    PHOTO = "photo"  # illustration, photo de couverture

    @property
    def is_visual(self) -> bool:
        return self in (Modality.SCHEMA, Modality.PHOTO)


class ExtractionMethod(StrEnum):
    """How the payload was obtained — OCR text is less reliable than a text
    layer, and a rendered region is an inference about where a figure sits."""

    NATIVE = "native"  # couche texte du PDF / image incorporee
    OCR = "ocr"  # texte reconnu sur une image ou une page scannee
    RENDER = "render"  # zone rendue (figure vectorielle sans image incorporee)


class ElementRef(BaseModel):
    """Canonical, hashable pointer to one element of one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1, max_length=128)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1, max_length=128)

    @property
    def ref(self) -> str:
        """Canonical string form, e.g. `SAN-MAN-001:p04:p04_image_01`."""
        return f"{self.document_id}:p{self.page}:{self.element_id}"

    @classmethod
    def parse(cls, ref: str) -> Self:
        match = REF_RE.match(ref)
        if match is None:
            raise ValueError(
                f"Référence malformée : {ref!r} (attendu document_id:pN:element_id)"
            )
        return cls(
            document_id=match["document_id"],
            page=int(match["page"]),
            element_id=match["element_id"],
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.ref


class Element(BaseModel):
    """One indexed unit of the corpus, with its provenance and access policy.

    `acl_groups` must never be empty: an element nobody is explicitly allowed to
    see is unreachable rather than public. Phase 4 filters on this field *before*
    retrieval, so an empty value here would silently widen access.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity / provenance ---
    document_id: str = Field(min_length=1, max_length=128)
    document_version: str = Field(min_length=1, max_length=64)
    document_sha256: str = Field(pattern=SHA256_RE.pattern)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1, max_length=128)
    modality: Modality

    # --- Access control ---
    equipment_family: str = Field(min_length=1, max_length=64)
    acl_groups: tuple[str, ...] = Field(min_length=1)

    # --- Payload ---
    text: str | None = None
    image_uri: str | None = None
    image_sha256: str | None = Field(default=None, pattern=SHA256_RE.pattern)
    bbox: tuple[float, float, float, float] | None = None
    figure_id: str | None = None
    extraction: ExtractionMethod = ExtractionMethod.NATIVE

    @model_validator(mode="after")
    def _payload_matches_modality(self) -> Self:
        if self.modality.is_visual:
            if not self.image_uri:
                raise ValueError(f"{self.modality.value} requiert image_uri")
        elif not (self.text and self.text.strip()):
            raise ValueError(f"{self.modality.value} requiert un texte non vide")
        return self

    @property
    def ref(self) -> ElementRef:
        return ElementRef(
            document_id=self.document_id, page=self.page, element_id=self.element_id
        )

    def readable_by(self, groups: frozenset[str]) -> bool:
        """Deny by default: no overlap with the caller's groups means no access."""
        return bool(groups.intersection(self.acl_groups))
