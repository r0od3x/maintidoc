"""Corpus manifest: the declared identity and access policy of each document.

`document_id`, version and `acl_groups` are *declared*, never inferred from a
filename or a folder. Guessing them would make citations unstable and would
silently widen access, so the manifest is the one place they are decided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentSource(BaseModel):
    """One source document declared in the manifest."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    document_id: str = Field(min_length=1, max_length=128)
    document_version: str = Field(min_length=1, max_length=64)
    equipment_family: str = Field(min_length=1, max_length=64)
    acl_groups: tuple[str, ...] = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = "fr"
    scanned: bool = False

    def resolve(self, root: Path) -> Path:
        return self.path if self.path.is_absolute() else root / self.path


class CorpusManifest(BaseModel):
    """Full corpus declaration, versioned alongside the documents."""

    model_config = ConfigDict(extra="forbid")

    corpus_version: str = Field(min_length=1)
    documents: tuple[DocumentSource, ...] = ()

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        ids = [d.document_id for d in self.documents]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"document_id en double : {sorted(duplicates)}")
        return self

    @classmethod
    def from_file(cls, path: Path) -> Self:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} : le manifeste doit etre un mapping YAML")
        return cls.model_validate(data)

    def get(self, document_id: str) -> DocumentSource:
        for document in self.documents:
            if document.document_id == document_id:
                return document
        raise KeyError(f"Document absent du manifeste : {document_id}")

    @property
    def families(self) -> frozenset[str]:
        return frozenset(d.equipment_family for d in self.documents)
