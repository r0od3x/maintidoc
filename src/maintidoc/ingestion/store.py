"""Persistence for ingested elements and extracted binaries.

Phase 2 writes to the local filesystem. `ContentStore` is the seam MinIO slots
into in phase 10: a stored object is addressed by a backend-agnostic URI, never
by a filesystem path, so nothing downstream has to change.

Images are content-addressed by SHA-256. A figure reused across pages or
documents is stored once, while every occurrence keeps its own provenance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from maintidoc.contracts import Element

URI_SCHEME = "local://"


class ContentStore(Protocol):
    """Binary object storage. Implemented over MinIO in phase 10."""

    def put(self, data: bytes, *, extension: str) -> str: ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...


class LocalContentStore:
    """Content-addressed store under root/objects/xx/<sha256>.<ext>."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects"

    def _path(self, digest: str, extension: str) -> Path:
        clean = extension.lstrip(".")
        return self.objects / digest[:2] / f"{digest}.{clean}"

    def put(self, data: bytes, *, extension: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest, extension)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return URI_SCHEME + digest + "." + extension.lstrip(".")

    def _resolve(self, uri: str) -> Path:
        if not uri.startswith(URI_SCHEME):
            raise ValueError(f"URI non geree par ce stockage : {uri}")
        digest, _, extension = uri[len(URI_SCHEME):].partition(".")
        return self._path(digest, extension)

    def get(self, uri: str) -> bytes:
        return self._resolve(uri).read_bytes()

    def exists(self, uri: str) -> bool:
        return self._resolve(uri).is_file()


class DocumentRecord(BaseModel):
    """What was ingested, and from which bytes. Drives idempotency."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version: str
    corpus_version: str
    source_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    element_counts: dict[str, int] = Field(default_factory=dict)
    ocr_applied: bool = False
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_elements(self) -> int:
        return sum(self.element_counts.values())


class CorpusStore:
    """Elements, binaries and ingestion records for one derived corpus."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.content = LocalContentStore(root)
        self.elements_dir = root / "elements"
        self.registry_path = root / "documents.json"

    def _registry(self) -> dict[str, dict]:
        if not self.registry_path.is_file():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def record_for(self, document_id: str) -> DocumentRecord | None:
        raw = self._registry().get(document_id)
        return DocumentRecord.model_validate(raw) if raw else None

    def records(self) -> list[DocumentRecord]:
        return [DocumentRecord.model_validate(v) for v in self._registry().values()]

    def is_up_to_date(self, document_id: str, sha256: str) -> bool:
        """True when these exact bytes were already ingested and persisted."""
        record = self.record_for(document_id)
        return (
            record is not None
            and record.sha256 == sha256
            and self._elements_path(document_id).is_file()
        )

    def save_record(self, record: DocumentRecord) -> None:
        registry = self._registry()
        registry[record.document_id] = record.model_dump(mode="json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def find_by_sha256(self, sha256: str) -> DocumentRecord | None:
        return next((r for r in self.records() if r.sha256 == sha256), None)

    def _elements_path(self, document_id: str) -> Path:
        return self.elements_dir / (document_id.replace("/", "_") + ".jsonl")

    def write_elements(self, document_id: str, elements: list[Element]) -> None:
        path = self._elements_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(e.model_dump(mode="json"), ensure_ascii=False) for e in elements
        ]
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    def read_elements(self, document_id: str) -> list[Element]:
        path = self._elements_path(document_id)
        if not path.is_file():
            return []
        return [
            Element.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def iter_all_elements(self) -> Iterator[Element]:
        for record in sorted(self.records(), key=lambda r: r.document_id):
            yield from self.read_elements(record.document_id)
