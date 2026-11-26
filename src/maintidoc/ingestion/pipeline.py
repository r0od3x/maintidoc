"""Ingestion orchestration.

Idempotency is keyed on the SHA-256 of the source bytes: re-running the
pipeline over an unchanged corpus does no work and changes no element id. That
matters beyond speed — a reindex that renumbered elements would invalidate
every citation already recorded in an evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pymupdf

from maintidoc.contracts import Element
from maintidoc.ingestion.manifest import CorpusManifest, DocumentSource
from maintidoc.ingestion.ocr import NullOcr, OcrEngine
from maintidoc.ingestion.pdf import ExtractionOptions, PdfExtractor
from maintidoc.ingestion.store import CorpusStore, DocumentRecord


class IngestionStatus(StrEnum):
    INGESTED = "ingested"
    REPLACED = "replaced"  # same id, new bytes
    SKIPPED = "skipped"  # unchanged since last run
    FAILED = "failed"


@dataclass
class IngestionResult:
    document_id: str
    status: IngestionStatus
    sha256: str = ""
    element_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def total_elements(self) -> int:
        return sum(self.element_counts.values())


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(elements: list[Element]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.modality.value] = counts.get(element.modality.value, 0) + 1
    return dict(sorted(counts.items()))


class IngestionPipeline:
    def __init__(
        self,
        store: CorpusStore,
        *,
        corpus_root: Path,
        ocr: OcrEngine | None = None,
        options: ExtractionOptions | None = None,
    ) -> None:
        self.store = store
        self.corpus_root = corpus_root
        self.ocr = ocr or NullOcr()
        self.extractor = PdfExtractor(store.content, ocr=self.ocr, options=options)

    def ingest_document(
        self, source: DocumentSource, corpus_version: str, *, force: bool = False
    ) -> IngestionResult:
        path = source.resolve(self.corpus_root)
        if not path.is_file():
            return IngestionResult(
                source.document_id, IngestionStatus.FAILED,
                error=f"Fichier introuvable : {path}",
            )

        sha256 = _sha256_file(path)
        previous = self.store.record_for(source.document_id)
        if not force and self.store.is_up_to_date(source.document_id, sha256):
            return IngestionResult(
                source.document_id, IngestionStatus.SKIPPED, sha256=sha256,
                element_counts=previous.element_counts if previous else {},
            )

        warnings: list[str] = []
        duplicate = self.store.find_by_sha256(sha256)
        if duplicate and duplicate.document_id != source.document_id:
            warnings.append(
                f"Octets identiques a {duplicate.document_id} : deux entrees "
                f"de manifeste pointent sur le meme fichier"
            )

        try:
            with pymupdf.open(path) as doc:
                elements, stats = self.extractor.extract(doc, source, sha256)
                page_count = doc.page_count
        except Exception as exc:
            return IngestionResult(
                source.document_id, IngestionStatus.FAILED, sha256=sha256,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not elements:
            warnings.append("Aucun element extrait : verifier le PDF")

        self.store.write_elements(source.document_id, elements)
        self.store.save_record(
            DocumentRecord(
                document_id=source.document_id,
                document_version=source.document_version,
                corpus_version=corpus_version,
                source_path=str(source.path).replace("\\", "/"),
                sha256=sha256,
                page_count=page_count,
                element_counts=_counts(elements),
                ocr_applied=stats.ocr_elements > 0,
            )
        )
        status = (
            IngestionStatus.REPLACED
            if previous and previous.sha256 != sha256
            else IngestionStatus.INGESTED
        )
        return IngestionResult(
            source.document_id, status, sha256=sha256,
            element_counts=_counts(elements), warnings=warnings + stats.warnings,
        )

    def ingest_corpus(
        self,
        manifest: CorpusManifest,
        *,
        force: bool = False,
        only: str | None = None,
    ) -> list[IngestionResult]:
        documents = (
            [manifest.get(only)] if only else list(manifest.documents)
        )
        return [
            self.ingest_document(document, manifest.corpus_version, force=force)
            for document in documents
        ]
