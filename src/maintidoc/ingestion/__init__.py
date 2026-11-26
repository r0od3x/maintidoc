"""Phase 2 — ingestion : PDF vers elements traçables.

    from maintidoc.ingestion import CorpusManifest, CorpusStore, IngestionPipeline
"""

from maintidoc.ingestion.manifest import CorpusManifest, DocumentSource
from maintidoc.ingestion.ocr import NullOcr, OcrEngine, TesseractOcr, default_engine
from maintidoc.ingestion.pdf import ExtractionOptions, ExtractionStats, PdfExtractor
from maintidoc.ingestion.pipeline import IngestionPipeline, IngestionResult, IngestionStatus
from maintidoc.ingestion.store import ContentStore, CorpusStore, DocumentRecord, LocalContentStore

__all__ = [
    "ContentStore",
    "CorpusManifest",
    "CorpusStore",
    "DocumentRecord",
    "DocumentSource",
    "ExtractionOptions",
    "ExtractionStats",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionStatus",
    "LocalContentStore",
    "NullOcr",
    "OcrEngine",
    "PdfExtractor",
    "TesseractOcr",
    "default_engine",
]
