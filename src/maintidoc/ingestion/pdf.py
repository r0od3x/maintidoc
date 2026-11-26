"""PDF extraction into `Element` objects.

Produces four kinds of element, each citable on its own: text chunks, tables,
figures (with their caption) and uncaptioned illustrations.

Three faults of the prototype are corrected here:

  * table cells were indexed twice, once as a table and once as loose text
    blocks, which inflates retrieval scores for whatever a table mentions;
  * running headers and footers were indexed as content on every page;
  * figures were identified by page number rather than by caption.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from maintidoc.contracts import Element, ExtractionMethod, Modality
from maintidoc.ingestion.figures import (
    Caption,
    find_captions,
    pick_image_for_caption,
    vector_region_above,
)
from maintidoc.ingestion.manifest import DocumentSource
from maintidoc.ingestion.ocr import NullOcr, OcrEngine
from maintidoc.ingestion.store import ContentStore

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
DIGITS = re.compile(r"\d+")
OCR_LANGUAGES = {"fr": "fra", "en": "eng"}


@dataclass(frozen=True, slots=True)
class ExtractionOptions:
    max_chunk_chars: int = 1100
    min_image_width: int = 120
    min_image_height: int = 85
    render_dpi: int = 150
    scanned_page_threshold: int = 60
    ocr_figures: bool = True
    margin_ratio: float = 0.10


@dataclass
class ExtractionStats:
    pages: int = 0
    boilerplate_blocks: int = 0
    captions: int = 0
    rendered_figures: int = 0
    ocr_elements: int = 0
    skipped_small_images: int = 0
    warnings: list[str] = field(default_factory=list)


def _flat(text: str) -> str:
    return " ".join(text.split())


def boilerplate_key(text: str) -> str:
    """Normalised form used to recognise running furniture.

    Digits are collapsed because a footer almost always carries the page
    number: "... | Page 2" and "... | Page 3" are the same furniture and would
    otherwise each look unique and survive the filter.
    """
    return DIGITS.sub("#", _flat(text))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_boilerplate(doc: pymupdf.Document, margin_ratio: float = 0.10) -> set[str]:
    """Running headers and footers, recognised by repeating in the page margins.

    The search is restricted to the margin bands so that a sentence genuinely
    repeated in the body of a document is not discarded as furniture.
    """
    seen: Counter[str] = Counter()
    for page in doc:
        height = page.rect.height
        top, bottom = height * margin_ratio, height * (1 - margin_ratio)
        on_page = set()
        for _x0, y0, _x1, y1, text, *_rest in page.get_text("blocks", sort=True):
            if y1 < top or y0 > bottom:
                key = boilerplate_key(text)
                if 0 < len(key) <= 200:
                    on_page.add(key)
        seen.update(on_page)
    threshold = max(2, math.ceil(doc.page_count * 0.5))
    return {text for text, count in seen.items() if count >= threshold}


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Markdown rendering of a table, readable both by the embedding model and
    by a human checking a citation against the page."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [[_flat(cell or "") for cell in row] + [""] * (width - len(row)) for row in rows]
    header, *body = padded
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def chunk_blocks(blocks: list[tuple], max_chars: int) -> list[tuple[str, tuple]]:
    """Group text blocks into chunks bounded by `max_chars`, without overlap.

    Block boundaries are preserved, so a cited excerpt always reads as a whole
    thought. Overlap is deliberately avoided: duplicated text would let one
    passage be retrieved twice and would distort Recall@k.
    """
    chunks: list[tuple[str, tuple]] = []
    buffer: list[str] = []
    bbox: list[float] | None = None

    def flush() -> None:
        nonlocal buffer, bbox
        if buffer and bbox is not None:
            chunks.append(("\n".join(buffer), tuple(bbox)))
        buffer, bbox = [], None

    for x0, y0, x1, y1, text, *_rest in blocks:
        flat = _flat(text)
        if not flat:
            continue
        pieces: list[str] = [flat]
        if len(flat) > max_chars:
            pieces, current = [], ""
            for sentence in SENTENCE_SPLIT.split(flat):
                if current and len(current) + len(sentence) + 1 > max_chars:
                    pieces.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                pieces.append(current)
        for piece in pieces:
            if buffer and sum(len(b) for b in buffer) + len(piece) + 1 > max_chars:
                flush()
            buffer.append(piece)
            bbox = (
                [x0, y0, x1, y1]
                if bbox is None
                else [
                    min(bbox[0], x0),
                    min(bbox[1], y0),
                    max(bbox[2], x1),
                    max(bbox[3], y1),
                ]
            )
    flush()
    return chunks


class PdfExtractor:
    """Turns one PDF into validated `Element` objects."""

    def __init__(
        self,
        store: ContentStore,
        *,
        ocr: OcrEngine | None = None,
        options: ExtractionOptions | None = None,
    ) -> None:
        self.store = store
        self.ocr = ocr or NullOcr()
        self.options = options or ExtractionOptions()

    def extract(
        self, doc: pymupdf.Document, source: DocumentSource, sha256: str
    ) -> tuple[list[Element], ExtractionStats]:
        stats = ExtractionStats(pages=doc.page_count)
        boilerplate = detect_boilerplate(doc, self.options.margin_ratio)
        elements: list[Element] = []
        for number, page in enumerate(doc, start=1):
            elements.extend(
                self._extract_page(page, number, source, sha256, boilerplate, stats)
            )
        return elements, stats

    # --- helpers ----------------------------------------------------------

    def _build(
        self,
        source: DocumentSource,
        sha256: str,
        *,
        page: int,
        suffix: str,
        modality: Modality,
        **payload: object,
    ) -> Element:
        return Element(
            document_id=source.document_id,
            document_version=source.document_version,
            document_sha256=sha256,
            page=page,
            element_id=f"p{page:02d}_{suffix}",
            modality=modality,
            equipment_family=source.equipment_family,
            acl_groups=source.acl_groups,
            **payload,
        )

    def _page_images(
        self, page: pymupdf.Page, stats: ExtractionStats
    ) -> list[tuple[pymupdf.Rect, bytes, str]]:
        """Embedded raster images, deduplicated by xref and filtered by size."""
        found: list[tuple[pymupdf.Rect, bytes, str]] = []
        seen: set[int] = set()
        for info in page.get_images(full=True):
            xref, width, height = info[0], info[2], info[3]
            if xref in seen:
                continue
            seen.add(xref)
            if width < self.options.min_image_width or height < self.options.min_image_height:
                stats.skipped_small_images += 1
                continue
            try:
                payload = page.parent.extract_image(xref)
                rects = page.get_image_rects(xref)
            except Exception as exc:
                stats.warnings.append(f"xref {xref} illisible ({exc})")
                continue
            if not payload or not rects:
                continue
            found.append((rects[0], payload["image"], payload.get("ext", "png")))
        return found

    def _table_rects(self, page: pymupdf.Page) -> list[pymupdf.Rect]:
        try:
            return [pymupdf.Rect(table.bbox) for table in page.find_tables().tables]
        except Exception:
            return []

    # --- per page ---------------------------------------------------------

    def _extract_page(
        self,
        page: pymupdf.Page,
        number: int,
        source: DocumentSource,
        sha256: str,
        boilerplate: set[str],
        stats: ExtractionStats,
    ) -> list[Element]:
        elements: list[Element] = []
        blocks = [block for block in page.get_text("blocks", sort=True) if block[6] == 0]

        elements += self._tables(page, number, source, sha256, stats)

        captions = find_captions(blocks)
        stats.captions += len(captions)
        images = self._page_images(page, stats)
        claimed: set[int] = set()
        elements += self._figures(
            page, number, source, sha256, blocks, captions, images, claimed, stats
        )
        elements += self._illustrations(number, source, sha256, images, claimed)
        elements += self._body_text(
            page, number, source, sha256, blocks, captions, boilerplate, stats
        )
        return elements

    def _tables(
        self,
        page: pymupdf.Page,
        number: int,
        source: DocumentSource,
        sha256: str,
        stats: ExtractionStats,
    ) -> list[Element]:
        try:
            tables = page.find_tables().tables
        except Exception as exc:  # a detector failure must not lose the page
            stats.warnings.append(f"page {number} : detection de tableaux echouee ({exc})")
            return []
        elements: list[Element] = []
        for rank, table in enumerate(tables, start=1):
            rows = [[cell or "" for cell in (row or [])] for row in table.extract()]
            markdown = rows_to_markdown(rows)
            if not markdown.strip():
                continue
            elements.append(
                self._build(
                    source,
                    sha256,
                    page=number,
                    suffix=f"table_{rank:02d}",
                    modality=Modality.TABLE,
                    text=markdown,
                    bbox=tuple(table.bbox),
                )
            )
        return elements

    def _figures(
        self,
        page: pymupdf.Page,
        number: int,
        source: DocumentSource,
        sha256: str,
        blocks: list[tuple],
        captions: list[Caption],
        images: list[tuple[pymupdf.Rect, bytes, str]],
        claimed: set[int],
        stats: ExtractionStats,
    ) -> list[Element]:
        elements: list[Element] = []
        obstacles = [pymupdf.Rect(block[:4]) for block in blocks]

        for rank, caption in enumerate(captions, start=1):
            rect = pick_image_for_caption(caption, [r for r, _, _ in images])
            payload: tuple[bytes, str, tuple] | None = None
            method = ExtractionMethod.NATIVE

            if rect is not None:
                for position, (image_rect, data, extension) in enumerate(images):
                    if image_rect == rect and position not in claimed:
                        claimed.add(position)
                        payload = (data, extension, tuple(rect))
                        break

            if payload is None:
                # No embedded image: the figure is line art, which
                # `get_images()` never reports. Render the region instead.
                region = vector_region_above(caption, page.rect, obstacles)
                if region is None:
                    stats.warnings.append(
                        f"page {number} : legende {caption.label} sans figure localisable"
                    )
                    continue
                pixmap = page.get_pixmap(clip=region, dpi=self.options.render_dpi)
                payload = (pixmap.tobytes("png"), "png", tuple(region))
                method = ExtractionMethod.RENDER
                stats.rendered_figures += 1

            data, extension, bbox = payload
            elements.append(
                self._build(
                    source,
                    sha256,
                    page=number,
                    suffix=f"fig_{rank:02d}",
                    modality=Modality.SCHEMA,
                    text=caption.text,
                    image_uri=self.store.put(data, extension=extension),
                    image_sha256=_sha256(data),
                    bbox=bbox,
                    figure_id=caption.label,
                    extraction=method,
                )
            )
            elements += self._figure_ocr(data, source, sha256, number, rank, caption, stats)
        return elements

    def _figure_ocr(
        self,
        data: bytes,
        source: DocumentSource,
        sha256: str,
        number: int,
        rank: int,
        caption: Caption,
        stats: ExtractionStats,
    ) -> list[Element]:
        """Text printed inside a diagram, kept as its own citable element.

        A label on a schematic is often the answer to the question, and it is
        invisible to a text-only index.
        """
        if not (self.options.ocr_figures and self.ocr.available()):
            return []
        text = self.ocr.read(data, language=OCR_LANGUAGES.get(source.language, "fra"))
        if not text:
            return []
        stats.ocr_elements += 1
        return [
            self._build(
                source,
                sha256,
                page=number,
                suffix=f"fig_{rank:02d}_ocr",
                modality=Modality.TEXT,
                text=f"{caption.label} - texte lu sur la figure : {text}",
                extraction=ExtractionMethod.OCR,
            )
        ]

    def _illustrations(
        self,
        number: int,
        source: DocumentSource,
        sha256: str,
        images: list[tuple[pymupdf.Rect, bytes, str]],
        claimed: set[int],
    ) -> list[Element]:
        """Images no caption refers to: cover photos, logos, decoration."""
        elements: list[Element] = []
        for position, (rect, data, extension) in enumerate(images):
            if position in claimed:
                continue
            elements.append(
                self._build(
                    source,
                    sha256,
                    page=number,
                    suffix=f"img_{position + 1:02d}",
                    modality=Modality.PHOTO,
                    image_uri=self.store.put(data, extension=extension),
                    image_sha256=_sha256(data),
                    bbox=tuple(rect),
                )
            )
        return elements

    def _body_text(
        self,
        page: pymupdf.Page,
        number: int,
        source: DocumentSource,
        sha256: str,
        blocks: list[tuple],
        captions: list[Caption],
        boilerplate: set[str],
        stats: ExtractionStats,
    ) -> list[Element]:
        # Every block a caption consumed, continuation lines included.
        caption_blocks = {i for caption in captions for i in caption.block_indices}
        table_rects = self._table_rects(page)
        body: list[tuple] = []
        for index, block in enumerate(blocks):
            if index in caption_blocks:
                continue  # belongs to the figure, not to the prose
            if boilerplate_key(block[4]) in boilerplate:
                stats.boilerplate_blocks += 1
                continue
            centre = pymupdf.Point((block[0] + block[2]) / 2, (block[1] + block[3]) / 2)
            if any(centre in rect for rect in table_rects):
                continue  # already indexed as a table
            body.append(block)

        elements: list[Element] = []
        for rank, (text, bbox) in enumerate(
            chunk_blocks(body, self.options.max_chunk_chars), start=1
        ):
            elements.append(
                self._build(
                    source,
                    sha256,
                    page=number,
                    suffix=f"text_{rank:02d}",
                    modality=Modality.TEXT,
                    text=text,
                    bbox=bbox,
                )
            )

        native_chars = sum(len(_flat(block[4])) for block in blocks)
        if source.scanned or native_chars < self.options.scanned_page_threshold:
            elements += self._scanned_page(page, number, source, sha256, stats)
        return elements

    def _scanned_page(
        self,
        page: pymupdf.Page,
        number: int,
        source: DocumentSource,
        sha256: str,
        stats: ExtractionStats,
    ) -> list[Element]:
        if not self.ocr.available():
            stats.warnings.append(
                f"page {number} : peu de texte natif et aucun moteur OCR disponible"
            )
            return []
        pixmap = page.get_pixmap(dpi=self.options.render_dpi)
        text = self.ocr.read(
            pixmap.tobytes("png"), language=OCR_LANGUAGES.get(source.language, "fra")
        )
        if not text:
            return []
        stats.ocr_elements += 1
        return [
            self._build(
                source,
                sha256,
                page=number,
                suffix="ocr_01",
                modality=Modality.TEXT,
                text=text,
                extraction=ExtractionMethod.OCR,
            )
        ]
