"""Minimal flowing-layout engine over PyMuPDF.

Enough to emit documents that look like real manuals: headings, wrapped
paragraphs, ruled tables, figures with captions, and a running footer carrying
the page number.

Text is wrapped and measured before being placed, rather than handed to
`insert_textbox` and hoped for. Knowing the exact height in advance is what
makes page breaks deterministic — and a generator whose output shifts between
runs would break the stable element ids the pipeline depends on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PAGE_WIDTH, PAGE_HEIGHT = 595.28, 841.89
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 55.0, 62.0, 60.0
FOOTER_Y = PAGE_HEIGHT - 36.0
FIXED_DATE = "D:20261119000000Z"
BODY_FONT = "helv"
BOLD_FONT = "hebo"
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X


def text_width(text: str, fontsize: float, fontname: str = BODY_FONT) -> float:
    return pymupdf.get_text_length(text, fontname=fontname, fontsize=fontsize)


def wrap(text: str, width: float, fontsize: float, fontname: str = BODY_FONT) -> list[str]:
    """Greedy word wrap. A word longer than the line gets its own line."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and text_width(candidate, fontsize, fontname) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


@dataclass
class TableSpec:
    headers: list[str]
    rows: list[list[str]]
    fontsize: float = 8.0
    row_height: float = 16.0

    @property
    def height(self) -> float:
        return self.row_height * (len(self.rows) + 1)

    def column_widths(self, total: float) -> list[float]:
        """Share the width in proportion to the longest cell of each column."""
        counts = len(self.headers)
        longest = [
            max(
                [text_width(self.headers[i], self.fontsize, BOLD_FONT)]
                + [text_width(row[i] if i < len(row) else "", self.fontsize) for row in self.rows]
            )
            or 1.0
            for i in range(counts)
        ]
        scale = total / sum(longest)
        return [max(width * scale, 40.0) for width in longest]


@dataclass
class DocumentBuilder:
    """Builds one PDF, breaking pages as content is added."""

    title: str
    footer_label: str
    footer_note: str = "Document pedagogique fictif"

    doc: pymupdf.Document = field(default_factory=pymupdf.open)
    page: pymupdf.Page | None = None
    cursor: float = MARGIN_TOP
    page_number: int = 0
    figure_count: int = 0

    # --- page handling ----------------------------------------------------

    def new_page(self) -> pymupdf.Page:
        if self.page is not None:
            self._footer()
        self.page = self.doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.page_number += 1
        self.cursor = MARGIN_TOP
        return self.page

    def _footer(self) -> None:
        assert self.page is not None
        label = f"{self.footer_label}  |  MaintiDoc  |  {self.footer_note}"
        self.page.insert_text((MARGIN_X - 6, FOOTER_Y), label, fontsize=7.5, fontname=BODY_FONT)
        number = f"Page {self.page_number}"
        self.page.insert_text(
            (PAGE_WIDTH - MARGIN_X - text_width(number, 7.5), FOOTER_Y),
            number,
            fontsize=7.5,
            fontname=BODY_FONT,
        )

    @property
    def available(self) -> float:
        return PAGE_HEIGHT - MARGIN_BOTTOM - self.cursor

    def ensure(self, height: float) -> None:
        if self.page is None or self.available < height:
            self.new_page()

    # --- blocks -----------------------------------------------------------

    def heading(self, text: str, size: float = 12.5) -> int:
        self.ensure(size * 2.4)
        assert self.page is not None
        self.cursor += size
        self.page.insert_text((MARGIN_X, self.cursor), text, fontsize=size, fontname=BOLD_FONT)
        self.cursor += size * 0.9
        return self.page_number

    def paragraph(self, text: str, size: float = 9.5, leading: float = 1.35) -> int:
        lines = wrap(text, CONTENT_WIDTH, size)
        self.ensure(len(lines) * size * leading + size)
        assert self.page is not None
        start = self.page_number
        for line in lines:
            if self.available < size * leading:
                self.new_page()
            self.cursor += size * leading
            self.page.insert_text((MARGIN_X, self.cursor), line, fontsize=size)
        self.cursor += size * 0.7
        return start

    def bullets(self, items: list[str], size: float = 9.5) -> int:
        start = self.page_number
        for item in items:
            self.paragraph(f"•  {item}", size=size)
        return start

    def table(self, spec: TableSpec) -> int:
        """Ruled table. The grid lines are what make the table detectable."""
        self.ensure(spec.height + 14.0)
        assert self.page is not None
        page = self.page
        widths = spec.column_widths(CONTENT_WIDTH)
        top = self.cursor + 8.0
        bottom = top + spec.height
        start_page = self.page_number

        xs = [MARGIN_X]
        for width in widths:
            xs.append(xs[-1] + width)

        for index in range(len(spec.rows) + 2):
            y = top + index * spec.row_height
            page.draw_line(
                pymupdf.Point(xs[0], y),
                pymupdf.Point(xs[-1], y),
                color=(0.55, 0.55, 0.55),
                width=0.5,
            )
        for x in xs:
            page.draw_line(
                pymupdf.Point(x, top),
                pymupdf.Point(x, bottom),
                color=(0.55, 0.55, 0.55),
                width=0.5,
            )

        baseline = top + spec.row_height - 5.0
        for column, header in enumerate(spec.headers):
            page.insert_text(
                (xs[column] + 4.0, baseline),
                header,
                fontsize=spec.fontsize,
                fontname=BOLD_FONT,
            )
        for index, row in enumerate(spec.rows, start=1):
            y = baseline + index * spec.row_height
            for column in range(len(spec.headers)):
                cell = row[column] if column < len(row) else ""
                available = widths[column] - 8.0
                while cell and text_width(cell, spec.fontsize) > available:
                    cell = cell[:-2]
                page.insert_text((xs[column] + 4.0, y), cell, fontsize=spec.fontsize)

        self.cursor = bottom + 10.0
        return start_page

    def figure(
        self,
        caption: str,
        *,
        height: float,
        draw: Callable[[pymupdf.Page, pymupdf.Rect], None] | None = None,
        image: bytes | None = None,
    ) -> tuple[str, int]:
        """Place a figure and its caption, and return (label, page).

        Caption below the figure, which is where the extractor looks first, and
        long enough on occasion to wrap onto a second line — that wrapping is a
        case the extractor has to get right.
        """
        caption_lines = wrap(f"FIG-00 : {caption}", CONTENT_WIDTH, 8.5)
        needed = height + len(caption_lines) * 12.0 + 20.0
        self.ensure(needed)
        assert self.page is not None

        self.figure_count += 1
        label = f"FIG-{self.figure_count:02d}"
        top = self.cursor + 10.0
        rect = pymupdf.Rect(MARGIN_X + 6, top, PAGE_WIDTH - MARGIN_X - 6, top + height)

        if image is not None:
            self.page.insert_image(rect, stream=image)
        elif draw is not None:
            draw(self.page, rect)

        self.cursor = rect.y1 + 6.0
        page_of_figure = self.page_number
        for line in wrap(f"{label} : {caption}", CONTENT_WIDTH, 8.5):
            self.cursor += 11.5
            self.page.insert_text((MARGIN_X, self.cursor), line, fontsize=8.5)
        self.cursor += 8.0
        return label, page_of_figure

    def cover(self, subtitle: str, lines: list[str]) -> None:
        self.new_page()
        assert self.page is not None
        self.cursor = 120.0
        self.page.insert_text((MARGIN_X, self.cursor), self.title, fontsize=22, fontname=BOLD_FONT)
        self.cursor += 30
        self.page.insert_text((MARGIN_X, self.cursor), subtitle, fontsize=13)
        self.cursor += 40
        for line in lines:
            self.cursor += 16
            self.page.insert_text((MARGIN_X, self.cursor), line, fontsize=10)
        self.cursor += 60
        self.page.draw_rect(
            pymupdf.Rect(MARGIN_X, self.cursor, PAGE_WIDTH - MARGIN_X, self.cursor + 46),
            color=(0.8, 0.3, 0.1),
            width=1.0,
        )
        self.page.insert_text(
            (MARGIN_X + 10, self.cursor + 18),
            "AVERTISSEMENT",
            fontsize=9,
            fontname=BOLD_FONT,
        )
        self.page.insert_text(
            (MARGIN_X + 10, self.cursor + 34),
            "Document fictif genere pour le projet MaintiDoc. "
            "Ne remplace aucune notice constructeur.",
            fontsize=8,
        )
        self.cursor += 70

    def save(self, path: Path) -> int:
        """Write the PDF deterministically.

        Creation and modification dates are pinned and the file /ID is not
        regenerated. Left to their defaults, both carry the current timestamp,
        so two runs of the same seed would produce different bytes — the
        ingestion would re-extract an unchanged corpus every time.
        """
        if self.page is not None:
            self._footer()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.set_metadata(
            {
                "title": self.title,
                "producer": "MaintiDoc synth",
                "creator": "MaintiDoc synth",
                "creationDate": FIXED_DATE,
                "modDate": FIXED_DATE,
            }
        )
        self.doc.save(path, deflate=True, no_new_id=True)
        pages = self.doc.page_count
        self.doc.close()
        return pages
