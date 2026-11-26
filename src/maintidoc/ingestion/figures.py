"""Figure detection by caption, not by page number.

The prototype decided what a figure was with `kind = "schema" if 2 <= page <= 6`.
That is true of exactly one document and silently mislabels every other one.

Here a figure is something a caption refers to. Captions are matched to the
image region nearest to them, and a caption with no embedded image is taken as
a vector drawing, whose region is rendered — PyMuPDF's `get_images()` does not
report line art, so those figures were invisible to the prototype.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

# `FIG-01 :`, `Figure 3.2 —`, `Schema 4 -`, `Fig. 7 :`
CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<label>FIG[\s._-]?0*(?P<n>\d+)[A-Za-z]?)\s*[:.\u2013\u2014-]\s*", re.I),
    re.compile(r"^(?P<label>Fig(?:ure)?\.?\s+0*(?P<n>\d+)(?:\.\d+)?)\s*[:.\u2013\u2014-]\s*", re.I),
    re.compile(r"^(?P<label>Sch[eé]ma\s+0*(?P<n>\d+)(?:\.\d+)?)\s*[:.\u2013\u2014-]\s*", re.I),
)

# A caption usually sits just under its figure; allow a generous but bounded gap.
MAX_CAPTION_GAP = 90.0
MIN_HORIZONTAL_OVERLAP = 0.25
MIN_RENDER_HEIGHT = 40.0


@dataclass(frozen=True, slots=True)
class Caption:
    label: str
    text: str
    bbox: tuple[float, float, float, float]
    block_indices: tuple[int, ...]

    @property
    def block_index(self) -> int:
        return self.block_indices[0]

    @property
    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.bbox)


def normalise_label(raw: str, number: str) -> str:
    """`fig 1`, `Figure 01`, `FIG-1` all become `FIG-01`, so a reference in the
    body text can be resolved to the same figure later."""
    return f"FIG-{int(number):02d}"


def match_caption(text: str) -> tuple[str, str] | None:
    """Return (normalised label, caption text) when the text opens a caption."""
    stripped = " ".join(text.split())
    for pattern in CAPTION_PATTERNS:
        found = pattern.match(stripped)
        if found:
            return normalise_label(found["label"], found["n"]), stripped
    return None


def find_captions(blocks: list[tuple]) -> list[Caption]:
    """Captions among the page's text blocks, continuation lines absorbed.

    A caption often wraps onto a following block; that block is part of the
    caption and must not be indexed as body text.
    """
    captions: list[Caption] = []
    for index, block in enumerate(blocks):
        x0, y0, x1, y1, text, *_ = block
        found = match_caption(text)
        if found is None:
            continue
        label, caption_text = found
        bbox = [x0, y0, x1, y1]
        indices = [index]

        # Absorb immediately following blocks that continue the sentence.
        look = index + 1
        while look < len(blocks):
            nx0, ny0, nx1, ny1, ntext, *_ = blocks[look]
            gap = ny0 - bbox[3]
            flat = " ".join(ntext.split())
            if gap > 8.0 or not flat or match_caption(ntext) is not None:
                break
            if flat[:1].isupper() and caption_text.rstrip().endswith("."):
                break  # a new sentence that starts a new paragraph
            caption_text = f"{caption_text} {flat}"
            bbox = [min(bbox[0], nx0), bbox[1], max(bbox[2], nx1), ny1]
            indices.append(look)
            look += 1

        captions.append(
            Caption(
                label=label,
                text=caption_text,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                block_indices=tuple(indices),
            )
        )
    return captions


def _horizontal_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    span = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrowest = min(a.width, b.width) or 1.0
    return max(span, 0.0) / narrowest


def pick_image_for_caption(
    caption: Caption, rects: list[pymupdf.Rect]
) -> pymupdf.Rect | None:
    """The image a caption refers to: nearest vertically, above by preference."""
    best: tuple[float, pymupdf.Rect] | None = None
    for rect in rects:
        if _horizontal_overlap(caption.rect, rect) < MIN_HORIZONTAL_OVERLAP:
            continue
        above = caption.rect.y0 - rect.y1
        below = rect.y0 - caption.rect.y1
        if 0 <= above <= MAX_CAPTION_GAP:
            distance = above
        elif 0 <= below <= MAX_CAPTION_GAP:
            distance = below + 1.0  # tie-break towards the figure above
        else:
            continue
        if best is None or distance < best[0]:
            best = (distance, rect)
    return best[1] if best else None


def vector_region_above(
    caption: Caption, page_rect: pymupdf.Rect, obstacles: list[pymupdf.Rect]
) -> pymupdf.Rect | None:
    """Region a caption-without-image most likely describes.

    Bounded below by the caption and above by the nearest text block overlapping
    the same column, so the render captures the drawing and not the prose.
    """
    ceiling = page_rect.y0 + 20.0
    for rect in obstacles:
        if rect.y1 <= caption.rect.y0 and _horizontal_overlap(caption.rect, rect) > 0.1:
            ceiling = max(ceiling, rect.y1)
    region = pymupdf.Rect(
        page_rect.x0 + 20.0, ceiling + 2.0, page_rect.x1 - 20.0, caption.rect.y0 - 2.0
    )
    return region if region.height >= MIN_RENDER_HEIGHT and region.width > 0 else None
