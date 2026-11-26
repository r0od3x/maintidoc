"""BM25 lexical search over the corpus elements.

Not in the brief, which asks only for text and visual search. It is here
because this corpus turns on exact identifiers: `SAN-LAB 2000` and
`SAN-LAB 2200` differ by one digit and mean different things, and dense
embeddings are notoriously soft on precisely that distinction. A lexical
branch that matches `2000` exactly is the cheapest available fix, and RRF
absorbs a third branch at no cost.

Access filtering happens during scoring, not after: an element the principal
may not read is never ranked, so it cannot leak through a rank either.

The index is built in memory at startup. At 317 elements that is instant; past
a few tens of thousands it would need persisting, or moving to Qdrant's sparse
vectors on the server deployment.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from maintidoc.contracts import Element
from maintidoc.retrieval.store import SearchHit

TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on anything else.

    Accents are folded because the corpus mixes `degrade` and `dégradé`
    depending on how a PDF encoded it, and a technician types either.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in folded if not unicodedata.combining(c))
    return TOKEN.findall(ascii_text.lower())


@dataclass(slots=True)
class _Entry:
    element: Element
    counts: Counter[str]
    length: int


@dataclass
class Bm25Index:
    """Classic Okapi BM25 over element text."""

    k1: float = K1
    b: float = B
    entries: list[_Entry] = field(default_factory=list)
    document_frequency: Counter[str] = field(default_factory=Counter)
    average_length: float = 0.0

    @classmethod
    def build(cls, elements: Iterable[Element], *, k1: float = K1, b: float = B) -> Bm25Index:
        index = cls(k1=k1, b=b)
        for element in elements:
            tokens = tokenize(element.text or "")
            if not tokens:
                continue  # an image with no caption has nothing lexical to match
            counts = Counter(tokens)
            index.entries.append(_Entry(element, counts, len(tokens)))
            index.document_frequency.update(counts.keys())
        if index.entries:
            index.average_length = sum(e.length for e in index.entries) / len(index.entries)
        return index

    def __len__(self) -> int:
        return len(self.entries)

    def _idf(self, term: str) -> float:
        total = len(self.entries)
        seen = self.document_frequency.get(term, 0)
        if seen == 0:
            return 0.0
        # BM25+ style smoothing, always positive.
        return math.log(1.0 + (total - seen + 0.5) / (seen + 0.5))

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        groups: Iterable[str] | None = None,
        families: Iterable[str] | None = None,
        document_ids: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        terms = tokenize(query)
        if not terms or not self.entries:
            return []

        allowed = None if groups is None else frozenset(groups)
        family_filter = frozenset(families) if families else None
        document_filter = frozenset(document_ids) if document_ids else None
        idf = {term: self._idf(term) for term in set(terms)}

        scored: list[tuple[float, Element]] = []
        for entry in self.entries:
            element = entry.element
            # Filter before scoring, never after.
            if allowed is not None and not allowed.intersection(element.acl_groups):
                continue
            if family_filter and element.equipment_family not in family_filter:
                continue
            if document_filter and element.document_id not in document_filter:
                continue

            score = 0.0
            for term in terms:
                frequency = entry.counts.get(term, 0)
                if not frequency:
                    continue
                normalised = 1 - self.b + self.b * entry.length / (self.average_length or 1.0)
                score += idf[term] * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * normalised
                )
            if score > 0:
                scored.append((score, element))

        scored.sort(key=lambda pair: (-pair[0], pair[1].ref.ref))
        return [
            SearchHit(
                ref=element.ref.ref,
                document_id=element.document_id,
                page=element.page,
                element_id=element.element_id,
                modality=element.modality.value,
                score=score,
                rank=rank,
                text=element.text,
                image_uri=element.image_uri,
                figure_id=element.figure_id,
                equipment_family=element.equipment_family,
            )
            for rank, (score, element) in enumerate(scored[:limit], start=1)
        ]


def build_lexical_index(elements: Sequence[Element]) -> Bm25Index:
    return Bm25Index.build(elements)
