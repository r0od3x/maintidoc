"""Mining (question, positive, hard negative) triplets for the reranker.

The brief asks for 800-1500 verified examples whose negatives are *hard*, and
gives the example to aim at: telling a temperature alarm apart from a similar
indicator belonging to other equipment (§5).

The corpus was built for exactly this. The steady orange indicator exists on
all ten models and means something different on each, so the natural hard
negative is the same sentence about a different model. Those pairs are nearly
identical lexically and completely different semantically, which is where a
bi-encoder fails and a cross-encoder can win.

**Splitting is by model, never at random.** A random split would put the
SAN-LAB 2000 manual in training and its alarm sheet in test; the two restate
the same fact, so the reranker would be scored on something it had memorised.
Splitting by model means a model seen in training never appears in test, and
the measured gain is generalisation rather than recall.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from maintidoc.contracts import Element

# Varied phrasings, because a reranker trained on one sentence shape learns
# that shape. Each is a question a technician could plausibly type.
QUESTION_TEMPLATES = (
    "Que signifie le {phrase} sur le {model} ?",
    "Sur le {model}, que veut dire le {phrase} ?",
    "{model} : interpretation du {phrase}",
    "Le {phrase} du {model} indique quoi ?",
    "J'observe un {phrase} sur le {model}, quelle est la signification ?",
    "Signification du {phrase} pour l'equipement {model}",
    "{phrase} constate sur {model} : que dit la documentation ?",
    "Comment interpreter le {phrase} d'un {model} ?",
    "Quelle est la cause documentee du {phrase} sur le {model} ?",
    "Le {model} affiche un {phrase}. Que faut-il en conclure ?",
)


class Triplet(BaseModel):
    """One training example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    positive: str
    negative: str
    # Kept for auditing and for splitting; never fed to the model.
    model_label: str
    family: str
    positive_document: str
    negative_document: str
    negative_kind: str = "hard"

    @property
    def is_hard(self) -> bool:
        return self.negative_kind == "hard"


class TripletSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: tuple[Triplet, ...] = ()
    validation: tuple[Triplet, ...] = ()
    test: tuple[Triplet, ...] = ()

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def models_of(self, part: str) -> set[str]:
        return {t.model_label for t in getattr(self, part)}

    @property
    def leaks(self) -> set[str]:
        """Models appearing in more than one split. Must be empty."""
        train, validation, test = (
            self.models_of("train"),
            self.models_of("validation"),
            self.models_of("test"),
        )
        return (train & validation) | (train & test) | (validation & test)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TripletSplit:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class FactPassage:
    """A fact and the passage that states it."""

    model_label: str
    family: str
    phrase: str
    meaning: str
    document_id: str
    text: str


def _phrase_of(subject: str) -> str:
    return subject.split(" sur ")[0].strip()


def collect_passages(
    facts: Iterable[dict], elements: Sequence[Element]
) -> list[FactPassage]:
    """Attach each fact to a passage from its own document that states it.

    The fact file records the page; the element text is what the reranker will
    actually score, so the passage must come from the corpus rather than from
    the fact's summary.
    """
    by_document: dict[str, list[Element]] = {}
    for element in elements:
        if element.text:
            by_document.setdefault(element.document_id, []).append(element)

    passages: list[FactPassage] = []
    for fact in facts:
        candidates = by_document.get(fact["document_id"], [])
        if not candidates:
            continue
        phrase = _phrase_of(fact["subject"])
        # Prefer a passage on the recorded page that mentions the colour.
        colour = phrase.split()[1] if len(phrase.split()) > 1 else phrase
        scored = sorted(
            candidates,
            key=lambda e: (
                e.page != fact["page"],
                colour.lower() not in (e.text or "").lower(),
                -len(e.text or ""),
            ),
        )
        passages.append(
            FactPassage(
                model_label=fact["model"],
                family=fact["family"],
                phrase=phrase,
                meaning=fact["expected"],
                document_id=fact["document_id"],
                text=(scored[0].text or "")[:1200],
            )
        )
    return passages


def mine_triplets(
    passages: Sequence[FactPassage],
    *,
    negatives_per_positive: int = 10,
    easy_ratio: float = 0.15,
    seed: int = 20261120,
) -> list[Triplet]:
    """Build triplets, hard negatives first.

    A hard negative is the passage of a **different model in the same family**
    describing the **same indicator**: same words, different answer. A small
    share of easy negatives (other family) is kept so the model still sees the
    obvious case and does not overfit to near-duplicates.
    """
    rng = random.Random(seed)
    by_family: dict[str, list[FactPassage]] = {}
    for passage in passages:
        by_family.setdefault(passage.family, []).append(passage)

    triplets: list[Triplet] = []
    seen: set[tuple[str, str, str]] = set()
    for passage in passages:
        siblings = [
            other
            for other in by_family[passage.family]
            if other.model_label != passage.model_label
            and other.phrase.lower() == passage.phrase.lower()
        ]
        others = [
            other for other in passages if other.family != passage.family
        ]
        if not siblings and not others:
            continue

        for index in range(negatives_per_positive):
            template = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]
            query = template.format(phrase=passage.phrase, model=passage.model_label)

            use_easy = bool(others) and (not siblings or rng.random() < easy_ratio)
            pool = others if use_easy else siblings
            negative = rng.choice(pool)
            key = (query, passage.text, negative.text)
            if key in seen:
                continue
            seen.add(key)
            triplets.append(
                Triplet(
                    query=query,
                    positive=passage.text,
                    negative=negative.text,
                    model_label=passage.model_label,
                    family=passage.family,
                    positive_document=passage.document_id,
                    negative_document=negative.document_id,
                    negative_kind="easy" if use_easy else "hard",
                )
            )
    return triplets


def split_by_model(
    triplets: Sequence[Triplet],
    *,
    validation_models: int = 2,
    test_models: int = 2,
    seed: int = 20261120,
) -> TripletSplit:
    """Disjoint splits by equipment model.

    Models are drawn from both families so neither validation nor test ends up
    testing a single family.
    """
    rng = random.Random(seed)
    families: dict[str, list[str]] = {}
    for triplet in triplets:
        families.setdefault(triplet.family, [])
        if triplet.model_label not in families[triplet.family]:
            families[triplet.family].append(triplet.model_label)

    held_validation: set[str] = set()
    held_test: set[str] = set()
    for models in families.values():
        ordered = sorted(models)
        rng.shuffle(ordered)
        per_family_val = max(1, validation_models // max(len(families), 1))
        per_family_test = max(1, test_models // max(len(families), 1))
        held_validation.update(ordered[:per_family_val])
        held_test.update(ordered[per_family_val : per_family_val + per_family_test])

    def bucket(triplet: Triplet) -> str:
        if triplet.model_label in held_test:
            return "test"
        if triplet.model_label in held_validation:
            return "validation"
        return "train"

    parts: dict[str, list[Triplet]] = {"train": [], "validation": [], "test": []}
    for triplet in triplets:
        parts[bucket(triplet)].append(triplet)

    return TripletSplit(
        train=tuple(parts["train"]),
        validation=tuple(parts["validation"]),
        test=tuple(parts["test"]),
    )


def build_split(
    facts_path: Path,
    elements: Sequence[Element],
    *,
    negatives_per_positive: int = 3,
    seed: int = 20261120,
) -> TripletSplit:
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    passages = collect_passages(facts, elements)
    if not passages:
        raise ValueError(
            "Aucun passage associe aux faits : corpus ingere ? "
            "(python -m maintidoc.ingestion ingest)"
        )
    triplets = mine_triplets(
        passages, negatives_per_positive=negatives_per_positive, seed=seed
    )
    return split_by_model(triplets, seed=seed)


class TripletStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    hard: int
    easy: int
    models: int
    families: int
    sizes: dict[str, int] = Field(default_factory=dict)
    leaks: list[str] = Field(default_factory=list)

    @property
    def hard_ratio(self) -> float:
        return self.hard / self.total if self.total else 0.0

    @classmethod
    def of(cls, split: TripletSplit) -> TripletStats:
        every = [*split.train, *split.validation, *split.test]
        return cls(
            total=len(every),
            hard=sum(1 for t in every if t.is_hard),
            easy=sum(1 for t in every if not t.is_hard),
            models=len({t.model_label for t in every}),
            families=len({t.family for t in every}),
            sizes=split.sizes,
            leaks=sorted(split.leaks),
        )
