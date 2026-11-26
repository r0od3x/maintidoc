"""Reranker adaptation: triplet mining, splits, comparison, tracking.

The training itself is not run — the model does not fit on this machine — but
everything around it is: what goes into training, how it is split, and how the
result would be compared and recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maintidoc.contracts import Element, Modality
from maintidoc.training import (
    ComparisonHarness,
    NullTracker,
    TrainingConfig,
    TripletSplit,
    TripletStats,
    build_split,
    collect_passages,
    format_comparison,
    mine_triplets,
    sanitise_metric_name,
    split_by_model,
    to_training_pairs,
    tracking_run,
    train_reranker,
    write_manifest,
)
from maintidoc.training.triplets import FactPassage

SHA = "f" * 64

MEANINGS = {
    "SAN-LAB 2000": "avertissement thermique sur le controleur A",
    "SAN-LAB 2200": "reconstruction du groupe de disques en cours",
    "SAN-LAB 3000": "chemin Fibre Channel degrade",
    "SAN-RACK 120": "alimentation redondante indisponible",
    "UPS-LAB 500": "fonctionnement sur batterie",
    "UPS-LAB 800": "defaut de ventilation",
    "UPS-RACK 1500": "batterie en fin de vie",
    "UPS-RACK 3000": "bypass automatique actif",
}


def _passages() -> list[FactPassage]:
    return [
        FactPassage(
            model_label=model,
            family="san" if model.startswith("SAN") else "onduleur",
            phrase="voyant orange fixe",
            meaning=meaning,
            document_id=f"{model.replace(' ', '-')}-MANU",
            text=f"Sur le {model}, le voyant orange fixe signale : {meaning}.",
        )
        for model, meaning in MEANINGS.items()
    ]


# --- what makes a negative hard ------------------------------------------


def test_hard_negatives_come_from_the_same_family():
    """The brief's own example: a temperature alarm against a similar
    indicator on other equipment of the same kind."""
    triplets = mine_triplets(_passages(), negatives_per_positive=6, easy_ratio=0.0)
    hard = [t for t in triplets if t.is_hard]
    assert hard
    for triplet in hard:
        assert triplet.negative_document != triplet.positive_document
        assert triplet.model_label not in triplet.negative


def test_hard_negatives_are_lexically_close_but_semantically_different():
    """If the negative shared no words, a bi-encoder would already separate
    them and the reranker would have nothing to learn."""
    triplets = [
        t
        for t in mine_triplets(_passages(), negatives_per_positive=6, easy_ratio=0.0)
        if t.is_hard
    ]
    assert triplets
    sample = triplets[0]
    assert "voyant orange fixe" in sample.positive
    assert "voyant orange fixe" in sample.negative
    assert sample.positive != sample.negative


def test_most_negatives_are_hard():
    stats = TripletStats.of(
        split_by_model(mine_triplets(_passages(), negatives_per_positive=8))
    )
    assert stats.hard_ratio > 0.6


def test_questions_vary_in_phrasing():
    """A reranker trained on one sentence shape learns that shape."""
    triplets = mine_triplets(_passages(), negatives_per_positive=8)
    assert len({t.query for t in triplets}) > 10


def test_triplets_are_deduplicated():
    triplets = mine_triplets(_passages(), negatives_per_positive=12, seed=3)
    keys = [(t.query, t.positive, t.negative) for t in triplets]
    assert len(keys) == len(set(keys))


# --- splitting ------------------------------------------------------------


def test_splits_never_share_a_model():
    """A random split would put a manual in training and its alarm sheet in
    test; they restate the same fact, so the score would measure memorisation."""
    split = split_by_model(mine_triplets(_passages(), negatives_per_positive=6))
    assert split.leaks == set()
    assert split.train and split.validation and split.test


def test_held_out_models_are_drawn_from_both_families():
    split = split_by_model(mine_triplets(_passages(), negatives_per_positive=6))
    held = split.models_of("validation") | split.models_of("test")
    families = {"san" if m.startswith("SAN") else "onduleur" for m in held}
    assert families == {"san", "onduleur"}


def test_splitting_is_deterministic():
    triplets = mine_triplets(_passages(), negatives_per_positive=6)
    first = split_by_model(triplets, seed=11)
    second = split_by_model(triplets, seed=11)
    assert first.models_of("test") == second.models_of("test")


def test_training_refuses_a_leaking_split():
    """A number produced from leaked data is worse than no number."""
    triplets = mine_triplets(_passages(), negatives_per_positive=4)
    leaking = TripletSplit(
        train=tuple(triplets[:4]), validation=tuple(triplets[:2]), test=tuple(triplets[:2])
    )
    with pytest.raises(ValueError, match="[Ff]uite"):
        train_reranker(leaking, TrainingConfig(), dry_run=True)


def test_training_refuses_an_empty_set():
    with pytest.raises(ValueError, match="vide"):
        train_reranker(TripletSplit(), TrainingConfig(), dry_run=True)


# --- training plumbing ----------------------------------------------------


def test_each_triplet_becomes_two_labelled_pairs():
    """A cross-encoder scores one pair at a time."""
    split = split_by_model(mine_triplets(_passages(), negatives_per_positive=4))
    pairs = to_training_pairs(split, "train")
    assert len(pairs) == len(split.train) * 2
    assert {label for _, _, label in pairs} == {0.0, 1.0}


def test_a_dry_run_prepares_without_training():
    split = split_by_model(mine_triplets(_passages(), negatives_per_positive=4))
    result = train_reranker(split, TrainingConfig(), dry_run=True)
    assert result.trained is False
    assert "dry-run" in result.detail
    assert result.stats.total > 0


def test_the_manifest_records_what_would_be_reproduced(tmp_path: Path):
    split = split_by_model(mine_triplets(_passages(), negatives_per_positive=4))
    result = train_reranker(split, TrainingConfig(), dry_run=True)
    path = tmp_path / "manifest.json"
    write_manifest(result, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trained"] is False
    assert payload["config"]["base_model"]
    assert payload["dataset"]["total"] == result.stats.total
    assert payload["dataset"]["leaks"] == []


# --- comparison -----------------------------------------------------------


class _Fixed:
    """A reranker that reverses the order, to make an arm measurably different."""

    name = "fixe"

    def rerank(self, question, candidates, *, limit=None):
        ordered = list(reversed(candidates))
        return ordered[:limit] if limit else ordered


@pytest.fixture
def harness(tmp_path: Path):
    from maintidoc.api.deps import Runtime
    from maintidoc.evaluation import build_dataset

    elements = [
        Element(
            document_id=f"{model.replace(' ', '-')}-MANU",
            document_version="1.0",
            document_sha256=SHA,
            page=2,
            element_id="p02_text_01",
            modality=Modality.TEXT,
            equipment_family="san" if model.startswith("SAN") else "onduleur",
            acl_groups=(
                ("maintenance_san",)
                if model.startswith("SAN")
                else ("maintenance_energie",)
            ),
            text=f"Sur le {model}, le voyant orange fixe signale : {meaning}.",
        )
        for model, meaning in MEANINGS.items()
    ]
    facts = [
        {
            "document_id": f"{model.replace(' ', '-')}-MANU",
            "model": model,
            "family": "san" if model.startswith("SAN") else "onduleur",
            "kind": "manuel",
            "page": 2,
            "subject": f"voyant orange fixe sur {model}",
            "expected": meaning,
        }
        for model, meaning in MEANINGS.items()
    ]
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")

    runtime = Runtime.lean(elements=elements)
    dataset = build_dataset(facts_path)
    yield ComparisonHarness(runtime.retriever, dataset, service=runtime.service)
    runtime.close()


def test_the_comparison_has_the_three_arms_the_brief_asks_for(harness):
    comparison = harness.compare(base=_Fixed(), adapted=None)
    assert [arm.name for arm in comparison.arms] == ["sans_reranker", "base", "adapted"]
    assert comparison.arm("adapted").available is False


def test_a_missing_arm_is_reported_not_imputed(harness):
    comparison = harness.compare(base=None, adapted=None)
    assert comparison.delta_vs_base is None
    assert "incomplete" in comparison.verdict


def test_a_negative_delta_is_reported_plainly(harness):
    """The brief asks for the result even when adaptation does not help."""
    comparison = harness.compare(base=_Fixed(), adapted=_Fixed())
    delta = comparison.delta_vs_base
    assert delta is not None
    assert "degradation" in comparison.verdict or "aucun effet" in comparison.verdict


def test_the_comparison_restores_the_retriever(harness):
    """A measurement must not leave the system altered."""
    before = harness.retriever.reranker
    harness.compare(base=_Fixed(), adapted=None)
    assert harness.retriever.reranker is before


def test_the_comparison_is_printable(harness):
    text = format_comparison(harness.compare(base=_Fixed(), adapted=None))
    assert "sans_reranker" in text
    assert "Verdict" in text


def test_the_comparison_round_trips(harness, tmp_path: Path):
    comparison = harness.compare(base=_Fixed(), adapted=None)
    path = tmp_path / "comparison.json"
    comparison.save(path)
    assert path.is_file()
    assert "sans_reranker" in path.read_text(encoding="utf-8")


# --- tracking -------------------------------------------------------------


def test_metric_names_are_made_acceptable_to_mlflow():
    """MLflow rejects `@`, and the conventional names all contain it."""
    assert sanitise_metric_name("recall@5") == "recall_at_5"
    assert sanitise_metric_name("ndcg@5") == "ndcg_at_5"
    assert sanitise_metric_name("latence_p95_ms") == "latence_p95_ms"


def test_tracking_disabled_yields_a_null_tracker():
    with tracking_run("essai", enabled=False) as tracker:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"b": 2.0})
        assert isinstance(tracker, NullTracker)
        assert tracker.enabled is False
        assert tracker.reason


def test_a_null_tracker_never_claims_to_have_tracked():
    """An untracked run that looks tracked is worse than an obvious one."""
    tracker = NullTracker()
    tracker.log_metrics({"ndcg@5": 0.5})
    assert tracker.enabled is False


# --- the corpus-scale dataset --------------------------------------------


def test_the_real_corpus_yields_a_compliant_dataset():
    """800-1500 triplets, mostly hard, no leakage — the brief's §5."""
    facts_path = Path("eval/datasets/corpus_facts.json")
    split_path = Path("eval/training/triplets.json")
    if not (facts_path.is_file() and split_path.is_file()):
        pytest.skip("triplets non generes (python -m maintidoc.training mine)")

    stats = TripletStats.of(TripletSplit.load(split_path))
    assert 800 <= stats.total <= 1500, stats.total
    assert stats.hard_ratio >= 0.6
    assert stats.leaks == []
    assert stats.families == 2


def test_build_split_needs_an_ingested_corpus(tmp_path: Path):
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps([{"document_id": "ABSENT", "model": "X", "family": "san",
                     "kind": "manuel", "page": 1, "subject": "voyant orange fixe sur X",
                     "expected": "rien"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ingere"):
        build_split(facts, [])


def test_collect_passages_prefers_the_recorded_page():
    elements = [
        Element(
            document_id="D1", document_version="1", document_sha256=SHA, page=page,
            element_id=f"p{page:02d}_text_01", modality=Modality.TEXT,
            equipment_family="san", acl_groups=("maintenance_san",),
            text=f"Page {page} : voyant orange fixe et autres informations.",
        )
        for page in (1, 2, 3)
    ]
    facts = [{"document_id": "D1", "model": "SAN-LAB 2000", "family": "san",
              "kind": "manuel", "page": 2,
              "subject": "voyant orange fixe sur SAN-LAB 2000", "expected": "x"}]
    passages = collect_passages(facts, elements)
    assert passages
    assert "Page 2" in passages[0].text
