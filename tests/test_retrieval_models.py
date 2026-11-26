"""Acceptance tests for phase 3, against the real index.

These require models downloaded and an index built with them:

    uv run python -m maintidoc.retrieval build --recreate

They are skipped otherwise, rather than silently passing on the deterministic
stand-in — a hash embedder would satisfy the mechanics of every assertion here
while proving nothing about retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maintidoc.config import get_settings
from maintidoc.retrieval import (
    IndexJournal,
    Space,
    VectorStore,
    access_filter,
    build_text_embedder,
    build_visual_encoder,
)

pytestmark = pytest.mark.models

DERIVED = Path(__file__).resolve().parents[1] / "corpus" / "derived"
FAKE_MODELS = {"hash-text", "hash-visual", "hash-test"}


def _require_real_index() -> None:
    journal = IndexJournal(DERIVED / "index_runs.jsonl")
    latest = journal.latest()
    if latest is None:
        pytest.skip("Aucun index construit")
    if latest.text_model in FAKE_MODELS:
        pytest.skip("Index construit avec des embeddings factices")


@pytest.fixture(scope="module")
def store():
    _require_real_index()
    vector_store = VectorStore.local(DERIVED / "qdrant")
    if vector_store.count(Space.TEXT) == 0:
        vector_store.close()
        pytest.skip("Index texte vide")
    yield vector_store
    vector_store.close()


@pytest.fixture(scope="module")
def text_embedder():
    _require_real_index()
    return build_text_embedder(get_settings())


@pytest.fixture(scope="module")
def visual_encoder():
    _require_real_index()
    return build_visual_encoder(get_settings())


# --- the phase 3 acceptance criterion ------------------------------------


def test_a_french_question_finds_a_figure_without_english(store, visual_encoder):
    """The criterion for this phase.

    The prototype shipped a separate box where the technician had to retype
    their question in English, because CLIP ViT-B/32's text tower is English
    only. A French query must now reach the visual index directly.
    """
    hits = store.search(
        Space.VISUAL,
        visual_encoder.encode_query("schema des chemins redondants de la baie de stockage"),
        limit=5,
    )
    assert hits, "la recherche visuelle doit retourner des candidats"
    assert all(hit.image_uri for hit in hits)
    assert any(hit.figure_id for hit in hits[:3])


def test_a_question_and_its_translation_are_aligned(visual_encoder):
    """Cross-lingual alignment holds: a French query lands near its English
    translation, which is what removes the prototype's manual retyping step."""
    pairs = [
        ("front panel with indicator lights", "panneau avant avec voyants"),
        ("redundant paths diagram", "schema des chemins redondants"),
    ]
    for english, french in pairs:
        similarity = float(
            visual_encoder.encode_query(english) @ visual_encoder.encode_query(french)
        )
        assert similarity > 0.80, f"{french!r} trop eloigne de sa traduction ({similarity:.2f})"


def test_known_limit_french_visual_queries_discriminate_poorly(visual_encoder):
    """Documented limitation, measured rather than assumed.

    `clip-ViT-B-32-multilingual-v1` maps French into a narrow cone: two
    unrelated French queries sit around 0.97 cosine, where the English pair
    sits around 0.79. Cross-lingual alignment is fine; it is separation
    *within* French that is poor, so the visual index alone ranks French
    queries almost identically whatever they ask.

    Consequences, both already planned:
      * figures are reached mainly through their captions in the BGE-M3 text
        index, which handles French well (see the test below);
      * phase 4's reranker is what has to repair the ordering.

    This test guards the measurement. If a model swap improves French
    separation, it fails and the limitation should be removed from the README.
    """
    french = float(
        visual_encoder.encode_query("panneau avant avec voyants")
        @ visual_encoder.encode_query("schema des chemins redondants")
    )
    english = float(
        visual_encoder.encode_query("front panel with indicator lights")
        @ visual_encoder.encode_query("redundant paths diagram")
    )
    assert french > english, (
        "la separation du francais s'est amelioree "
        f"(fr={french:.3f}, en={english:.3f}) : mettre a jour la limitation documentee"
    )


def test_a_french_question_reaches_the_right_figure_through_its_caption(
    store, text_embedder
):
    """The acceptance criterion, by the path that actually carries it.

    Figure captions are indexed in the text collection, so a French question
    about a diagram finds the diagram — no English retyping anywhere.
    """
    hits = store.search(
        Space.TEXT,
        text_embedder.encode_query("Quelle figure illustre les chemins redondants ?"),
        limit=5,
    )
    figures = [hit for hit in hits if hit.figure_id]
    assert figures, "aucune figure retrouvee pour une question francaise"
    assert figures[0].rank <= 2, "la figure attendue doit remonter en tete"
    assert "chemins redondants" in (figures[0].text or "")


def test_text_search_finds_the_right_equipment(store, text_embedder):
    """Ten models share the phrase; the model named in the question decides."""
    hits = store.search(
        Space.TEXT,
        text_embedder.encode_query(
            "Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?"
        ),
        limit=10,
    )
    assert hits
    # The corpus gives this phrase ten different meanings; the model named in
    # the question must decide, and it must win outright rather than appear.
    assert "SAN-2000" in hits[0].document_id, (
        f"premier resultat {hits[0].document_id}, attendu un document SAN-2000"
    )


def test_access_filter_applies_to_the_real_index(store, text_embedder):
    query = text_embedder.encode_query("voyant orange fixe")
    energy = store.search(
        Space.TEXT, query, limit=10, query_filter=access_filter(["maintenance_energie"])
    )
    assert energy
    assert all(hit.equipment_family == "onduleur" for hit in energy)


def test_both_collections_are_populated(store):
    assert store.count(Space.TEXT) > 100
    assert store.count(Space.VISUAL) > 10


def test_the_journal_records_the_real_models():
    _require_real_index()
    latest = IndexJournal(DERIVED / "index_runs.jsonl").latest()
    assert latest is not None
    assert latest.text_model not in FAKE_MODELS
    assert latest.text_dimension >= 256
    assert latest.visual_model
    assert latest.visual_dimension
