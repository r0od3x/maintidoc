"""Prompts are files in Git, pinned by content hash, not just by tag."""

from pathlib import Path

import pytest

from maintidoc.prompts import load_all, load_prompt

EXPECTED = {"answer", "visual_analysis", "evidence_check", "ticket_draft"}


def test_every_prompt_named_in_the_brief_exists(prompts_dir: Path):
    assert set(load_all(prompts_dir)) == EXPECTED


def test_prompt_carries_a_content_hash(prompts_dir: Path):
    template = load_prompt("answer", "v1", prompts_dir)
    assert len(template.sha256) == 64
    assert template.version_record.name == "answer"


def test_hashes_differ_between_prompts(prompts_dir: Path):
    hashes = {t.sha256 for t in load_all(prompts_dir).values()}
    assert len(hashes) == len(EXPECTED)


def test_declared_placeholders_match_the_template(prompts_dir: Path):
    """A placeholder documented but absent (or vice versa) is a silent bug."""
    for template in load_all(prompts_dir).values():
        declared = set(template.metadata.get("placeholders", []))
        values = {name: "x" for name in declared}
        template.render_user(**values)  # raises if the template needs more


def test_missing_placeholder_is_loud(prompts_dir: Path):
    template = load_prompt("answer", "v1", prompts_dir)
    with pytest.raises(KeyError):
        template.render_user(question="q")


def test_evidence_text_with_braces_survives_rendering(prompts_dir: Path):
    """str.format would choke here; string.Template does not."""
    template = load_prompt("answer", "v1", prompts_dir)
    rendered = template.render_user(
        question="q", evidence='{"json": "dans un tableau"}', allowed_refs="[]"
    )
    assert '{"json": "dans un tableau"}' in rendered


def test_unknown_prompt_fails_clearly(prompts_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_prompt("inexistant", "v1", prompts_dir)
