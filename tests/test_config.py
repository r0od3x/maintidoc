"""Configuration defaults, budgets, and secret hygiene."""

from maintidoc.config import Settings
from maintidoc.versions import capture_versions


def test_no_secret_leaks_through_repr():
    settings = Settings(jwt_secret="super-secret-value", _env_file=None)
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())
    assert settings.jwt_secret.get_secret_value() == "super-secret-value"


def test_budgets_match_the_brief():
    settings = Settings(_env_file=None)
    assert settings.max_mcp_calls == 3
    assert settings.max_model_calls == 6
    assert settings.max_processing_seconds == 60


def test_default_vlm_is_an_actual_vision_model():
    """The prototype's `qwen3.8:27b` is neither a real tag nor multimodal."""
    settings = Settings(_env_file=None)
    assert settings.vlm_model != "qwen3.8:27b"
    assert "vl" in settings.vlm_model.lower()


def test_run_versions_fingerprint_includes_prompt_hashes(prompts_dir):
    versions = capture_versions(Settings(prompts_dir=prompts_dir, _env_file=None))
    assert len(versions.prompts) == 4
    assert all(len(p.sha256) == 64 for p in versions.prompts)
    assert versions.vlm_model and versions.reranker_model
