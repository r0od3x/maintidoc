"""Capture the version fingerprint of a run."""

from __future__ import annotations

from maintidoc.config import Settings, get_settings
from maintidoc.contracts import RunVersions
from maintidoc.prompts import load_all


def capture_versions(settings: Settings | None = None) -> RunVersions:
    """Snapshot models, corpus, index and prompt hashes for one execution."""
    cfg = settings or get_settings()
    prompts = tuple(t.version_record for t in load_all(cfg.prompts_dir).values())
    return RunVersions(
        app_version=cfg.app_version,
        vlm_model=cfg.vlm_model,
        embedding_model=cfg.embedding_model,
        visual_encoder=cfg.visual_encoder,
        reranker_model=cfg.reranker_model,
        corpus_version=cfg.corpus_version,
        index_version=cfg.index_version,
        prompts=prompts,
    )
