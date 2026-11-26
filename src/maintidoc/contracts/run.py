"""Run provenance: what produced a given answer.

The brief requires every execution to be tied to the versions of the model, the
prompt, the reranker and the index. Capturing that is cheap now and impossible
to reconstruct later, so it is a phase 1 contract rather than an MLOps add-on.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class PromptVersion(BaseModel):
    """Identity of a prompt file, content hash included.

    The version tag alone is not enough: an edited `answer_v1.yaml` keeps its
    tag, so the hash is what actually pins the text used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunVersions(BaseModel):
    """Full version fingerprint of one request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_version: str
    vlm_model: str
    embedding_model: str
    visual_encoder: str
    reranker_model: str
    corpus_version: str
    index_version: str
    prompts: tuple[PromptVersion, ...] = ()
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
