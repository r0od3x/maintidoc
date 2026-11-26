"""Runtime configuration.

All configuration, and every secret without exception, is read from the
environment at startup. Nothing here carries a real credential as a default,
and secrets use `SecretStr` so they cannot leak through `repr()`, logs or a
serialized settings dump.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CORPUS_DIR = PROJECT_ROOT / "corpus"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="MAINTIDOC_",
        extra="ignore",
    )

    app_version: str = "0.1.0"

    # --- Models -----------------------------------------------------------
    # The prototype used `qwen3.8:27b`, which is not a real Ollama tag and is
    # not multimodal. The default below follows the brief (§3).
    vlm_model: str = "qwen2.5vl:7b"
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "BAAI/bge-m3"
    # Two towers sharing one space. The image tower is CLIP ViT-B/32; the text
    # tower is its multilingual counterpart, so a French question searches
    # images directly. The prototype's single English tower forced the
    # technician to retype their question in English.
    # Both Apache-2.0. `jinaai/jina-clip-v2` scores better but is CC-BY-NC-4.0.
    visual_encoder: str = "sentence-transformers/clip-ViT-B-32"
    visual_text_encoder: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    model_cache_dir: Path | None = None

    # --- OCR (phase 2) ----------------------------------------------------
    # Frequently installed outside PATH on Windows; leave empty to autodetect.
    tesseract_cmd: Path | None = None
    tessdata_dir: Path | None = None

    # --- Stores (phases 2-3) ---------------------------------------------
    qdrant_url: str = "http://127.0.0.1:6333"
    # Local mode by default: same engine, no server. The URL is used in phase 10.
    qdrant_path: Path = CORPUS_DIR / "derived" / "qdrant"
    qdrant_local: bool = True
    postgres_dsn: SecretStr = SecretStr("")
    minio_endpoint: str = "http://127.0.0.1:9000"
    minio_access_key: SecretStr = SecretStr("")
    minio_secret_key: SecretStr = SecretStr("")

    # --- Security (phases 6-8) -------------------------------------------
    jwt_secret: SecretStr = SecretStr("")
    approval_token_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    # --- Budgets enforced in code (brief §9) ------------------------------
    max_mcp_calls: int = Field(default=3, ge=1)
    max_model_calls: int = Field(default=6, ge=1)
    max_processing_seconds: int = Field(default=60, ge=1)
    max_evidence_chars: int = Field(default=8500, ge=500)
    max_images: int = Field(default=1, ge=0)

    # --- Traceability -----------------------------------------------------
    corpus_version: str = "dev"
    index_version: str = "dev"

    prompts_dir: Path = PROMPTS_DIR


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
