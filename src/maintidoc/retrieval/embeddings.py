"""Embedding backends for the text and visual indexes.

Two protocols, so the vector store and the indexer never import a model:

  * `TextEmbedder` — passages and questions, same space (BGE-M3 by default).
  * `VisualEncoder` — images and *queries about images*, same space.

The visual side is deliberately two models. The prototype used CLIP ViT-B/32,
whose text tower is English-only, which is why its interface made the
technician type an English query beside their French question. Pairing the CLIP
image tower with a multilingual text tower aligned to it removes that step: a
French question searches images directly.

`HashEmbedder` is not a toy. Tests must be able to exercise indexing, filtering
and the store without downloading gigabytes, and a deterministic encoder makes
those tests reproducible. It is never a fallback at runtime — asking for a real
model that is unavailable raises rather than silently degrading retrieval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_TEXT_MODEL = "BAAI/bge-m3"
# Apache-2.0. `jinaai/jina-clip-v2` scores better but is CC-BY-NC-4.0, which the
# brief's licence review (§3) would have to accept before it could be used.
DEFAULT_IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
DEFAULT_VISUAL_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


@runtime_checkable
class TextEmbedder(Protocol):
    name: str
    dimension: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


@runtime_checkable
class VisualEncoder(Protocol):
    name: str
    dimension: int

    def encode_images(self, images: list[bytes]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity is a dot product."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


# --- deterministic stand-in ----------------------------------------------


class HashEmbedder:
    """Deterministic pseudo-embeddings over hashed character n-grams.

    Similar strings land near each other, which is enough to assert that
    indexing, filtering and ranking behave. It carries no semantics and must
    never be used to judge retrieval quality.
    """

    def __init__(self, dimension: int = 64, name: str = "hash-test") -> None:
        self.dimension = dimension
        self.name = name

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        cleaned = " ".join((text or "").lower().split())
        tokens = cleaned.split() + [cleaned[i : i + 4] for i in range(0, len(cleaned), 2)]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        if not vector.any():
            vector[0] = 1.0
        return vector

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def encode_images(self, images: list[bytes]) -> np.ndarray:
        digests = [hashlib.sha256(image).hexdigest() for image in images]
        return _normalise(np.stack([self._vector(d) for d in digests]))

    def encode_query(self, text: str) -> np.ndarray:
        return _normalise(self._vector(text))[0]


# --- real backends --------------------------------------------------------


class SentenceTransformerEmbedder:
    """BGE-M3 (or any sentence-transformers model) for text."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(
            model_name, device=device, cache_folder=str(cache_dir) if cache_dir else None
        )
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return _normalise(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )

    def encode_query(self, text: str) -> np.ndarray:
        return _normalise(
            self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        )[0]


class MultilingualClipEncoder:
    """CLIP image tower plus a multilingual text tower aligned to it.

    Both towers must share one space, otherwise a text query and an image
    vector are not comparable and the scores are meaningless. The dimensions
    are checked at construction rather than discovered later as bad results.
    """

    def __init__(
        self,
        image_model: str = DEFAULT_IMAGE_MODEL,
        text_model: str = DEFAULT_VISUAL_TEXT_MODEL,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        folder = str(cache_dir) if cache_dir else None
        self.name = f"{image_model}+{text_model}"
        self._images = SentenceTransformer(image_model, device=device, cache_folder=folder)
        self._texts = SentenceTransformer(text_model, device=device, cache_folder=folder)

        image_dimension = int(self._images.get_sentence_embedding_dimension())
        text_dimension = int(self._texts.get_sentence_embedding_dimension())
        if image_dimension != text_dimension:
            raise ValueError(
                f"Espaces incompatibles : {image_model} produit {image_dimension} "
                f"dimensions, {text_model} en produit {text_dimension}. Les deux "
                f"tours doivent partager le meme espace."
            )
        self.dimension = image_dimension

    def encode_images(self, images: list[bytes]) -> np.ndarray:
        import io

        from PIL import Image

        opened = []
        try:
            for payload in images:
                with Image.open(io.BytesIO(payload)) as handle:
                    opened.append(handle.convert("RGB").copy())
            return _normalise(
                self._images.encode(
                    opened, normalize_embeddings=True, show_progress_bar=False
                )
            )
        finally:
            for image in opened:
                image.close()

    def encode_query(self, text: str) -> np.ndarray:
        return _normalise(
            self._texts.encode([text], normalize_embeddings=True, show_progress_bar=False)
        )[0]


# --- factories ------------------------------------------------------------


def build_text_embedder(settings, *, fake: bool = False) -> TextEmbedder:
    """Real model unless explicitly asked for the deterministic stand-in."""
    if fake:
        return HashEmbedder(name="hash-text")
    return SentenceTransformerEmbedder(
        settings.embedding_model, cache_dir=settings.model_cache_dir
    )


def build_visual_encoder(settings, *, fake: bool = False) -> VisualEncoder:
    if fake:
        return HashEmbedder(name="hash-visual")
    return MultilingualClipEncoder(
        settings.visual_encoder,
        settings.visual_text_encoder,
        cache_dir=settings.model_cache_dir,
    )
