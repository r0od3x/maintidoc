"""Phases 3 et 4 — indexation, puis recuperation sourcee.

    from maintidoc.retrieval import Indexer, Retriever, VectorStore
"""

from maintidoc.retrieval.embeddings import (
    HashEmbedder,
    MultilingualClipEncoder,
    SentenceTransformerEmbedder,
    TextEmbedder,
    VisualEncoder,
    build_text_embedder,
    build_visual_encoder,
)
from maintidoc.retrieval.fusion import FusedCandidate, reciprocal_rank_fusion
from maintidoc.retrieval.indexer import Indexer, IndexJournal, IndexRun
from maintidoc.retrieval.lexical import Bm25Index, build_lexical_index, tokenize
from maintidoc.retrieval.rerank import (
    CrossEncoderReranker,
    NoReranker,
    Reranker,
    build_reranker,
)
from maintidoc.retrieval.retriever import (
    AccessDenied,
    RetrievalResult,
    RetrievalTrace,
    Retriever,
)
from maintidoc.retrieval.store import (
    SearchHit,
    Space,
    VectorStore,
    access_filter,
    element_payload,
)

__all__ = [
    "AccessDenied",
    "Bm25Index",
    "CrossEncoderReranker",
    "FusedCandidate",
    "HashEmbedder",
    "IndexJournal",
    "IndexRun",
    "Indexer",
    "MultilingualClipEncoder",
    "NoReranker",
    "Reranker",
    "RetrievalResult",
    "RetrievalTrace",
    "Retriever",
    "SearchHit",
    "SentenceTransformerEmbedder",
    "Space",
    "TextEmbedder",
    "VectorStore",
    "VisualEncoder",
    "access_filter",
    "build_lexical_index",
    "build_reranker",
    "build_text_embedder",
    "build_visual_encoder",
    "element_payload",
    "reciprocal_rank_fusion",
    "tokenize",
]
