"""Dense retrieval with deterministic hashing TF-IDF embeddings (no model download).

A hashing embedder maps each token to a bucket in a fixed-width vector using
BLAKE2b, weights it by an IDF computed over the corpus, and gives each token a
deterministic sign so collisions cancel rather than always adding up.  Vectors are
L2-normalised, which makes the dot product equal to cosine similarity.

This is deliberately not a neural encoder: it captures lexical overlap with term
weighting, costs nothing at query time, and is fully deterministic.  It cannot
capture synonymy, which is why the hybrid retriever fuses it with BM25 instead of
relying on it alone.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .bm25 import BM25Error, RetrievalHit, tokenize
from .chunking import Chunk
from .config import RetrievalConfig


class DenseError(ValueError):
    """Raised for unfitted embedders, bad dimensions or empty input."""


@dataclass(frozen=True)
class EmbeddingStats:
    """Summary of a fitted embedder."""

    dim: int
    documents: int
    vocabulary: int
    mean_norm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "documents": self.documents,
            "vocabulary": self.vocabulary,
            "mean_norm": round(self.mean_norm, 6),
        }


def _hash_token(token: str) -> int:
    """Stable 64-bit hash of a token (independent of Python's hash randomisation)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def token_bucket(token: str, dim: int) -> int:
    """Bucket index for a token in a ``dim``-wide vector."""
    if dim <= 0:
        raise DenseError(f"dim must be positive, got {dim}")
    return _hash_token(token) % int(dim)


def token_sign(token: str) -> float:
    """Deterministic +/-1 sign for a token, taken from the top hash bit."""
    return 1.0 if (_hash_token(token) >> 63) & 1 == 0 else -1.0


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit length (zero vectors are returned unchanged)."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return (array / norm).astype(np.float32)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine similarity of two vectors (0 when either is zero)."""
    left = np.asarray(first, dtype=np.float32).reshape(-1)
    right = np.asarray(second, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise DenseError(f"shape mismatch: {left.shape} vs {right.shape}")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class HashingTfIdfEmbedder:
    """Deterministic hashing TF-IDF encoder.

    ``fit`` collects document frequencies; ``embed`` then produces one vector per
    text.  Unseen terms receive the maximum IDF, which is the usual choice: a term
    the corpus has never seen is maximally informative when it does show up.
    """

    def __init__(self, embedding_dim: int = 256, min_term_length: int = 2) -> None:
        if embedding_dim <= 0:
            raise DenseError(f"embedding_dim must be positive, got {embedding_dim}")
        if min_term_length < 1:
            raise DenseError(f"min_term_length must be >= 1, got {min_term_length}")
        self.dim = int(embedding_dim)
        self.min_term_length = int(min_term_length)
        self._document_frequency: dict[str, int] = {}
        self._documents = 0
        self._fitted = False

    # --------------------------------------------------------------- fitting
    def fit(self, texts: Sequence[str]) -> "HashingTfIdfEmbedder":
        """Compute document frequencies from ``texts``."""
        if not texts:
            raise DenseError("cannot fit on an empty corpus")
        counts: dict[str, int] = {}
        for text in texts:
            for term in set(tokenize(text, self.min_term_length)):
                counts[term] = counts.get(term, 0) + 1
        self._document_frequency = counts
        self._documents = len(texts)
        self._fitted = True
        return self

    def fit_chunks(self, chunks: Sequence[Chunk]) -> "HashingTfIdfEmbedder":
        """Fit on the text of chunks."""
        return self.fit([chunk.text for chunk in chunks])

    # -------------------------------------------------------------- embedding
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise DenseError("embedder is not fitted; call fit() first")

    def idf(self, term: str) -> float:
        """IDF weight for ``term``; unseen terms get the maximum IDF."""
        self._require_fitted()
        df = self._document_frequency.get(term, 0)
        if df == 0:
            return math.log(1.0 + (self._documents + 0.5) / 0.5)
        return math.log(1.0 + (self._documents - df + 0.5) / (df + 0.5))

    def embed(self, text: str) -> np.ndarray:
        """Unit-length TF-IDF hashing vector for ``text``."""
        self._require_fitted()
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text, self.min_term_length)
        if not tokens:
            return vector
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        for token, count in frequencies.items():
            weight = (1.0 + math.log(count)) * self.idf(token)
            vector[token_bucket(token, self.dim)] += np.float32(token_sign(token) * weight)
        return l2_normalise(vector)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Matrix of embeddings, shape ``(len(texts), dim)``."""
        if not texts:
            raise DenseError("cannot embed an empty batch")
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)

    def stats(self) -> EmbeddingStats:
        """Vocabulary and vector-shape summary."""
        self._require_fitted()
        return EmbeddingStats(
            dim=self.dim,
            documents=self._documents,
            vocabulary=len(self._document_frequency),
            mean_norm=1.0,
        )


class DenseIndex:
    """Cosine search over a fixed embedding of the chunk corpus."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.embedder = HashingTfIdfEmbedder(self.config.embedding_dim, self.config.min_term_length)
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: Sequence[Chunk]) -> "DenseIndex":
        """Fit the embedder and encode every chunk."""
        if not chunks:
            raise DenseError("cannot build a dense index from zero chunks")
        self._chunks = list(chunks)
        self.embedder.fit_chunks(chunks)
        self._matrix = self.embedder.embed_batch([chunk.text for chunk in chunks])
        return self

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], config: RetrievalConfig | None = None) -> "DenseIndex":
        return cls(config).build(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def matrix(self) -> np.ndarray:
        """``(n_chunks, dim)`` embedding matrix."""
        if self._matrix is None:
            raise DenseError("index is empty; call build() first")
        return self._matrix

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` against every chunk, in insertion order."""
        vector = self.embedder.embed(query)
        return self.matrix @ vector

    def search(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Top ``k`` chunks by cosine similarity, dropping non-positive scores."""
        top_k = self.config.top_k if k is None else int(k)
        if top_k <= 0:
            raise DenseError(f"k must be positive, got {k}")
        scores = self.similarities(query)
        ranked = sorted(
            ((float(score), index) for index, score in enumerate(scores) if score > 0.0),
            key=lambda pair: (-pair[0], pair[1]),
        )[:top_k]
        return [
            RetrievalHit(chunk_id=self._chunks[index].chunk_id, score=score, rank=rank, source="dense")
            for rank, (score, index) in enumerate(ranked, start=1)
        ]

    def chunk_by_id(self, chunk_id: str) -> Chunk:
        """Look up a chunk by id."""
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise BM25Error(f"unknown chunk id {chunk_id!r}")


def embed_texts(texts: Sequence[str], dim: int = 256) -> np.ndarray:
    """Fit-and-embed helper for ad-hoc use in notebooks and tests."""
    return HashingTfIdfEmbedder(dim).fit(texts).embed_batch(texts)


def nearest_neighbours(
    embedder: HashingTfIdfEmbedder, texts: Sequence[str], query: str, k: int = 3
) -> list[tuple[int, float]]:
    """Indices and cosine scores of the ``k`` closest texts to ``query``."""
    if not texts:
        raise DenseError("need at least one text")
    matrix = embedder.embed_batch(texts)
    vector = embedder.embed(query)
    scores = matrix @ vector
    order = np.argsort(-scores)[:k]
    return [(int(index), float(scores[index])) for index in order]
