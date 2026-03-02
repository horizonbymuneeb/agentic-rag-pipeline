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
