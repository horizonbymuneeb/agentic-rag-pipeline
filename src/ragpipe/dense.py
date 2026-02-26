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
