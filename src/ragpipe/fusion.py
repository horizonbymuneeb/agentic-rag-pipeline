"""Hybrid retrieval by reciprocal rank fusion of the lexical and dense indexes.

RRF is used instead of score blending because BM25 scores are unbounded and
corpus-dependent while cosine similarities live in ``[-1, 1]``; adding them would
require a normalisation step that is itself a tuning decision.  Ranks are
comparable, so fusion only needs one constant::

    score(d) = sum over lists L containing d of 1 / (k + rank_L(d))

with ``k = 60`` by default.  The constant damps the top ranks so one retriever
cannot dominate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .bm25 import BM25Error, BM25Index, RetrievalHit
from .chunking import Chunk
from .config import RetrievalConfig
from .dense import DenseIndex


class FusionError(ValueError):
    """Raised for empty rankings or invalid fusion parameters."""


@dataclass(frozen=True)
class FusionHit:
    """A chunk that survived fusion, with per-source ranks for traceability."""

    chunk_id: str
    score: float
    rank: int
    sources: tuple[str, ...]
    source_ranks: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "sources": list(self.sources),
            "source_ranks": dict(self.source_ranks),
        }
