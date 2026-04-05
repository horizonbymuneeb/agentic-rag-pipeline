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


def reciprocal_rank_scores(rankings: Iterable[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Fusion score per chunk id from several ranked id lists."""
    if k <= 0:
        raise FusionError(f"rrf k must be positive, got {k}")
    scores: dict[str, float] = {}
    lists = 0
    for ranking in rankings:
        lists += 1
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    if lists == 0:
        raise FusionError("no rankings supplied")
    return scores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[FusionHit]:
    """Fuse ranked id lists into :class:`FusionHit` objects, best first.

    ``weights`` optionally scales each list's contribution (used for the
    configurable BM25/dense balance).  Ties are broken by the first appearance
    order so the result is deterministic.
    """
    if not rankings:
        raise FusionError("no rankings supplied")
    if weights is not None:
        if len(weights) != len(rankings):
            raise FusionError(f"{len(weights)} weights for {len(rankings)} rankings")
        if any(weight < 0 for weight in weights):
            raise FusionError("fusion weights must be >= 0")
    else:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    order: dict[str, int] = {}
    counter = 0
    for label, ranking in enumerate(rankings):
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            contribution = weights[label] / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            ranks.setdefault(chunk_id, {})[f"list{label}"] = rank
            if chunk_id not in order:
                order[chunk_id] = counter
                counter += 1
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))
    if top_n is not None:
        if top_n <= 0:
            raise FusionError(f"top_n must be positive, got {top_n}")
        ranked = ranked[:top_n]
    return [
        FusionHit(
            chunk_id=chunk_id,
            score=score,
            rank=rank,
            sources=tuple(sorted(ranks[chunk_id])),
            source_ranks=dict(ranks[chunk_id]),
        )
        for rank, (chunk_id, score) in enumerate(ranked, start=1)
    ]


def fuse_hits(
    bm25_hits: Sequence[RetrievalHit],
    dense_hits: Sequence[RetrievalHit],
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_n: int | None = None,
) -> list[FusionHit]:
    """Fuse two lists of :class:`RetrievalHit` (the common call site)."""
    return reciprocal_rank_fusion(
        [[hit.chunk_id for hit in bm25_hits], [hit.chunk_id for hit in dense_hits]],
        k=k,
        weights=[bm25_weight, dense_weight],
        top_n=top_n,
    )


class HybridIndex:
    """Both retrievers over the same chunk corpus, fused with RRF."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.bm25: BM25Index | None = None
        self.dense: DenseIndex | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Sequence[Chunk]) -> "HybridIndex":
        """Build the lexical and dense indexes over ``chunks``."""
        if not chunks:
            raise FusionError("cannot build a hybrid index from zero chunks")
        self._chunks = list(chunks)
        self.bm25 = BM25Index(self.config).build(chunks)
        self.dense = DenseIndex(self.config).build(chunks)
        return self

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], config: RetrievalConfig | None = None) -> "HybridIndex":
        return cls(config).build(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def _require_built(self) -> tuple[BM25Index, DenseIndex]:
        if self.bm25 is None or self.dense is None:
            raise FusionError("hybrid index is empty; call build() first")
        return self.bm25, self.dense

    def lexical_hits(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """BM25-only ranking."""
        bm25, _ = self._require_built()
        return bm25.search(query, k or self.config.candidate_k)

    def dense_hits(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Dense-only ranking."""
        _, dense = self._require_built()
        return dense.search(query, k or self.config.candidate_k)

    def search(self, query: str, k: int | None = None) -> list[FusionHit]:
        """Fused top-``k`` results for ``query``."""
        top_k = self.config.top_k if k is None else int(k)
        if top_k <= 0:
            raise FusionError(f"k must be positive, got {k}")
        lexical = self.lexical_hits(query, self.config.candidate_k)
        dense = self.dense_hits(query, self.config.candidate_k)
        return fuse_hits(
            lexical,
            dense,
            k=self.config.rrf_k,
            bm25_weight=self.config.bm25_weight,
            dense_weight=self.config.dense_weight,
            top_n=top_k,
        )

    def chunk_by_id(self, chunk_id: str) -> Chunk:
        """Look up a chunk by id."""
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise BM25Error(f"unknown chunk id {chunk_id!r}")

    def chunks(self) -> list[Chunk]:
        """All indexed chunks."""
        self._require_built()
        return list(self._chunks)

    def source_agreement(self, query: str, k: int = 5) -> dict[str, Any]:
        """How much the two retrievers agree in their top ``k`` (a tuning diagnostic)."""
        lexical = {hit.chunk_id for hit in self.lexical_hits(query, k)}
        dense = {hit.chunk_id for hit in self.dense_hits(query, k)}
        union = lexical | dense
        return {
            "query": query,
            "lexical_only": sorted(lexical - dense),
            "dense_only": sorted(dense - lexical),
            "shared": sorted(lexical & dense),
            "jaccard": len(lexical & dense) / len(union) if union else 0.0,
        }

    def stats(self) -> dict[str, Any]:
        """Both child indexes' statistics plus the fusion constants."""
        bm25, dense = self._require_built()
        return {
            "chunks": len(self._chunks),
            "bm25": bm25.stats().as_dict(),
            "dense": dense.embedder.stats().as_dict(),
            "rrf_k": self.config.rrf_k,
            "bm25_weight": self.config.bm25_weight,
            "dense_weight": self.config.dense_weight,
        }


def compare_fusions(
    bm25_hits: Sequence[RetrievalHit], dense_hits: Sequence[RetrievalHit], k_values: Sequence[int]
) -> dict[int, list[str]]:
    """Fused rankings for several ``k`` constants (used by the fusion tests)."""
    return {
        int(k): [hit.chunk_id for hit in fuse_hits(bm25_hits, dense_hits, k=int(k))]
        for k in k_values
    }
