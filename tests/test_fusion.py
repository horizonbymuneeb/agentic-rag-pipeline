"""Reciprocal rank fusion and the hybrid index."""

from __future__ import annotations

import pytest

from src.ragpipe.bm25 import RetrievalHit
from src.ragpipe.config import RetrievalConfig
from src.ragpipe.fusion import (
    FusionError,
    FusionHit,
    HybridIndex,
    compare_fusions,
    fuse_hits,
    reciprocal_rank_fusion,
    reciprocal_rank_scores,
)


def hit(chunk_id: str, score: float, rank: int, source: str) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, score=score, rank=rank, source=source)


def test_reciprocal_rank_scores_exact() -> None:
    scores = reciprocal_rank_scores([["a", "b"], ["b", "a"]], k=9)
    assert scores["a"] == pytest.approx(0.1 + 1 / 11)
    assert scores["b"] == pytest.approx(1 / 11 + 0.1)
    single = reciprocal_rank_scores([["only"]], k=9)
    assert single["only"] == pytest.approx(0.1)


def test_reciprocal_rank_scores_dedupe_within_a_list() -> None:
    scores = reciprocal_rank_scores([["a", "a", "b"]], k=9)
    assert scores["a"] == pytest.approx(0.1)
    assert scores["b"] == pytest.approx(1 / 11)


def test_reciprocal_rank_scores_validation() -> None:
    with pytest.raises(FusionError):
        reciprocal_rank_scores([], k=60)
    with pytest.raises(FusionError):
        reciprocal_rank_scores([["a"]], k=0)


def test_fusion_order_uses_first_appearance_for_ties() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=9)
    assert [item.chunk_id for item in fused] == ["a", "b"]
    assert all(isinstance(item, FusionHit) for item in fused)
    assert fused[0].source_ranks == {"list0": 1, "list1": 2}
    assert fused[0].sources == ("list0", "list1")
    assert fused[0].rank == 1


def test_fusion_weights_change_the_order() -> None:
    lexical_first = reciprocal_rank_fusion([["a"], ["b"]], k=9, weights=[2.0, 1.0])
    dense_first = reciprocal_rank_fusion([["a"], ["b"]], k=9, weights=[1.0, 2.0])
    assert lexical_first[0].chunk_id == "a"
    assert dense_first[0].chunk_id == "b"
    assert lexical_first[0].score == pytest.approx(2 * 0.1)


def test_fusion_top_n_and_validation() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=9, top_n=2)
    assert len(fused) == 2
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([], k=9)
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([["a"], ["b"]], k=9, weights=[1.0])
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([["a"]], k=9, weights=[-1.0])
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([["a"]], k=9, top_n=0)


def test_fuse_hits_integration() -> None:
    lexical = [hit("x", 3.0, 1, "bm25"), hit("y", 1.0, 2, "bm25")]
    dense = [hit("y", 0.9, 1, "dense"), hit("z", 0.5, 2, "dense")]
    fused = fuse_hits(lexical, dense, k=60)
    ids = [item.chunk_id for item in fused]
    assert ids == ["y", "x", "z"]
    assert fused[0].source_ranks == {"list0": 2, "list1": 1}
    assert compare_fusions(lexical, dense, [9, 60]).keys() == {9, 60}


def test_hybrid_index_search(hybrid_index: HybridIndex) -> None:
    hits = hybrid_index.search("reciprocal rank fusion", k=3)
    assert 0 < len(hits) <= 3
    assert all(isinstance(item, FusionHit) for item in hits)
    assert hits[0].chunk_id.startswith("rank-fusion") or "rank" in hits[0].chunk_id
    assert [item.rank for item in hits] == list(range(1, len(hits) + 1))
    assert all(hybrid_index.chunk_by_id(item.chunk_id) for item in hits)
