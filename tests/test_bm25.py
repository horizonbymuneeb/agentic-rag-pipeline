"""BM25 from scratch: tokenisation, exact IDF, exact scores, ranking and errors."""

from __future__ import annotations

import math

import pytest

from src.ragpipe.bm25 import (
    BM25Error,
    BM25Index,
    build_index,
    compare_queries,
    describe_index,
    score_breakdown,
    term_overlap,
    tokenize,
)
from src.ragpipe.config import RetrievalConfig


def test_tokenize_filters_stopwords_and_short_tokens() -> None:
    assert tokenize("Beta, beta GAMMA the") == ["beta", "beta", "gamma"]
    assert tokenize("a I x of to") == []
    assert tokenize("K1 b0 c3") == ["k1", "b0", "c3"]
    with pytest.raises(BM25Error):
        tokenize("text", min_length=0)


def test_exact_inverse_document_frequency(toy_chunks, bm25_config: RetrievalConfig) -> None:
    index = BM25Index(bm25_config).build(toy_chunks)
    # df(alpha)=1 of N=3 -> ln(1 + (3-1+0.5)/(1+0.5)) = ln(2.6667)
    assert index.inverse_document_frequency("alpha") == pytest.approx(math.log(1 + 2.5 / 1.5))
    assert index.inverse_document_frequency("alpha") == pytest.approx(0.9808292530117263)
    # df(beta)=2 of N=3 -> ln(1 + 1.5/2.5)
    assert index.inverse_document_frequency("beta") == pytest.approx(0.4700036292457356)
    assert index.inverse_document_frequency("missing") == pytest.approx(0.0)


def test_exact_bm25_scores_without_length_normalisation(toy_chunks, bm25_config: RetrievalConfig) -> None:
    index = BM25Index(bm25_config).build(toy_chunks)  # b = 0 -> no length term
    # tf(beta, c0) = 2, k1 = 1.5 -> 2*2.5/(2+1.5) = 1.428571
    assert index.score("beta", 0) == pytest.approx(0.4700036292457356 * (5.0 / 3.5))
    assert index.score("beta", 0) == pytest.approx(0.6714337560653366)
    # tf(beta, c1) = 1 -> 2.5/2.5 = 1.0
    assert index.score("beta", 1) == pytest.approx(0.4700036292457356)
    assert index.score("alpha", 0) == pytest.approx(0.9808292530117263)
    assert index.score("gamma", 0) == pytest.approx(0.0)


def test_length_normalisation_lowers_the_score(toy_chunks) -> None:
    without = BM25Index(RetrievalConfig(bm25_k1=1.5, bm25_b=0.0)).build(toy_chunks)
    with_norm = BM25Index(RetrievalConfig(bm25_k1=1.5, bm25_b=0.75)).build(toy_chunks)
    assert with_norm.score("beta", 0) == pytest.approx(0.6149580195738596)
    assert with_norm.score("beta", 0) < without.score("beta", 0)


def test_index_statistics_exact(toy_chunks) -> None:
    index = BM25Index().build(toy_chunks)
    stats = index.stats()
    assert stats.documents == 3
    assert stats.vocabulary == 5  # alpha beta gamma delta epsilon
    assert index.average_length() == pytest.approx((3 + 2 + 2) / 3)
    assert len(index) == 3


def test_search_ranks_and_drops_zero_scores(toy_chunks, bm25_config: RetrievalConfig) -> None:
    index = BM25Index(bm25_config).build(toy_chunks)
    hits = index.search("beta")
    assert [hit.chunk_id for hit in hits] == ["doc::c0", "doc::c1"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score > hits[1].score
    assert all(hit.source == "bm25" for hit in hits)
    assert index.search("beta", k=1)[0].chunk_id == "doc::c0"


def test_search_unknown_query_returns_nothing(toy_chunks) -> None:
    index = BM25Index().build(toy_chunks)
    assert index.search("the and of") == []
    assert index.search("zeta") == []
    with pytest.raises(BM25Error):
        index.search("beta", k=0)


def test_query_terms_and_unknown_terms(toy_chunks) -> None:
    index = BM25Index().build(toy_chunks)
    assert index.query_terms("beta zeta") == ["beta"]
    assert index.unknown_terms("beta zeta") == ["zeta"]
    assert index.document_frequency("beta") == 2
    assert index.vocabulary == ["alpha", "beta", "delta", "epsilon", "gamma"]


def test_build_and_query_errors() -> None:
    with pytest.raises(BM25Error, match="zero chunks"):
        BM25Index().build([])
    index = BM25Index()
    with pytest.raises(BM25Error, match="call build"):
        index.search("anything")
    with pytest.raises(BM25Error):
        index.chunk_by_id("nope")


def test_score_bounds_and_validation(toy_chunks) -> None:
    index = BM25Index().build(toy_chunks)
    with pytest.raises(BM25Error):
        index.score("beta", 99)
    with pytest.raises(BM25Error):
        index.chunk_by_id("unknown")


def test_helpers_and_index_description(toy_chunks) -> None:
    index = build_index(toy_chunks)
    described = describe_index(index)
    assert described["documents"] == 3
    assert described["top_idf_terms"][0] in {"alpha", "delta", "epsilon", "gamma"}
    mapping = compare_queries(index, ["beta", "gamma"], k=2)
    assert mapping["beta"][0] == "doc::c0"
    assert mapping["gamma"][0] == "doc::c1"


def test_term_overlap_jaccard_exact() -> None:
    assert term_overlap(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert term_overlap(["a"], ["a"]) == pytest.approx(1.0)
    assert term_overlap([], ["a"]) == pytest.approx(0.0)


def test_score_breakdown_sums_to_total(toy_chunks, bm25_config: RetrievalConfig) -> None:
    index = BM25Index(bm25_config).build(toy_chunks)
    breakdown = score_breakdown(index, "beta alpha", "doc::c0")
    assert set(breakdown) == {"beta", "alpha"}
    assert sum(breakdown.values()) == pytest.approx(index.score("beta alpha", 0))
    with pytest.raises(BM25Error):
        score_breakdown(index, "beta", "missing")
