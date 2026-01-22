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
