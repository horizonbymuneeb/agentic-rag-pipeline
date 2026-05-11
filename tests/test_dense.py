"""Dense retrieval: deterministic hashing, IDF weighting, cosine search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ragpipe.dense import (
    DenseError,
    DenseIndex,
    HashingTfIdfEmbedder,
    cosine_similarity,
    embed_texts,
    l2_normalise,
    nearest_neighbours,
    token_bucket,
    token_sign,
)
from src.ragpipe.config import RetrievalConfig


def test_token_bucket_and_sign_are_deterministic() -> None:
    # Values pinned so a change of hash function is caught by the suite.
    assert token_bucket("alpha", 64) == 26
    assert token_bucket("alpha", 256) == 154
    assert token_bucket("beta", 256) == 174
    assert token_sign("alpha") in (-1.0, 1.0)
    assert token_sign("alpha") == token_sign("alpha")
    assert token_bucket("alpha", 256) == token_bucket("alpha", 256)
    with pytest.raises(DenseError):
        token_bucket("alpha", 0)


def test_fit_and_idf_values() -> None:
    embedder = HashingTfIdfEmbedder(512).fit(["alpha beta beta", "beta gamma", "delta epsilon"])
    assert embedder.idf("beta") == pytest.approx(0.47000362924573563)
    # unseen terms get the maximum idf: ln(1 + (N + 0.5) / 0.5)
    assert embedder.idf("zzz") == pytest.approx(math.log(8.0))
    with pytest.raises(DenseError, match="not fitted"):
        HashingTfIdfEmbedder(64).idf("alpha")
    with pytest.raises(DenseError):
        HashingTfIdfEmbedder(0)
    with pytest.raises(DenseError):
        HashingTfIdfEmbedder(64).fit([])


def test_embeddings_are_deterministic_and_normalised() -> None:
    embedder = HashingTfIdfEmbedder(256).fit(["alpha beta", "gamma delta"])
    first = embedder.embed("alpha beta")
    second = embedder.embed("alpha beta")
    assert np.array_equal(first, second)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-5)
    assert first.dtype == np.float32


def test_cosine_similarity_bounds() -> None:
    embedder = HashingTfIdfEmbedder(256).fit(["alpha beta", "gamma delta"])
    identical = cosine_similarity(embedder.embed("alpha beta"), embedder.embed("alpha beta"))
    disjoint = cosine_similarity(embedder.embed("alpha beta"), embedder.embed("gamma delta"))
    assert identical == pytest.approx(1.0, abs=1e-6)
    assert disjoint == pytest.approx(0.0)
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0
    with pytest.raises(DenseError):
        cosine_similarity(np.zeros(2), np.zeros(3))


def test_l2_normalise_zero_vector() -> None:
    zero = np.zeros(4, dtype=np.float32)
    assert np.array_equal(l2_normalise(zero), zero)
    scaled = l2_normalise(np.array([3.0, 4.0], dtype=np.float32))
    assert float(np.linalg.norm(scaled)) == pytest.approx(1.0)


def test_embed_batch_shape_and_empty_query() -> None:
    embedder = HashingTfIdfEmbedder(128).fit(["alpha beta", "beta gamma"])
    batch = embedder.embed_batch(["alpha beta", "gamma"])
    assert batch.shape == (2, 128)
    stopword_only = embedder.embed("the and of")
    assert float(np.linalg.norm(stopword_only)) == pytest.approx(0.0)
    with pytest.raises(DenseError):
        embedder.embed_batch([])


def test_dense_index_retrieves_the_exact_chunk(toy_chunks) -> None:
    index = DenseIndex(RetrievalConfig(embedding_dim=256, top_k=2, candidate_k=3)).build(toy_chunks)
    hits = index.search("alpha beta beta")
    assert hits[0].chunk_id == "doc::c0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].source == "dense"
    assert len(hits) <= 2
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_dense_index_validation_and_lookup(toy_chunks) -> None:
    with pytest.raises(DenseError, match="zero chunks"):
        DenseIndex().build([])
    index = DenseIndex().build(toy_chunks)
    assert len(index) == 3
    assert index.matrix.shape == (3, RetrievalConfig().embedding_dim)
    assert index.chunk_by_id("doc::c1").text == "beta gamma"
    with pytest.raises(DenseError):
        index.search("alpha", k=0)
    empty = DenseIndex()
    with pytest.raises(DenseError):
        empty.similarities("alpha")


def test_similarities_length_matches_corpus(toy_chunks) -> None:
    index = DenseIndex().build(toy_chunks)
    scores = index.similarities("beta")
    assert scores.shape == (3,)
    assert scores.max() > 0.0
    # chunk 2 has no query term and no shared buckets
    assert scores[2] == pytest.approx(0.0)


def test_embed_texts_and_nearest_neighbours() -> None:
    texts = ["alpha beta", "beta gamma", "delta epsilon"]
    matrix = embed_texts(texts, dim=256)
    assert matrix.shape == (3, 256)
    embedder = HashingTfIdfEmbedder(256).fit(texts)
    neighbours = nearest_neighbours(embedder, texts, "alpha beta", k=1)
    assert neighbours[0][0] == 0
    assert neighbours[0][1] == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(DenseError):
        nearest_neighbours(embedder, [], "alpha")


def test_embedder_stats() -> None:
    embedder = HashingTfIdfEmbedder(64).fit(["alpha beta", "beta gamma"])
    stats = embedder.stats().as_dict()
    assert stats["dim"] == 64
    assert stats["documents"] == 2
    assert stats["vocabulary"] == 3
