"""Query rewriting: coverage, gap terms, budgets and stop reasons."""

from __future__ import annotations

import pytest

from src.ragpipe.chunking import Chunk
from src.ragpipe.config import RewriteConfig
from src.ragpipe.rewrite import (
    QueryRewriter,
    RewriteError,
    analyse_retrieval,
    coverage_of_terms,
    rewrite_once,
    truncate_query,
)


def chunk(index: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"c{index}", doc_id="d", index=index, text=text, start=0, end=len(text))


def test_analysis_coverage_and_gap_terms() -> None:
    retrieved = [chunk(0, "bm25 term frequency saturates"), chunk(1, "term frequency saturation")]
    analysis = analyse_retrieval("bm25 k1", retrieved, RewriteConfig(min_gap_frequency=1))
    # "bm25" is covered, "k1" is not
    assert analysis.query_terms == ("bm25", "k1")
    assert analysis.covered_terms == ("bm25",)
    assert analysis.coverage == pytest.approx(0.5)
    assert "k1" not in analysis.gap_terms
    # "term" and "frequency" appear in both chunks, so they rank first by frequency
    assert analysis.gap_terms[:2] == ("frequency", "term")
    assert analysis.has_gaps is True
    assert analysis.top_chunk_ids == ("c0", "c1")


def test_gap_frequency_filter() -> None:
    retrieved = [chunk(0, "alpha beta"), chunk(1, "alpha gamma")]
    strict = analyse_retrieval("alpha", retrieved, RewriteConfig(min_gap_frequency=2))
    assert strict.gap_terms == ()  # every other term appears once
    lenient = analyse_retrieval("alpha", retrieved, RewriteConfig(min_gap_frequency=1))
    assert set(lenient.gap_terms) == {"beta", "gamma"}
