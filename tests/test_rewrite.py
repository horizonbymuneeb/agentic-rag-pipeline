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


def test_analysis_requires_query_terms() -> None:
    with pytest.raises(RewriteError):
        analyse_retrieval("the of and", [chunk(0, "alpha beta")], RewriteConfig())


def test_rewrite_appends_limited_gap_terms() -> None:
    rewriter = QueryRewriter(RewriteConfig(max_new_terms=2))
    retrieved = [chunk(0, "alpha beta gamma delta")]
    analysis = rewriter.analyze("alpha", retrieved)
    rewritten = rewriter.rewrite("alpha", analysis)
    assert rewritten.split()[0] == "alpha"
    assert len(rewritten.split()) == 3  # original + 2 new terms
    assert set(rewritten.split()[1:]) <= {"beta", "gamma", "delta"}


def test_rewrite_is_idempotent_when_nothing_to_add() -> None:
    rewriter = QueryRewriter(RewriteConfig())
    retrived = [chunk(0, "alpha beta")]
    analysis = rewriter.analyze("alpha beta gamma", retrived)
    assert rewriter.rewrite("alpha beta gamma", analysis) == "alpha beta gamma"


def test_run_stops_on_coverage(one_round_rewrite: RewriteConfig) -> None:
    rewriter = QueryRewriter(one_round_rewrite)
    result = rewriter.run("alpha", lambda query: [chunk(0, "alpha beta gamma")])
    assert result.round_count == 1
    assert result.stopped_reason == "coverage"
    assert result.rewritten is False


def test_run_rewrites_and_respects_budget() -> None:
    config = RewriteConfig(max_rounds=2, max_new_terms=1, coverage_threshold=0.99, min_gap_frequency=1)
    rewriter = QueryRewriter(config)
    calls: list[str] = []

    def retrieve(query: str) -> list[Chunk]:
        calls.append(query)
        return [chunk(0, "alpha beta gamma delta")]

    result = rewriter.run("alpha k1", retrieve, max_rounds=2)
    assert calls[0] == "alpha k1"
    assert len(result.rounds) == 2
    assert result.rewritten is True
    assert result.final_query.startswith("alpha k1")
    assert result.stopped_reason in {"budget", "no_change", "no_gaps"}


def test_run_disabled_does_single_round() -> None:
    rewriter = QueryRewriter(RewriteConfig(enabled=False))
    result = rewriter.run("alpha", lambda query: [chunk(0, "alpha beta")])
    assert result.round_count == 1
    assert result.stopped_reason == "disabled"
    assert result.final_query == "alpha"


def test_run_validation() -> None:
    rewriter = QueryRewriter(RewriteConfig())
    with pytest.raises(RewriteError):
        rewriter.run("", lambda query: [chunk(0, "alpha")])
    with pytest.raises(RewriteError):
        rewriter.run("alpha", None)  # type: ignore[arg-type]
    with pytest.raises(RewriteError):
        rewriter.run("alpha", lambda query: [chunk(0, "alpha")], max_rounds=0)
    from src.ragpipe.config import ConfigError

    with pytest.raises(ConfigError):
        RewriteConfig(max_rounds=0)


def test_rewrite_once_and_helpers() -> None:
    retrieved = [chunk(0, "alpha beta gamma")]
    rewritten = rewrite_once("alpha", retrieved, RewriteConfig(max_new_terms=1))
    assert rewritten.startswith("alpha ")
    assert coverage_of_terms(["alpha", "zeta"], retrieved) == pytest.approx(0.5)
    assert truncate_query("a b c d", 2) == "a b"
    with pytest.raises(RewriteError):
        coverage_of_terms([], retrieved)
    with pytest.raises(RewriteError):
        truncate_query("a b", 0)


def test_analysis_serialises() -> None:
    analysis = analyse_retrieval("alpha", [chunk(0, "alpha beta")], RewriteConfig())
    payload = analysis.as_dict()
    assert payload["coverage"] == pytest.approx(1.0)
    assert payload["query_terms"] == ["alpha"]
    assert "gap_terms" in payload
