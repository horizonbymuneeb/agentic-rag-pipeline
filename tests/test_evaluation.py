"""Offline evaluation: recall@k, answer coverage and the assembled question set."""

from __future__ import annotations

import pytest

from src.ragpipe.agent import RAGAgent
from src.ragpipe.evaluation import (
    METHOD,
    QUESTION_SPECS,
    EvaluationError,
    EvalResult,
    answer_coverage,
    build_eval_set,
    evaluate_answers,
    evaluate_corpus,
    evaluate_pipeline,
    evaluate_retrieval,
    gold_chunks_for,
    recall_at_k,
)
from src.ragpipe.fusion import HybridIndex


def test_recall_at_k_exact() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["b"], k=4) == pytest.approx(1.0)
    assert recall_at_k(retrieved, ["b"], k=1) == pytest.approx(0.0)
    assert recall_at_k(retrieved, ["a", "d"], k=4) == pytest.approx(1.0)
    assert recall_at_k(retrieved, ["a", "z"], k=4) == pytest.approx(0.5)
    # duplicate gold ids are counted once
    assert recall_at_k(["a"], ["a", "a"], k=1) == pytest.approx(1.0)


def test_recall_at_k_validation() -> None:
    with pytest.raises(EvaluationError):
        recall_at_k(["a"], ["a"], k=0)
    with pytest.raises(EvaluationError):
        recall_at_k(["a"], [], k=1)


def test_answer_coverage_exact() -> None:
    answer = "Term frequency saturates; length normalisation lowers the score."
    assert answer_coverage(answer, ["term", "frequency"]) == pytest.approx(1.0)
    assert answer_coverage(answer, ["term", "frequency", "missing", "absent"]) == pytest.approx(0.5)
    assert answer_coverage(answer, ["quantum"]) == pytest.approx(0.0)
    assert answer_coverage("K1 PARAMETER", ["k1"]) == pytest.approx(1.0)
    with pytest.raises(EvaluationError):
        answer_coverage(answer, [])


def test_build_eval_set_is_materialised_from_chunks(corpus_chunks) -> None:
    questions = build_eval_set(corpus_chunks)
    assert len(questions) == len(QUESTION_SPECS) == 16
    chunk_ids = {chunk.chunk_id for chunk in corpus_chunks}
    covered_docs = set()
    for question in questions:
        assert question.gold_chunk_ids
        assert set(question.gold_chunk_ids) <= chunk_ids
        assert question.expected_terms
        covered_docs.add(question.doc_id)
    assert covered_docs == {chunk.doc_id for chunk in corpus_chunks}
    with pytest.raises(EvaluationError):
        build_eval_set([])


def test_gold_chunks_for_unknown_doc(corpus_chunks) -> None:
    with pytest.raises(EvaluationError, match="no chunks found"):
        gold_chunks_for("not-a-page", corpus_chunks)


def test_evaluate_retrieval_bounds(hybrid_index: HybridIndex, corpus_chunks) -> None:
    questions = build_eval_set(corpus_chunks)
    summary = evaluate_retrieval(hybrid_index, questions, k=5)
    assert set(summary) == {"k", "mean_recall", "questions"}
    assert summary["k"] == 5.0
    assert 0.0 <= summary["mean_recall"] <= 1.0
    assert summary["mean_recall"] >= 0.5
    with pytest.raises(EvaluationError):
        evaluate_retrieval(hybrid_index, [], k=5)


def test_evaluate_answers_returns_traces(hybrid_index: HybridIndex, corpus_chunks, pipeline_config) -> None:
    questions = build_eval_set(corpus_chunks)[:3]
    agent = RAGAgent(hybrid_index, pipeline_config)
    traces, coverages = evaluate_answers(agent, questions, k=3)
    assert len(traces) == 3
    assert len(coverages) == 3
    assert all(0.0 <= value <= 1.0 for value in coverages)
    with pytest.raises(EvaluationError):
        evaluate_answers(agent, [], k=3)


def test_evaluate_pipeline_report(hybrid_index: HybridIndex, corpus_chunks, pipeline_config) -> None:
    questions = build_eval_set(corpus_chunks)
    agent = RAGAgent(hybrid_index, pipeline_config)
    report = evaluate_pipeline(hybrid_index, agent, questions, k=5)
    assert isinstance(report, EvalResult)
    assert report.k == 5
    assert len(report.questions) == 16
    assert len(report.recalls) == len(report.coverages) == 16
    assert 0.0 <= report.mean_recall <= 1.0
    assert 0.0 <= report.mean_coverage <= 1.0
    # measured on this corpus; the floor is deliberately below the observed value
    assert report.mean_recall >= 0.8
    assert report.mean_coverage >= 0.7
    assert report.perfect_recall_count >= 8
    payload = report.as_dict()
    assert payload["method"] == METHOD
    assert payload["questions"] == 16
    assert len(payload["per_question"]) == 16
    assert set(payload["per_question"][0]) == {"question", "doc_id", "recall", "coverage"}


def test_evaluate_pipeline_validation(hybrid_index: HybridIndex, pipeline_config) -> None:
    agent = RAGAgent(hybrid_index, pipeline_config)
    with pytest.raises(EvaluationError):
        evaluate_pipeline(hybrid_index, agent, [], k=5)


def test_evaluate_corpus_wrapper(documents, corpus_chunks, hybrid_index: HybridIndex, pipeline_config) -> None:
    agent = RAGAgent(hybrid_index, pipeline_config)
    report = evaluate_corpus(documents, corpus_chunks, hybrid_index, agent, k=3)
    assert report.k == 3
    assert len(report.questions) == 16
    with pytest.raises(EvaluationError):
        evaluate_corpus([], corpus_chunks, hybrid_index, agent, k=3)


def test_eval_result_aggregates() -> None:
    from src.ragpipe.evaluation import EvalQuestion

    question = EvalQuestion(question="q", doc_id="d", expected_terms=("a",), gold_chunk_ids=("c",))
    result = EvalResult(k=3, questions=(question, question), recalls=(1.0, 0.0), coverages=(0.5, 1.0))
    assert result.mean_recall == pytest.approx(0.5)
    assert result.mean_coverage == pytest.approx(0.75)
    assert result.perfect_recall_count == 1
