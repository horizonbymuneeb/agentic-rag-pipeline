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
