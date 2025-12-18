"""The agent loop: tool use, step budget, answer assembly and trace shape."""

from __future__ import annotations

import pytest

from src.ragpipe.agent import (
    AgentError,
    AgentStep,
    RAGAgent,
    answer_question,
    compare_agents,
    compose_answer,
    missing_query_terms,
    step_kinds,
    summarise_trace,
)
from src.ragpipe.config import AgentConfig, PipelineConfig, RewriteConfig
from src.ragpipe.fusion import HybridIndex


def test_compose_answer_appends_sources() -> None:
    answer = compose_answer(["First claim.", "Second claim."], ["a", "b", "a"])
    assert answer == "First claim. Second claim. Sources: a, b."
    assert compose_answer([], []) == "No relevant passages were found in the corpus for this question."
    assert compose_answer(["Only claim."], []) == "Only claim."


def test_missing_query_terms_detects_gaps(toy_chunks) -> None:
    assert missing_query_terms("beta zeta", toy_chunks) == ["zeta"]
    assert missing_query_terms("alpha beta", toy_chunks) == []
    assert missing_query_terms("the of", toy_chunks) == []


def test_agent_answers_question_with_evidence(hybrid_index: HybridIndex, pipeline_config: PipelineConfig) -> None:
    agent = RAGAgent(hybrid_index, pipeline_config)
    trace = agent.run("What does the k1 parameter control in BM25?")
    assert trace.question.startswith("What does")
    assert trace.answer
    assert trace.citations
    assert all(chunk_id in trace.evidence_ids for chunk_id in trace.citations)
    assert trace.step_count >= 2
    kinds = step_kinds(trace)
    assert "retrieve" in kinds
    assert "summarise" in kinds
    assert "ground" in kinds
    assert trace.grounding is not None
    assert 0.0 <= trace.grounding.coverage <= 1.0
