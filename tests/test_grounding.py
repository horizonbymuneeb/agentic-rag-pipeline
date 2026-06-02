"""Grounding: claim extraction, token-overlap support and report arithmetic."""

from __future__ import annotations

import pytest

from src.ragpipe.chunking import Chunk
from src.ragpipe.config import GroundingConfig
from src.ragpipe.grounding import (
    GroundingError,
    best_evidence,
    claim_support,
    extract_claims,
    ground_answer,
    grounding_summary,
    token_support,
    unsupported_claims,
)


def chunk(index: int, text: str, chunk_id: str | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id or f"c{index}", doc_id="d", index=index, text=text, start=0, end=len(text))


def test_extract_claims_drops_sources_suffix() -> None:
    answer = "BM25 saturates term frequency with k1. The b parameter normalises length. Sources: a, b."
    claims = extract_claims(answer, GroundingConfig(min_claim_tokens=2))
    assert claims == ["BM25 saturates term frequency with k1.", "The b parameter normalises length."]


def test_extract_claims_min_tokens_and_limit() -> None:
    answer = "Tiny. One two three four five six. Seven eight nine ten eleven. Twelve thirteen fourteen."
    claims = extract_claims(answer, GroundingConfig(min_claim_tokens=4, max_claims=1))
    assert len(claims) == 1
    assert claims[0].startswith("One two three four")
    with pytest.raises(GroundingError):
        extract_claims("   ")


def test_claim_support_exact_fractions() -> None:
    chunks = [chunk(0, "alpha beta delta"), chunk(1, "gamma epsilon")]
    support = claim_support("alpha beta gamma", chunks, threshold=0.6)
    assert support.score == pytest.approx(2 / 3)
    assert support.supported is True
    assert support.evidence_id == "c0"
    assert support.matched_terms == ("alpha", "beta")
    strict = claim_support("alpha beta gamma", chunks, threshold=0.8)
    assert strict.supported is False


def test_claim_support_without_overlap() -> None:
    support = claim_support("quantum qubits superconducting", [chunk(0, "alpha beta")], threshold=0.5)
    assert support.score == pytest.approx(0.0)
    assert support.supported is False
    assert support.evidence_id is None


def test_claim_support_validation() -> None:
    with pytest.raises(GroundingError):
        claim_support("alpha beta", [chunk(0, "alpha")], threshold=1.5)
    with pytest.raises(GroundingError):
        claim_support("the of and", [chunk(0, "alpha")])


def test_ground_answer_report_arithmetic() -> None:
    chunks = [chunk(0, "BM25 saturates term frequency with k1 parameter.")]
    answer = (
        "BM25 saturates term frequency with k1. "
        "Quantum annealing solves protein folding. "
        "Sources: c0."
    )
    report = ground_answer(answer, chunks, GroundingConfig(support_threshold=0.6, min_claim_tokens=3, max_claims=5))
    assert len(report.claims) == 2
    assert report.supported_count == 1
    assert report.coverage == pytest.approx(0.5)
    assert report.all_supported is False
    assert unsupported_claims(report) == ["Quantum annealing solves protein folding."]
    assert "heuristic" in report.method
    payload = report.as_dict()
    assert payload["threshold"] == pytest.approx(0.6)
    assert len(payload["unsupported"]) == 1


def test_ground_answer_all_supported() -> None:
    chunks = [chunk(0, "recall at k measures gold chunks in the first k results")]
    report = ground_answer("Recall at k measures gold chunks.", chunks, GroundingConfig(support_threshold=0.6))
    assert report.all_supported is True
    assert report.coverage == pytest.approx(1.0)


def test_ground_answer_validation() -> None:
    with pytest.raises(GroundingError):
        ground_answer("Some claim about alpha.", [], GroundingConfig())


def test_token_support_and_best_evidence() -> None:
    chunks = [chunk(0, "alpha beta gamma"), chunk(1, "delta alpha")]
    assert token_support("alpha beta", chunks[0]) == pytest.approx(1.0)
    assert token_support("alpha beta delta", chunks[0]) == pytest.approx(2 / 3)
    best = best_evidence("alpha beta gamma", chunks)
    assert best is not None
    assert best[0] == "c0"
    assert best[1] == pytest.approx(1.0)
    assert best_evidence("quantum", chunks) is None
    with pytest.raises(GroundingError):
        token_support("the of", chunks[0])
    with pytest.raises(GroundingError):
        best_evidence("alpha", [])


def test_grounding_summary_keys() -> None:
    report = ground_answer("BM25 saturates term frequency.", [chunk(0, "BM25 saturates term frequency")])
    summary = grounding_summary(report)
    assert summary["claims"] == 1
    assert summary["supported"] == 1
    assert summary["all_supported"] is True
    assert "heuristic" in summary["method"]
