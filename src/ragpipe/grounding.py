"""Answer grounding: can each claim in the answer be traced to a retrieved chunk?

The check is lexical: a claim is *supported* when the fraction of its content tokens
that appear in some cited chunk is at least ``support_threshold``.  That is a
heuristic, not entailment -- it cannot see that "BM25 uses k1 to normalise length"
borrows the right words while stating the wrong relationship, and it will call a
correct paraphrase with no shared tokens unsupported.  The report labels itself
accordingly so downstream code does not mistake it for a fact check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .bm25 import tokenize
from .chunking import Chunk, split_sentences
from .config import GroundingConfig

METHOD = "lexical token overlap (heuristic, not entailment)"
_SOURCES_MARKER = re.compile(r"\bSources?\s*:", re.IGNORECASE)


class GroundingError(ValueError):
    """Raised when the grounding check is called with nothing to check."""


@dataclass(frozen=True)
class ClaimSupport:
    """One claim and the best evidence found for it."""

    claim: str
    supported: bool
    score: float
    evidence_id: str | None
    matched_terms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "supported": self.supported,
            "score": round(self.score, 4),
            "evidence_id": self.evidence_id,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class GroundingReport:
    """All claims of an answer with their support scores."""

    claims: tuple[ClaimSupport, ...]
    threshold: float
    method: str = METHOD

    @property
    def supported_count(self) -> int:
        """Number of claims above the threshold."""
        return sum(1 for claim in self.claims if claim.supported)

    @property
    def unsupported(self) -> tuple[str, ...]:
        """The claims that failed the check."""
        return tuple(claim.claim for claim in self.claims if not claim.supported)

    @property
    def coverage(self) -> float:
        """Fraction of claims that are supported (``nan``-free: 0.0 when there are none)."""
        if not self.claims:
            return 0.0
        return self.supported_count / len(self.claims)

    @property
    def all_supported(self) -> bool:
        """True when every claim is supported."""
        return bool(self.claims) and self.supported_count == len(self.claims)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "claims": [claim.as_dict() for claim in self.claims],
            "supported": self.supported_count,
            "unsupported": list(self.unsupported),
            "coverage": round(self.coverage, 4),
        }


def extract_claims(answer: str, config: GroundingConfig | None = None) -> list[str]:
    """Split an answer into checkable claims.

    The trailing ``Sources: ...`` line is dropped (it is provenance, not a claim),
    as are sentences with fewer than ``min_claim_tokens`` content tokens.  At most
    ``max_claims`` claims are kept, in order.
    """
    cfg = config or GroundingConfig()
    if not answer or not answer.strip():
        raise GroundingError("cannot extract claims from an empty answer")
    body = _SOURCES_MARKER.split(answer)[0]
    claims: list[str] = []
    for sentence in split_sentences(body):
        if len(tokenize(sentence)) >= cfg.min_claim_tokens:
            claims.append(sentence)
        if len(claims) >= cfg.max_claims:
            break
    return claims


def claim_support(claim: str, chunks: Sequence[Chunk], threshold: float = 0.6) -> ClaimSupport:
    """Best token-overlap support for one claim across ``chunks``."""
    if not 0.0 <= threshold <= 1.0:
        raise GroundingError(f"threshold must be in [0, 1], got {threshold}")
    claim_terms = tokenize(claim)
    if not claim_terms:
        raise GroundingError(f"claim has no content tokens: {claim!r}")
    claim_set = set(claim_terms)
    best_score = 0.0
    best_id: str | None = None
    best_matched: tuple[str, ...] = ()
    for chunk in chunks:
        chunk_terms = set(tokenize(chunk.text))
        matched = claim_set & chunk_terms
        score = len(matched) / len(claim_set)
        if score > best_score:
            best_score = score
            best_id = chunk.chunk_id
            best_matched = tuple(sorted(matched))
    return ClaimSupport(
        claim=claim,
        supported=best_score >= threshold,
        score=best_score,
        evidence_id=best_id,
        matched_terms=best_matched,
    )


def ground_answer(
    answer: str, chunks: Sequence[Chunk], config: GroundingConfig | None = None
) -> GroundingReport:
    """Check every claim of ``answer`` against ``chunks``."""
    cfg = config or GroundingConfig()
    if not chunks:
        raise GroundingError("grounding needs at least one chunk")
    claims = extract_claims(answer, cfg)
    supports = tuple(claim_support(claim, chunks, cfg.support_threshold) for claim in claims)
    return GroundingReport(claims=supports, threshold=cfg.support_threshold)


def unsupported_claims(report: GroundingReport) -> list[str]:
    """The claims that need better evidence (empty when everything is supported)."""
    return list(report.unsupported)


def grounding_summary(report: GroundingReport) -> dict[str, Any]:
    """Compact numbers for logs and the evaluation report."""
    return {
        "claims": len(report.claims),
        "supported": report.supported_count,
        "coverage": round(report.coverage, 4),
        "all_supported": report.all_supported,
        "method": report.method,
    }
