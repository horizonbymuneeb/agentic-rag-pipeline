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
