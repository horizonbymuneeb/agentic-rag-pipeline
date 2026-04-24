"""Query rewriting driven by retrieval gaps, with a hard round budget.

The signal used here is a *gap term*: a token that appears repeatedly in the
passages the previous round retrieved but not in the question.  Adding a couple of
those high-frequency terms pulls the query towards the vocabulary of the relevant
documents without throwing away the original wording.  The loop stops early when
the question's terms are already covered or when a rewrite would change nothing,
and it can never run more rounds than ``RewriteConfig.max_rounds``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .bm25 import tokenize
from .chunking import Chunk
from .config import RewriteConfig


class RewriteError(ValueError):
    """Raised for invalid budget values or a missing retrieve callback."""


@dataclass(frozen=True)
class QueryAnalysis:
    """What one retrieval round says about the current query."""

    query: str
    query_terms: tuple[str, ...]
    covered_terms: tuple[str, ...]
    gap_terms: tuple[str, ...]
    gap_frequencies: tuple[tuple[str, int], ...]
    coverage: float
    top_chunk_ids: tuple[str, ...]

    @property
    def has_gaps(self) -> bool:
        """True when there is at least one usable gap term."""
        return bool(self.gap_terms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_terms": list(self.query_terms),
            "covered_terms": list(self.covered_terms),
            "gap_terms": list(self.gap_terms),
            "coverage": round(self.coverage, 4),
            "top_chunk_ids": list(self.top_chunk_ids),
        }


@dataclass(frozen=True)
class RewriteRound:
    """One iteration of the loop: the query used and what it retrieved."""

    round_index: int
    query: str
    coverage: float
    gap_terms: tuple[str, ...]
    top_chunk_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "query": self.query,
            "coverage": round(self.coverage, 4),
            "gap_terms": list(self.gap_terms),
            "top_chunk_ids": list(self.top_chunk_ids),
        }


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of the rewriting loop."""

    original_query: str
    final_query: str
    rounds: tuple[RewriteRound, ...]
    stopped_reason: str

    @property
    def rewritten(self) -> bool:
        """True when the final query differs from the original."""
        return self.final_query.strip() != self.original_query.strip()

    @property
    def round_count(self) -> int:
        """Number of retrieval rounds performed."""
        return len(self.rounds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "final_query": self.final_query,
            "rewritten": self.rewritten,
            "rounds": [round_.as_dict() for round_ in self.rounds],
            "stopped_reason": self.stopped_reason,
        }
