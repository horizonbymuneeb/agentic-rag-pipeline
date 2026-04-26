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


def analyse_retrieval(
    query: str, chunks: Sequence[Chunk], config: RewriteConfig, min_term_length: int = 2
) -> QueryAnalysis:
    """Measure query coverage and collect gap terms from retrieved ``chunks``."""
    query_terms = tuple(dict.fromkeys(tokenize(query, min_term_length)))
    if not query_terms:
        raise RewriteError("query has no usable terms")
    chunk_counts: Counter[str] = Counter()
    for chunk in chunks:
        for term in set(tokenize(chunk.text, min_term_length)):
            chunk_counts[term] += 1
    query_set = set(query_terms)
    covered = tuple(term for term in query_terms if chunk_counts.get(term, 0) > 0)
    coverage = len(covered) / len(query_terms)
    ranked_gaps = sorted(
        ((term, count) for term, count in chunk_counts.items() if term not in query_set and count >= config.min_gap_frequency),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return QueryAnalysis(
        query=query,
        query_terms=query_terms,
        covered_terms=covered,
        gap_terms=tuple(term for term, _ in ranked_gaps),
        gap_frequencies=tuple(ranked_gaps),
        coverage=coverage,
        top_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
    )


class QueryRewriter:
    """Appends the strongest gap terms to the query, up to a hard round budget."""

    def __init__(self, config: RewriteConfig | None = None, min_term_length: int = 2) -> None:
        self.config = config or RewriteConfig()
        if self.config.max_rounds <= 0:
            raise RewriteError("max_rounds must be positive")
        self.min_term_length = int(min_term_length)

    def analyze(self, query: str, chunks: Sequence[Chunk]) -> QueryAnalysis:
        """Public wrapper around :func:`analyse_retrieval`."""
        return analyse_retrieval(query, chunks, self.config, self.min_term_length)

    def rewrite(self, query: str, analysis: QueryAnalysis) -> str:
        """Append up to ``max_new_terms`` gap terms that the query does not contain."""
        if not analysis.gap_terms:
            return query
        additions = [
            term
            for term in analysis.gap_terms
            if term not in set(tokenize(query, self.min_term_length))
        ][: self.config.max_new_terms]
        if not additions:
            return query
        return f"{query.strip()} {' '.join(additions)}".strip()

    def run(
        self,
        query: str,
        retrieve: Callable[[str], Sequence[Chunk]],
        max_rounds: int | None = None,
    ) -> RewriteResult:
        """Run the loop, calling ``retrieve`` once per round.

        ``retrieve`` receives the current query and must return passages in
        relevance order; the loop never inspects scores, only coverage and gaps.
        """
        if not callable(retrieve):
            raise RewriteError("retrieve must be callable")
        rounds_limit = self.config.max_rounds if max_rounds is None else int(max_rounds)
        if rounds_limit < 1:
            raise RewriteError(f"max_rounds must be >= 1, got {rounds_limit}")
        current = query.strip()
        if not current:
            raise RewriteError("query is empty")
        rounds: list[RewriteRound] = []
        reason = "budget"
        if not self.config.enabled:
            chunks = list(retrieve(current))
            analysis = self.analyze(current, chunks)
            rounds.append(
                RewriteRound(0, current, analysis.coverage, analysis.gap_terms, analysis.top_chunk_ids)
            )
            return RewriteResult(query, current, tuple(rounds), "disabled")
        for round_index in range(rounds_limit):
            chunks = list(retrieve(current))
            analysis = self.analyze(current, chunks)
            rounds.append(
                RewriteRound(round_index, current, analysis.coverage, analysis.gap_terms, analysis.top_chunk_ids)
            )
            if analysis.coverage >= self.config.coverage_threshold:
                reason = "coverage"
                break
            if not analysis.has_gaps:
                reason = "no_gaps"
                break
            candidate = self.rewrite(current, analysis)
            if candidate == current:
                reason = "no_change"
                break
            current = candidate
        return RewriteResult(query, current, tuple(rounds), reason)


def rewrite_once(query: str, chunks: Sequence[Chunk], config: RewriteConfig | None = None) -> str:
    """Single-shot rewrite helper (one analysis, one reformulation)."""
    rewriter = QueryRewriter(config)
    analysis = rewriter.analyze(query, chunks)
    return rewriter.rewrite(query, analysis)


def coverage_of_terms(terms: Sequence[str], chunks: Sequence[Chunk]) -> float:
    """Fraction of ``terms`` that appear in at least one chunk."""
    if not terms:
        raise RewriteError("no terms supplied")
    present = 0
    for term in terms:
        if any(term in chunk.text.lower() for chunk in chunks):
            present += 1
    return present / len(terms)


def truncate_query(query: str, max_terms: int) -> str:
    """Keep the first ``max_terms`` tokens of a query (guards against runaway rewrites)."""
    if max_terms <= 0:
        raise RewriteError(f"max_terms must be positive, got {max_terms}")
    terms = query.split()
    return " ".join(terms[:max_terms])
