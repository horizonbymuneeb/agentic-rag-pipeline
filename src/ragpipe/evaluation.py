"""Offline evaluation: retrieval recall@k and answer coverage on a fixed question set.

Gold evidence is computed from the corpus, not hard-coded: a question is written
against one source page, and its gold chunks are the chunks that came from that
page.  Recall@k then measures whether the retriever surfaced any of them, and answer
coverage measures how many of the question's expected key terms the produced answer
contains.  Both are proxies -- they reward wording overlap, not correctness -- and
the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .agent import RAGAgent, AgentTrace
from .chunking import Chunk
from .corpus import Document
from .fusion import HybridIndex

METHOD = "recall@k over source-page chunks + key-term answer coverage (proxies, not correctness)"


class EvaluationError(ValueError):
    """Raised for empty question sets or invalid k."""


@dataclass(frozen=True)
class EvalQuestion:
    """One evaluation item: the question, its page, and the terms a good answer uses."""

    question: str
    doc_id: str
    expected_terms: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "doc_id": self.doc_id,
            "expected_terms": list(self.expected_terms),
            "gold_chunk_ids": list(self.gold_chunk_ids),
        }


@dataclass(frozen=True)
class EvalResult:
    """Per-question metrics with the aggregate numbers derived from them."""

    k: int
    questions: tuple[EvalQuestion, ...]
    recalls: tuple[float, ...]
    coverages: tuple[float, ...]

    @property
    def mean_recall(self) -> float:
        return sum(self.recalls) / len(self.recalls) if self.recalls else 0.0

    @property
    def mean_coverage(self) -> float:
        return sum(self.coverages) / len(self.coverages) if self.coverages else 0.0

    @property
    def perfect_recall_count(self) -> int:
        return sum(1 for value in self.recalls if value >= 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "method": METHOD,
            "questions": len(self.questions),
            "recall_at_k": round(self.mean_recall, 4),
            "answer_coverage": round(self.mean_coverage, 4),
            "perfect_recall_count": self.perfect_recall_count,
            "per_question": [
                {
                    "question": question.question,
                    "doc_id": question.doc_id,
                    "recall": round(recall, 4),
                    "coverage": round(coverage, 4),
                }
                for question, recall, coverage in zip(self.questions, self.recalls, self.coverages)
            ],
        }


# Questions written against the synthetic pages in ``corpus.SYNTHETIC_PAGES``.
# Each one uses vocabulary that appears on exactly one page, so the gold set is well
# defined; the expected terms are what a correct extractive answer must contain.
QUESTION_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "What does the k1 parameter control in BM25?",
        "bm25-scoring",
        ("term", "frequency", "saturates"),
    ),
    (
        "How does the b parameter normalise for document length?",
        "bm25-scoring",
        ("length", "normalisation"),
    ),
    (
        "How does reciprocal rank fusion combine ranked lists?",
        "rank-fusion",
        ("sum", "rank", "sixty"),
    ),
    (
        "Why can lexical scores and cosine similarities not be added directly?",
        "rank-fusion",
        ("different", "scales"),
    ),
    (
        "How are hashing embeddings built without a vocabulary?",
        "hashing-embeddings",
        ("buckets", "deterministic", "hash"),
    ),
    (
        "Why does sign hashing help with collisions?",
        "hashing-embeddings",
        ("sign", "collisions"),
    ),
    (
        "What is the difference between fixed-size and sentence-aware chunking?",
        "chunking",
        ("sentence", "character"),
    ),
    (
        "Why does chunk overlap protect against answers spanning a boundary?",
        "chunking",
        ("overlap", "sentence", "boundary"),
    ),
    (
        "What signal is used to rewrite a query?",
        "query-rewriting",
        ("gap", "term", "frequency"),
    ),
    (
        "Why does the rewriting loop need a budget?",
        "query-rewriting",
        ("budget", "retrieval"),
    ),
    (
        "How is answer grounding checked with token overlap?",
        "grounding",
        ("claims", "tokens", "threshold"),
    ),
    (
        "What can the lexical grounding heuristic not detect?",
        "grounding",
        ("entailment", "paraphrase"),
    ),
    (
        "What happens when the agent step budget runs out?",
        "agent-loop",
        ("budget", "answer"),
    ),
    (
        "Why should tools validate their arguments?",
        "agent-loop",
        ("structured", "results"),
    ),
    (
        "What is recall at k measuring?",
        "evaluation",
        ("gold", "chunks", "fraction"),
    ),
    (
        "Which two metrics does the offline evaluation report?",
        "evaluation",
        ("recall", "coverage"),
    ),
)


def gold_chunks_for(doc_id: str, chunks: Sequence[Chunk]) -> tuple[str, ...]:
    """Chunk ids produced from ``doc_id`` (the gold evidence for a page-level question)."""
    gold = tuple(chunk.chunk_id for chunk in chunks if chunk.doc_id == doc_id)
    if not gold:
        raise EvaluationError(f"no chunks found for doc_id {doc_id!r}")
    return gold


def build_eval_set(chunks: Sequence[Chunk], specs: Sequence[tuple[str, str, tuple[str, ...]]] = QUESTION_SPECS) -> list[EvalQuestion]:
    """Materialise :data:`QUESTION_SPECS` against the chunked corpus."""
    if not chunks:
        raise EvaluationError("cannot build an evaluation set from zero chunks")
    questions: list[EvalQuestion] = []
    for question, doc_id, expected_terms in specs:
        questions.append(
            EvalQuestion(
                question=question,
                doc_id=doc_id,
                expected_terms=tuple(expected_terms),
                gold_chunk_ids=gold_chunks_for(doc_id, chunks),
            )
        )
    return questions


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Fraction of gold chunks present in the first ``k`` retrieved ids."""
    if k <= 0:
        raise EvaluationError(f"k must be positive, got {k}")
    if not gold_ids:
        raise EvaluationError("gold_ids is empty")
    top = set(list(retrieved_ids)[:k])
    hits = sum(1 for chunk_id in set(gold_ids) if chunk_id in top)
    return hits / len(set(gold_ids))


def answer_coverage(answer: str, expected_terms: Sequence[str]) -> float:
    """Fraction of expected terms present in the answer (case-insensitive substring)."""
    if not expected_terms:
        raise EvaluationError("expected_terms is empty")
    lowered = answer.lower()
    present = sum(1 for term in expected_terms if term.lower() in lowered)
    return present / len(expected_terms)


def evaluate_retrieval(index: HybridIndex, questions: Sequence[EvalQuestion], k: int = 5) -> dict[str, float]:
    """Mean recall@k of the hybrid retriever over the question set."""
    if not questions:
        raise EvaluationError("no questions to evaluate")
    recalls = []
    for question in questions:
        retrieved = [hit.chunk_id for hit in index.search(question.question, k)]
        recalls.append(recall_at_k(retrieved, question.gold_chunk_ids, k))
    return {"k": float(k), "mean_recall": sum(recalls) / len(recalls), "questions": float(len(questions))}


def evaluate_answers(
    agent: RAGAgent, questions: Sequence[EvalQuestion], k: int = 5
) -> tuple[list[AgentTrace], list[float]]:
    """Run every question through the agent and score answer coverage."""
    if not questions:
        raise EvaluationError("no questions to evaluate")
    traces: list[AgentTrace] = []
    coverages: list[float] = []
    for question in questions:
        trace = agent.run(question.question)
        traces.append(trace)
        coverages.append(answer_coverage(trace.answer, question.expected_terms))
    return traces, coverages
