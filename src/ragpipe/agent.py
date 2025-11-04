"""The agent loop: plan, call tools, observe, answer — under a hard step budget.

Each loop iteration consumes at least one step from ``AgentConfig.max_steps`` and
the loop cannot exceed it.  The trace records every step so a wrong answer can be
explained after the fact, which is the whole point of keeping the agent small.

Order of work for one question:

1. optional query rewriting (one step per retrieval round, bounded by the rewrite
   budget and by the remaining agent budget);
2. ``retrieve`` for the fused top-k passages;
3. ``lookup`` for a question term that the passages do not contain, when the
   budget allows (this is the step that makes the loop "agentic" rather than a
   fixed pipeline);
4. ``summarise`` for an extractive answer;
5. grounding check, and an explicit caveat when the answer is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .bm25 import tokenize
from .chunking import Chunk
from .config import AgentConfig, GroundingConfig, PipelineConfig, RewriteConfig
from .fusion import HybridIndex
from .grounding import GroundingReport, ground_answer
from .rewrite import QueryRewriter, RewriteResult
from .tools import ToolRegistry, build_default_tools, select_sentences


class AgentError(ValueError):
    """Raised when the agent is constructed or driven incorrectly."""


@dataclass(frozen=True)
class AgentStep:
    """One recorded action of the loop."""

    index: int
    kind: str
    detail: str
    ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.index, "kind": self.kind, "detail": self.detail, "ok": self.ok}


@dataclass
class AgentTrace:
    """Everything that happened while answering one question."""

    question: str
    steps: list[AgentStep] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    rewrite: RewriteResult | None = None
    grounding: GroundingReport | None = None
    retrieved_queries: list[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "step_count": self.step_count,
            "steps": [step.as_dict() for step in self.steps],
            "budget_exhausted": self.budget_exhausted,
            "rewrite": self.rewrite.as_dict() if self.rewrite else None,
            "grounding": self.grounding.as_dict() if self.grounding else None,
        }


def compose_answer(sentences: Sequence[str], citations: Sequence[str]) -> str:
    """Join summary sentences and append a sources line.

    The sources line is part of the answer text on purpose: a reader should see
    which chunks were used without opening a separate panel.
    """
    body = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
    if not body:
        return "No relevant passages were found in the corpus for this question."
    if citations:
        return f"{body} Sources: {', '.join(dict.fromkeys(citations))}."
    return body


def missing_query_terms(question: str, chunks: Sequence[Chunk], min_term_length: int = 2) -> list[str]:
    """Question terms that do not appear in any retrieved chunk (lookup candidates)."""
    query_terms = tokenize(question, min_term_length)
    if not query_terms:
        return []
    corpus_text = " ".join(chunk.text.lower() for chunk in chunks)
    return [term for term in dict.fromkeys(query_terms) if term not in corpus_text]


class RAGAgent:
    """Tool-using retrieval agent with a bounded loop."""

    def __init__(
        self,
        index: HybridIndex,
        config: PipelineConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        if len(index) == 0:
            raise AgentError("agent needs a non-empty index")
        self.config = config or PipelineConfig()
        self.index = index
        self.tools = tools or build_default_tools(index, self.config.agent.max_summary_sentences)
        self.rewriter = QueryRewriter(self.config.rewrite, self.config.retrieval.min_term_length)

    # ------------------------------------------------------------------ helpers
    def _retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        return [self.index.chunk_by_id(hit.chunk_id) for hit in self.index.search(query, k)]

    def _remaining(self, budget: int, used: int) -> int:
        return max(0, budget - used)

    # ---------------------------------------------------------------------- run
    def run(self, question: str) -> AgentTrace:
        """Answer ``question`` within ``AgentConfig.max_steps`` steps."""
        if not question or not question.strip():
            raise AgentError("question is empty")
        config: AgentConfig = self.config.agent
        budget = config.max_steps
        used = 0
        trace = AgentTrace(question=question)
        evidence: list[Chunk] = []

        query = question.strip()
        rewrite: RewriteResult | None = None
        if not self.config.rewrite.enabled:
            # Disabled rewriting costs no steps: record the decision, retrieve once.
            rewrite = RewriteResult(original_query=query, final_query=query, rounds=(), stopped_reason="disabled")
        else:
            allowed_rounds = min(self.config.rewrite.max_rounds, self._remaining(budget, used))
            if allowed_rounds > 0:
                rewrite = self.rewriter.run(
                    question,
                    lambda q: self._retrieve_chunks(q, self.config.retrieval.candidate_k),
                    max_rounds=allowed_rounds,
                )
                for round_ in rewrite.rounds:
                    used += 1
                    trace.retrieved_queries.append(round_.query)
                    trace.steps.append(
                        AgentStep(
                            index=used,
                            kind="rewrite_round",
                            detail=f"query={round_.query!r} coverage={round_.coverage:.2f} "
                            f"chunks={len(round_.top_chunk_ids)}",
                        )
                    )
                query = rewrite.final_query
        trace.rewrite = rewrite

        if self._remaining(budget, used) <= 0:
            trace.budget_exhausted = True
            trace.answer = compose_answer([], [])
            return trace

        result = self.tools.call("retrieve", query=query, k=config.top_k)
        used += 1
        trace.steps.append(
            AgentStep(
                index=used,
                kind="retrieve",
                detail=f"tool=retrieve query={query!r} ok={result.ok}",
                ok=result.ok,
            )
        )
        if result.ok:
            evidence = list(result.payload.get("chunks", []))

        # Agentic branch: an unanswered question term may deserve an exact lookup.
        gaps = missing_query_terms(question, evidence, self.config.retrieval.min_term_length)
        if gaps and self._remaining(budget, used) > 0:
            term = gaps[0]
            lookup = self.tools.call("lookup", term=term)
            used += 1
            trace.steps.append(
                AgentStep(
                    index=used,
                    kind="lookup",
                    detail=f"term={term!r} ok={lookup.ok} matches={lookup.payload.get('match_count', 0)}",
                    ok=lookup.ok,
                )
            )
            if lookup.ok:
                known = {chunk.chunk_id for chunk in evidence}
                for match in lookup.payload.get("matches", []):
                    chunk_id = match["chunk_id"]
                    if chunk_id not in known:
                        evidence.append(self.index.chunk_by_id(chunk_id))
                        known.add(chunk_id)

        sentences: list[str] = []
        if evidence and self._remaining(budget, used) > 0:
            summary = self.tools.call(
                "summarise", query=question, chunk_ids=[chunk.chunk_id for chunk in evidence]
            )
            used += 1
            trace.steps.append(
                AgentStep(
                    index=used,
                    kind="summarise",
                    detail=f"tool=summarise chunks={len(evidence)} ok={summary.ok}",
                    ok=summary.ok,
                )
            )
            if summary.ok:
                sentences = list(summary.payload.get("sentences", []))
        if not sentences and evidence:
            sentences = select_sentences(question, evidence, config.max_summary_sentences)

        citations = [chunk.chunk_id for chunk in evidence]
        trace.answer = compose_answer(sentences, citations)
        trace.citations = citations
        trace.evidence_ids = [chunk.chunk_id for chunk in evidence]

        if self.config.agent.require_grounding and evidence and self._remaining(budget, used) > 0:
            report = ground_answer(trace.answer, evidence, self.config.grounding)
            trace.grounding = report
            used += 1
            trace.steps.append(
                AgentStep(
                    index=used,
                    kind="ground",
                    detail=f"claims={len(report.claims)} supported={report.supported_count} "
                    f"coverage={report.coverage:.2f}",
                    ok=report.coverage >= self.config.grounding.support_threshold,
                )
            )
            if report.coverage < self.config.grounding.support_threshold:
                trace.answer = (
                    trace.answer
                    + " Note: part of this answer is not directly supported by the retrieved passages."
                )
        trace.budget_exhausted = used >= budget and not sentences
        if used > budget:  # pragma: no cover - invariant guarded above
            raise AgentError(f"step budget exceeded: {used} > {budget}")
        return trace


def answer_question(question: str, index: HybridIndex, config: PipelineConfig | None = None) -> AgentTrace:
    """One-shot helper: build an agent over ``index`` and answer ``question``."""
    return RAGAgent(index, config).run(question)
