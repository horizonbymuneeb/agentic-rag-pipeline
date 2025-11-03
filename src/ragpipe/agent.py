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
