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
