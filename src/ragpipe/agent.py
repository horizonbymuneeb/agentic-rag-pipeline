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
