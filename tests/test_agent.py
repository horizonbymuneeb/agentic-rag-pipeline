"""The agent loop: tool use, step budget, answer assembly and trace shape."""

from __future__ import annotations

import pytest

from src.ragpipe.agent import (
    AgentError,
    AgentStep,
    RAGAgent,
    answer_question,
    compare_agents,
    compose_answer,
    missing_query_terms,
    step_kinds,
    summarise_trace,
)
from src.ragpipe.config import AgentConfig, PipelineConfig, RewriteConfig
from src.ragpipe.fusion import HybridIndex
