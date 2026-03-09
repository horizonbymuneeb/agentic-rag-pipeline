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
