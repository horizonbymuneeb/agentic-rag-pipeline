"""The tools the agent may call, and a registry that validates their arguments.

Three tools, each with a narrow job:

* ``retrieve`` -- hybrid search over the chunk corpus.
* ``lookup``   -- does an exact term occur, and where?  Useful when a question
  hinges on one identifier that ranking buried.
* ``summarise``-- extractive summary of given chunks, ranked by query overlap.

A tool returns a :class:`ToolResult` instead of raising on bad input, because the
agent loop must be able to observe a failure and continue; programming errors still
raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .bm25 import tokenize
from .chunking import Chunk, split_sentences
from .fusion import HybridIndex


class ToolError(ValueError):
    """Raised when a tool is unknown or its arguments are structurally invalid."""
