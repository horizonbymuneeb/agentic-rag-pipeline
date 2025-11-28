"""Shared fixtures: small hand-written corpora so every expected value is exact."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ragpipe.chunking import Chunk, Chunker  # noqa: E402
from src.ragpipe.config import (  # noqa: E402
    AgentConfig,
    ChunkConfig,
    GroundingConfig,
    PipelineConfig,
    RetrievalConfig,
    RewriteConfig,
)
from src.ragpipe.corpus import Document, build_corpus  # noqa: E402
from src.ragpipe.fusion import HybridIndex  # noqa: E402
