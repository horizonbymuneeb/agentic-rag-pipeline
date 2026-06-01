"""Grounding: claim extraction, token-overlap support and report arithmetic."""

from __future__ import annotations

import pytest

from src.ragpipe.chunking import Chunk
from src.ragpipe.config import GroundingConfig
from src.ragpipe.grounding import (
    GroundingError,
    best_evidence,
    claim_support,
    extract_claims,
    ground_answer,
    grounding_summary,
    token_support,
    unsupported_claims,
)
