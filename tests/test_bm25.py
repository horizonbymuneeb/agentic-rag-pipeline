"""BM25 from scratch: tokenisation, exact IDF, exact scores, ranking and errors."""

from __future__ import annotations

import math

import pytest

from src.ragpipe.bm25 import (
    BM25Error,
    BM25Index,
    build_index,
    compare_queries,
    describe_index,
    score_breakdown,
    term_overlap,
    tokenize,
)
from src.ragpipe.config import RetrievalConfig
