"""Chunking: fixed character windows and sentence-aware packing with overlap.

Two modes exist because they trade off differently.  Fixed-size chunks are trivial
to reason about and guaranteed to have a bounded length, but they cut mid-sentence.
Sentence-aware chunks never cut a sentence and keep whole trailing sentences as
overlap, at the cost of a variable chunk length (bounded by ``max_chars`` plus one
long sentence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .config import ChunkConfig
from .corpus import Document


class ChunkError(ValueError):
    """Raised for invalid chunk sizes or overlap settings."""
