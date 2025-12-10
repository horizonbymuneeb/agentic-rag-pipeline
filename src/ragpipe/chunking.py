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


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage plus the provenance needed to cite it."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ChunkError(f"chunk {self.chunk_id!r} is empty")
        if self.start < 0 or self.end < self.start:
            raise ChunkError(f"chunk {self.chunk_id!r} has an invalid span {self.start}..{self.end}")

    @property
    def length(self) -> int:
        """Character length of the chunk text."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Whitespace word count."""
        return len(self.text.split())

    def preview(self, chars: int = 70) -> str:
        """Single-line preview for logs and traces."""
        flat = " ".join(self.text.split())
        return flat[:chars] + ("..." if len(flat) > chars else "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences on terminal punctuation plus whitespace.

    Paragraph breaks are treated as sentence boundaries as well, and empty
    fragments are dropped.  Abbreviations such as ``e.g.`` will split, which is
    acceptable for chunking because the pieces are re-joined in order.
    """
    if not text.strip():
        return []
    parts: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        for piece in _SENTENCE_END.split(paragraph.strip()):
            cleaned = " ".join(piece.split())
            if cleaned:
                parts.append(cleaned)
    return parts
