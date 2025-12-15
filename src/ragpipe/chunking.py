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


def fixed_size_chunks(text: str, size: int, overlap: int, min_chars: int = 1) -> list[Chunk]:
    """Character windows of ``size`` with ``overlap`` characters shared.

    The step is ``size - overlap``; when ``overlap >= size`` the configuration
    cannot make progress and is rejected.  Chunks shorter than ``min_chars`` are
    dropped (they are almost always the trailing remainder).
    """
    if size <= 0:
        raise ChunkError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ChunkError(f"overlap must be in [0, size), got {overlap}")
    if not text.strip():
        raise ChunkError("cannot chunk empty text")
    chunks: list[Chunk] = []
    step = size - overlap
    index = 0
    position = 0
    while position < len(text):
        end = min(position + size, len(text))
        piece = text[position:end]
        if len(piece.strip()) >= min_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"fixed-{index}",
                    doc_id="",
                    index=index,
                    text=piece,
                    start=position,
                    end=end,
                )
            )
            index += 1
        if end >= len(text):
            break
        position += step
    if not chunks:
        raise ChunkError("chunking produced no chunk")
    return chunks


def sentence_aware_chunks(
    text: str, max_chars: int, overlap_sentences: int = 1, min_chars: int = 40
) -> list[Chunk]:
    """Pack whole sentences into chunks no longer than ``max_chars`` (plus one sentence).

    ``overlap_sentences`` trailing sentences are repeated at the start of the next
    chunk.  A single sentence longer than ``max_chars`` becomes its own chunk rather
    than being cut, because a split sentence is exactly what this mode exists to
    avoid.
    """
    if max_chars <= 0:
        raise ChunkError(f"max_chars must be positive, got {max_chars}")
    if overlap_sentences < 0:
        raise ChunkError(f"overlap_sentences must be >= 0, got {overlap_sentences}")
    if not text.strip():
        raise ChunkError("cannot chunk empty text")
    sentences = split_sentences(text)
    if not sentences:
        raise ChunkError("no sentences found in text")
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    index = 0
    position = 0

    def flush() -> None:
        nonlocal current, current_len, index, position
        if not current:
            return
        joined = " ".join(current)
        if len(joined.strip()) >= min_chars or not chunks:
            chunks.append(
                Chunk(
                    chunk_id=f"sent-{index}",
                    doc_id="",
                    index=index,
                    text=joined,
                    start=position,
                    end=position + len(joined),
                )
            )
            index += 1
            position += len(joined) + 1
        current = current[-overlap_sentences:] if overlap_sentences else []
        current_len = sum(len(sentence) + 1 for sentence in current)

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            flush()
            addition = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += addition
    if current and (not chunks or " ".join(current) not in [chunk.text for chunk in chunks]):
        joined = " ".join(current)
        if len(joined.strip()) >= min_chars or not chunks:
            chunks.append(
                Chunk(
                    chunk_id=f"sent-{index}",
                    doc_id="",
                    index=index,
                    text=joined,
                    start=position,
                    end=position + len(joined),
                )
            )
    if not chunks:
        raise ChunkError("chunking produced no chunk")
    return chunks


def chunk_document(document: Document, config: ChunkConfig) -> list[Chunk]:
    """Chunk one document according to ``config``, prefixing ids with the doc id."""
    raw = (
        fixed_size_chunks(document.text, config.size, config.overlap, config.min_chars)
        if config.mode == "fixed"
        else sentence_aware_chunks(document.text, config.max_chars, config.overlap_sentences, config.min_chars)
    )
    return [
        Chunk(
            chunk_id=f"{document.doc_id}::{chunk.chunk_id}",
            doc_id=document.doc_id,
            index=chunk.index,
            text=chunk.text,
            start=chunk.start,
            end=chunk.end,
        )
        for chunk in raw
    ]
