"""Chunking: fixed windows, sentence packing, overlap and provenance."""

from __future__ import annotations

import pytest

from src.ragpipe.chunking import (
    Chunker,
    Chunk,
    ChunkError,
    chunk_document,
    chunk_documents,
    chunk_statistics,
    fixed_size_chunks,
    overlap_ratio,
    sentence_aware_chunks,
    split_sentences,
)
from src.ragpipe.config import ChunkConfig
from src.ragpipe.corpus import Document


def test_split_sentences_handles_paragraphs() -> None:
    text = "First sentence here. Second one follows! Third question?\n\nNew paragraph sentence."
    sentences = split_sentences(text)
    assert sentences == [
        "First sentence here.",
        "Second one follows!",
        "Third question?",
        "New paragraph sentence.",
    ]
    assert split_sentences("   ") == []


def test_fixed_chunks_exact_spans() -> None:
    text = "abcdefghij" * 3  # 30 chars
    chunks = fixed_size_chunks(text, size=10, overlap=0, min_chars=1)
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "abcdefghij", "abcdefghij"]
    assert [chunk.start for chunk in chunks] == [0, 10, 20]
    assert [chunk.end for chunk in chunks] == [10, 20, 30]
    assert all(chunk.chunk_id.startswith("fixed-") for chunk in chunks)


def test_fixed_overlap_repeats_text() -> None:
    text = "0123456789"
    chunks = fixed_size_chunks(text, size=6, overlap=2, min_chars=1)
    assert [chunk.text for chunk in chunks] == ["012345", "456789"]
    assert chunks[1].start == 4
