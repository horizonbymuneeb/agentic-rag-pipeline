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
