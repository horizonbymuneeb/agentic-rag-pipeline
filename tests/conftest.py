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


def make_chunk(index: int, text: str, doc_id: str = "doc", chunk_id: str | None = None) -> Chunk:
    """Chunk helper used by several test modules."""
    return Chunk(
        chunk_id=chunk_id or f"{doc_id}::c{index}",
        doc_id=doc_id,
        index=index,
        text=text,
        start=0,
        end=len(text),
    )


@pytest.fixture
def toy_chunks() -> list[Chunk]:
    """Three short chunks with known term frequencies (see test_bm25)."""
    return [
        make_chunk(0, "alpha beta beta"),
        make_chunk(1, "beta gamma"),
        make_chunk(2, "delta epsilon"),
    ]


@pytest.fixture
def bm25_config() -> RetrievalConfig:
    """BM25 config with length normalisation off, so scores are easy to compute."""
    return RetrievalConfig(bm25_k1=1.5, bm25_b=0.0, top_k=3, candidate_k=5, embedding_dim=256)


@pytest.fixture
def documents() -> list[Document]:
    """The built-in offline corpus (eight technical pages)."""
    return build_corpus(seed=0)


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Default pipeline configuration."""
    return PipelineConfig()


@pytest.fixture
def chunks(documents: list[Document], pipeline_config: PipelineConfig) -> list[Chunk]:
    """Sentence-aware chunks of the built-in corpus."""
    return Chunker(pipeline_config.chunking).split_documents(documents)
