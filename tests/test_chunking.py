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


def test_fixed_chunks_drop_short_tail() -> None:
    chunks = fixed_size_chunks("abcdefghijklm", size=10, overlap=0, min_chars=4)
    assert [chunk.text for chunk in chunks] == ["abcdefghij"]
    chunks = fixed_size_chunks("abcdefghijklm", size=10, overlap=0, min_chars=1)
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "klm"]


def test_fixed_chunk_validation() -> None:
    with pytest.raises(ChunkError):
        fixed_size_chunks("abc", size=0, overlap=0)
    with pytest.raises(ChunkError):
        fixed_size_chunks("abc", size=5, overlap=5)
    with pytest.raises(ChunkError):
        fixed_size_chunks("   ", size=5, overlap=0)
    with pytest.raises(ChunkError):
        fixed_size_chunks("ab", size=40, overlap=0, min_chars=10)


def test_sentence_chunks_keep_sentences_whole() -> None:
    text = "One sentence is here. Another sentence follows. A third sentence ends it."
    chunks = sentence_aware_chunks(text, max_chars=40, overlap_sentences=0, min_chars=10)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert chunk.text.count(".") == 1


def test_sentence_chunks_overlap_repeats_sentences() -> None:
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three."
    chunks = sentence_aware_chunks(text, max_chars=25, overlap_sentences=1, min_chars=5)
    assert len(chunks) >= 2
    assert chunks[1].text.split(".")[0].strip() in chunks[0].text


def test_long_sentence_becomes_its_own_chunk() -> None:
    long_sentence = "word " * 40 + "end."
    chunks = sentence_aware_chunks(long_sentence + " Short one.", max_chars=50, overlap_sentences=0, min_chars=5)
    assert len(chunks) == 2
    assert chunks[0].length > 50


def test_sentence_chunk_validation() -> None:
    with pytest.raises(ChunkError):
        sentence_aware_chunks("abc.", max_chars=0)
    with pytest.raises(ChunkError):
        sentence_aware_chunks("abc.", max_chars=10, overlap_sentences=-1)
    with pytest.raises(ChunkError):
        sentence_aware_chunks("   ", max_chars=10)


def test_chunk_properties_and_validation() -> None:
    chunk = Chunk(chunk_id="c", doc_id="d", index=0, text="two words", start=0, end=9)
    assert chunk.length == 9
    assert chunk.word_count == 2
    assert chunk.preview(4) == "two ..."
    with pytest.raises(ChunkError):
        Chunk(chunk_id="c", doc_id="d", index=0, text=" ", start=0, end=1)
    with pytest.raises(ChunkError):
        Chunk(chunk_id="c", doc_id="d", index=0, text="ok", start=5, end=2)


def test_chunk_document_prefixes_ids() -> None:
    document = Document(doc_id="page", title="Page", text="First sentence here. Second sentence here.")
    chunks = chunk_document(document, ChunkConfig(mode="fixed", size=25, overlap=5, min_chars=5))
    assert all(chunk.chunk_id.startswith("page::fixed-") for chunk in chunks)
    assert all(chunk.doc_id == "page" for chunk in chunks)


def test_chunker_class_wrappers() -> None:
    chunker = Chunker(ChunkConfig(mode="sentence", max_chars=100, overlap_sentences=0, min_chars=20))
    document = Document(doc_id="d", title="D", text="One sentence here now. Two sentence here now.")
    chunks = chunker.split_documents([document])
    assert len(chunks) >= 1
    inline = chunker.split_text("Standalone text sentence here.", doc_id="inline")
    assert inline[0].doc_id == "inline"
    described = chunker.describe()
    assert described["mode"] == "sentence"


def test_chunk_documents_rejects_duplicate_ids() -> None:
    first = Document(doc_id="same", title="A", text="Sentence one here. Sentence two here.")
    second = Document(doc_id="same", title="B", text="Other sentence one. Other sentence two.")
    with pytest.raises(ChunkError, match="duplicate chunk id"):
        chunk_documents([first, second], ChunkConfig(mode="sentence", max_chars=200, overlap_sentences=0, min_chars=10))


def test_chunk_statistics_and_overlap_ratio(chunks: list[Chunk]) -> None:
    stats = chunk_statistics(chunks)
    assert stats["chunks"] == len(chunks)
    assert stats["documents"] == 8
    assert stats["max_chars"] >= stats["min_chars"]
    ratio = overlap_ratio(chunks)
    assert 0.0 < ratio <= 1.0
    with pytest.raises(ChunkError):
        chunk_statistics([])
