"""Corpus construction: synthetic pages, local files and rejection of bad input."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ragpipe.corpus import (
    SYNTHETIC_PAGES,
    CorpusError,
    Document,
    build_corpus,
    corpus_statistics,
    documents_by_id,
    filter_documents,
    load_corpus,
    synthetic_documents,
    write_corpus,
)


def test_synthetic_corpus_shape_and_uniqueness() -> None:
    documents = build_corpus(seed=0)
    assert len(documents) == len(SYNTHETIC_PAGES) == 8
    assert len({document.doc_id for document in documents}) == 8
    assert all(document.word_count >= 70 for document in documents)
    assert all(document.source == "synthetic" for document in documents)


def test_document_validation() -> None:
    with pytest.raises(CorpusError):
        Document(doc_id="", title="x", text="body")
    with pytest.raises(CorpusError):
        Document(doc_id="x", title="x", text="   ")


def test_document_helpers() -> None:
    document = Document(doc_id="d", title="t", text="one two three four")
    assert document.word_count == 4
    assert document.preview(10).endswith("...")
    assert document.preview(100) == "one two three four"
    assert document.as_dict()["doc_id"] == "d"


def test_build_corpus_rejects_bad_seed_and_duplicates() -> None:
    with pytest.raises(CorpusError):
        build_corpus(seed=-1)
    extra = Document(doc_id=SYNTHETIC_PAGES[0][0], title="dup", text="body text")
    with pytest.raises(CorpusError, match="unique"):
        build_corpus(extra_documents=[extra])


def test_build_corpus_extends_with_caller_documents() -> None:
    extra = Document(doc_id="custom", title="Custom page", text="custom words about vectors")
    documents = build_corpus(extra_documents=[extra])
    assert documents[-1].doc_id == "custom"
    assert len(documents) == len(SYNTHETIC_PAGES) + 1


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    documents = synthetic_documents()[:2]
    paths = write_corpus(documents, tmp_path / "corpus")
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    loaded = load_corpus(tmp_path / "corpus")
    assert [document.doc_id for document in loaded] == [document.doc_id for document in documents]
    assert loaded[0].text.startswith("# ")
    assert loaded[0].title == documents[0].title
