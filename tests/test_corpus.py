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
