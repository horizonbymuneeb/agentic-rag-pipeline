"""Okapi BM25 implemented from scratch: tokenisation, index, scoring and search.

The formula is the standard one::

    score(q, d) = sum_{t in q} idf(t) * (tf(t,d) * (k1 + 1))
                  / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

``k1`` controls term-frequency saturation and ``b`` controls document-length
normalisation.  Both are read from :class:`~src.ragpipe.config.RetrievalConfig`
and are asserted against hand-computed values in the tests.  There is no stemming:
``retrieve`` and ``retrieval`` are different terms (noted in the README).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .chunking import Chunk
from .config import RetrievalConfig

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does",
        "for", "from", "had", "has", "have", "how", "if", "in", "into", "is", "it", "its",
        "may", "more", "most", "no", "not", "of", "on", "or", "our", "so", "such", "than",
        "that", "the", "their", "then", "there", "these", "they", "this", "those", "to",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "why",
        "will", "with", "would", "you", "your",
    }
)


class BM25Error(ValueError):
    """Raised when the index is queried before being built or with bad parameters."""


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieved chunk with its score and 1-based rank."""

    chunk_id: str
    score: float
    rank: int
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "score": round(self.score, 6), "rank": self.rank, "source": self.source}


def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords and short tokens removed."""
    if min_length < 1:
        raise BM25Error(f"min_length must be >= 1, got {min_length}")
    tokens = _TOKEN.findall(text.lower())
    return [token for token in tokens if len(token) >= min_length and token not in STOPWORDS]


def term_frequencies(tokens: Sequence[str]) -> Counter[str]:
    """Bag of words for one document."""
    return Counter(tokens)


@dataclass(frozen=True)
class BM25Stats:
    """Index-level statistics reported by the CLI and used in the tests."""

    documents: int
    vocabulary: int
    average_length: float
    k1: float
    b: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "vocabulary": self.vocabulary,
            "average_length": round(self.average_length, 4),
            "k1": self.k1,
            "b": self.b,
        }


class BM25Index:
    """Inverted index over :class:`~src.ragpipe.chunking.Chunk` objects.

    ``build`` is separate from ``__init__`` so an index can be constructed
    cheaply and then reused (the CLI rebuilds it per corpus, the tests build
    several small ones).
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self._k1 = float(self.config.bm25_k1)
        self._b = float(self.config.bm25_b)
        self._min_length = int(self.config.min_term_length)
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._doc_freqs: list[Counter[str]] = []
        self._inverted: dict[str, list[int]] = {}
        self._average_length = 0.0
        self._built = False

    # ------------------------------------------------------------- building
    def build(self, chunks: Sequence[Chunk]) -> "BM25Index":
        """Index ``chunks``; replaces any previous content."""
        if not chunks:
            raise BM25Error("cannot build an index from zero chunks")
        self._chunks = list(chunks)
        self._doc_tokens = [tokenize(chunk.text, self._min_length) for chunk in self._chunks]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        if not any(self._doc_lengths):
            raise BM25Error("every chunk is empty after tokenisation")
        self._doc_freqs = [term_frequencies(tokens) for tokens in self._doc_tokens]
        inverted: dict[str, list[int]] = {}
        for index, freqs in enumerate(self._doc_freqs):
            for term in freqs:
                inverted.setdefault(term, []).append(index)
        self._inverted = inverted
        self._average_length = sum(self._doc_lengths) / len(self._doc_lengths)
        self._built = True
        return self

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], config: RetrievalConfig | None = None) -> "BM25Index":
        """Convenience constructor that builds immediately."""
        return cls(config).build(chunks)

    # -------------------------------------------------------------- reading
    def __len__(self) -> int:
        return len(self._chunks)

    def _require_built(self) -> None:
        if not self._built:
            raise BM25Error("index is empty; call build() first")

    @property
    def chunks(self) -> list[Chunk]:
        """Chunks in insertion order."""
        self._require_built()
        return list(self._chunks)

    @property
    def vocabulary(self) -> list[str]:
        """Sorted vocabulary of the index."""
        self._require_built()
        return sorted(self._inverted)

    def document_frequency(self, term: str) -> int:
        """Number of chunks containing ``term``."""
        self._require_built()
        return len(self._inverted.get(term, ()))

    def inverse_document_frequency(self, term: str) -> float:
        """Smoothed idf for ``term`` (0 when the term is absent from the index)."""
        self._require_built()
        count = len(self._chunks)
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(1.0 + (count - df + 0.5) / (df + 0.5))

    def average_length(self) -> float:
        """Mean chunk length in tokens."""
        self._require_built()
        return self._average_length

    def stats(self) -> BM25Stats:
        """Index statistics."""
        self._require_built()
        return BM25Stats(
            documents=len(self._chunks),
            vocabulary=len(self._inverted),
            average_length=self._average_length,
            k1=self._k1,
            b=self._b,
        )

    # -------------------------------------------------------------- scoring
    def term_score(self, term: str, document_index: int) -> float:
        """Contribution of one term to one document's score (0 if absent)."""
        self._require_built()
        freqs = self._doc_freqs[document_index]
        tf = freqs.get(term, 0)
        if tf == 0:
            return 0.0
        idf = self.inverse_document_frequency(term)
        length = self._doc_lengths[document_index]
        denominator = tf + self._k1 * (1.0 - self._b + self._b * length / self._average_length)
        return idf * (tf * (self._k1 + 1.0)) / denominator

    def score(self, query: str, document_index: int) -> float:
        """BM25 score of one document for a free-text query."""
        self._require_built()
        if not 0 <= document_index < len(self._chunks):
            raise BM25Error(f"document_index {document_index} outside [0, {len(self._chunks)})")
        terms = tokenize(query, self._min_length)
        if not terms:
            return 0.0
        return sum(self.term_score(term, document_index) for term in terms)

    def score_all(self, query: str) -> list[float]:
        """Scores for every chunk, in insertion order."""
        self._require_built()
        return [self.score(query, index) for index in range(len(self._chunks))]

    def search(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Top ``k`` chunks by BM25 score, ties broken by chunk order.

        Chunks scoring zero are dropped: returning arbitrary zero-score hits makes
        downstream fusion and grounding behave as if evidence existed.
        """
        self._require_built()
        top_k = self.config.top_k if k is None else int(k)
        if top_k <= 0:
            raise BM25Error(f"k must be positive, got {k}")
        scores = self.score_all(query)
        ranked = sorted(
            ((score, index) for index, score in enumerate(scores) if score > 0.0),
            key=lambda pair: (-pair[0], pair[1]),
        )[:top_k]
        return [
            RetrievalHit(chunk_id=self._chunks[index].chunk_id, score=score, rank=rank, source="bm25")
            for rank, (score, index) in enumerate(ranked, start=1)
        ]

    def query_terms(self, query: str) -> list[str]:
        """Tokens of ``query`` that exist in the index vocabulary."""
        self._require_built()
        return [term for term in tokenize(query, self._min_length) if term in self._inverted]

    def unknown_terms(self, query: str) -> list[str]:
        """Tokens of ``query`` that the index has never seen (out-of-vocabulary)."""
        self._require_built()
        return [term for term in tokenize(query, self._min_length) if term not in self._inverted]

    def chunk_by_id(self, chunk_id: str) -> Chunk:
        """Look up a chunk by id."""
        self._require_built()
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise BM25Error(f"unknown chunk id {chunk_id!r}")
