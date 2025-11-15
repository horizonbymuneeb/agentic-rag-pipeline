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
