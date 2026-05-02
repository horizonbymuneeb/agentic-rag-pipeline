"""Dense retrieval: deterministic hashing, IDF weighting, cosine search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ragpipe.dense import (
    DenseError,
    DenseIndex,
    HashingTfIdfEmbedder,
    cosine_similarity,
    embed_texts,
    l2_normalise,
    nearest_neighbours,
    token_bucket,
    token_sign,
)
from src.ragpipe.config import RetrievalConfig
