"""Typed configuration for the retrieval pipeline, the rewriting loop and the agent.

Every tunable is a frozen dataclass so a run can be validated up front and logged
afterwards.  ``load_config`` reads ``config.yaml`` and rejects unknown keys, because
a silently ignored typo in a retrieval parameter is very hard to notice from the
answer quality alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ConfigError(ValueError):
    """Raised when configuration values are missing, unknown or out of range."""


def _check(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ConfigError(f"{name}={value!r} is outside [{low}, {high}]")


def _check_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ChunkConfig:
    """How documents are split.

    ``mode`` is either ``fixed`` (character windows) or ``sentence`` (whole
    sentences up to ``max_chars``).  ``overlap`` is measured in characters for
    fixed mode and in sentences for sentence mode.
    """

    mode: str = "sentence"
    size: int = 320
    overlap: int = 60
    max_chars: int = 400
    overlap_sentences: int = 1
    min_chars: int = 40

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in {"fixed", "sentence"}:
            raise ConfigError(f"chunk mode must be 'fixed' or 'sentence', got {self.mode!r}")
        _check_positive("size", self.size)
        _check_positive("max_chars", self.max_chars)
        _check_positive("min_chars", self.min_chars)
        if self.overlap < 0 or self.overlap >= self.size:
            raise ConfigError(f"overlap must be in [0, size), got {self.overlap}")
        if self.overlap_sentences < 0 or self.overlap_sentences > 5:
            raise ConfigError(f"overlap_sentences must be in [0, 5], got {self.overlap_sentences}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ChunkConfig":
        return _build(cls, raw)

    def with_overrides(self, **overrides: Any) -> "ChunkConfig":
        """Copy with ``overrides`` applied and re-validated."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class RetrievalConfig:
    """BM25 constants, embedding width and fusion parameters."""

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k: int = 5
    candidate_k: int = 10
    embedding_dim: int = 256
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    min_term_length: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.bm25_k1 <= 0:
            raise ConfigError("bm25_k1 must be > 0")
        _check("bm25_b", self.bm25_b, 0.0, 1.0)
        _check_positive("top_k", self.top_k)
        _check_positive("candidate_k", self.candidate_k)
        _check_positive("embedding_dim", self.embedding_dim)
        _check_positive("rrf_k", self.rrf_k)
        _check_positive("min_term_length", self.min_term_length)
        if self.candidate_k < self.top_k:
            raise ConfigError("candidate_k must be >= top_k")
        if self.bm25_weight < 0 or self.dense_weight < 0:
            raise ConfigError("fusion weights must be >= 0")
        if self.bm25_weight == 0 and self.dense_weight == 0:
            raise ConfigError("at least one fusion weight must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RetrievalConfig":
        return _build(cls, raw)

    def with_overrides(self, **overrides: Any) -> "RetrievalConfig":
        """Copy with ``overrides`` applied and re-validated."""
        return replace(self, **overrides)
