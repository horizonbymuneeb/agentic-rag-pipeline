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


@dataclass(frozen=True)
class RewriteConfig:
    """Query-rewriting loop: how many rounds, how many terms per rewrite."""

    enabled: bool = True
    max_rounds: int = 2
    max_new_terms: int = 2
    coverage_threshold: float = 0.5
    min_gap_frequency: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_positive("max_rounds", self.max_rounds)
        _check_positive("max_new_terms", self.max_new_terms)
        _check("coverage_threshold", self.coverage_threshold, 0.0, 1.0)
        if self.min_gap_frequency < 1:
            raise ConfigError("min_gap_frequency must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RewriteConfig":
        return _build(cls, raw)

    def with_overrides(self, **overrides: Any) -> "RewriteConfig":
        """Copy with ``overrides`` applied and re-validated."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class AgentConfig:
    """Step budget, tool limits and answer assembly for the agent loop."""

    max_steps: int = 8
    top_k: int = 4
    max_summary_sentences: int = 3
    per_tool_step_cost: int = 1
    require_grounding: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_positive("max_steps", self.max_steps)
        _check_positive("top_k", self.top_k)
        _check_positive("max_summary_sentences", self.max_summary_sentences)
        _check_positive("per_tool_step_cost", self.per_tool_step_cost)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AgentConfig":
        return _build(cls, raw)

    def with_overrides(self, **overrides: Any) -> "AgentConfig":
        """Copy with ``overrides`` applied and re-validated."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class GroundingConfig:
    """Thresholds for the token-overlap grounding heuristic."""

    support_threshold: float = 0.6
    min_claim_tokens: int = 3
    max_claims: int = 8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check("support_threshold", self.support_threshold, 0.0, 1.0)
        _check_positive("min_claim_tokens", self.min_claim_tokens)
        _check_positive("max_claims", self.max_claims)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GroundingConfig":
        return _build(cls, raw)

    def with_overrides(self, **overrides: Any) -> "GroundingConfig":
        """Copy with ``overrides`` applied and re-validated."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class PipelineConfig:
    """The whole pipeline: chunking, retrieval, rewriting, agent, grounding, eval."""

    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    corpus_seed: int = 0
    corpus_path: str | None = None
    eval_k: int = 5
    run_name: str = "agentic-rag"

    def __post_init__(self) -> None:
        if self.corpus_seed < 0:
            raise ConfigError("corpus_seed must be >= 0")
        _check_positive("eval_k", self.eval_k)
        if not self.run_name:
            raise ConfigError("run_name must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rewrite"]["enabled"] = self.rewrite.enabled
        return data

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PipelineConfig":
        raw = dict(raw)
        chunking = ChunkConfig.from_mapping(raw.pop("chunking", {}) or {})
        retrieval = RetrievalConfig.from_mapping(raw.pop("retrieval", {}) or {})
        rewrite = RewriteConfig.from_mapping(raw.pop("rewrite", {}) or {})
        agent = AgentConfig.from_mapping(raw.pop("agent", {}) or {})
        grounding = GroundingConfig.from_mapping(raw.pop("grounding", {}) or {})
        return _build(
            cls,
            raw,
            chunking=chunking,
            retrieval=retrieval,
            rewrite=rewrite,
            agent=agent,
            grounding=grounding,
        )


def _build(cls: type, raw: Mapping[str, Any], **prebuilt: Any):
    """Instantiate ``cls`` from ``raw``, converting unknown keys into ``ConfigError``."""
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {', '.join(unknown)}")
    try:
        return cls(**raw, **prebuilt)
    except TypeError as exc:
        raise ConfigError(f"invalid {cls.__name__}: {exc}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level of the config must be a mapping")
    return data


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from YAML."""
    return PipelineConfig.from_mapping(load_yaml(path))


def default_config() -> PipelineConfig:
    """All-defaults configuration, used by tests and the CLI's ``--config`` fallback."""
    return PipelineConfig()


def dump_config(config: PipelineConfig) -> dict[str, Any]:
    """Nested plain-dict snapshot for logging and checkpoints."""
    return config.to_dict()


def with_retrieval_overrides(config: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """Copy of ``config`` with retrieval fields overridden and re-validated."""
    return replace(config, retrieval=config.retrieval.with_overrides(**overrides))


def with_chunking_overrides(config: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """Copy of ``config`` with chunking fields overridden and re-validated."""
    return replace(config, chunking=config.chunking.with_overrides(**overrides))
