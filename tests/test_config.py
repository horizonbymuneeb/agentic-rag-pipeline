"""Configuration loading, validation, overrides and sweeps."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ragpipe.config import (
    AgentConfig,
    ChunkConfig,
    ConfigError,
    GroundingConfig,
    PipelineConfig,
    RetrievalConfig,
    RewriteConfig,
    default_config,
    dump_config,
    load_config,
    load_yaml,
    sweep_retrieval,
    with_chunking_overrides,
    with_retrieval_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_config_matches_defaults() -> None:
    config = load_config(REPO_ROOT / "config.yaml")
    assert config.chunking.mode == "sentence"
    assert config.chunking.max_chars == 400
    assert config.retrieval.bm25_k1 == pytest.approx(1.5)
    assert config.retrieval.rrf_k == 60
    assert config.agent.max_steps == 8
    assert config.grounding.support_threshold == pytest.approx(0.6)
    assert config.eval_k == 5


def test_defaults_are_consistent() -> None:
    config = default_config()
    assert config.retrieval.candidate_k >= config.retrieval.top_k
    assert config.chunking.overlap < config.chunking.size
    assert config.rewrite.max_rounds >= 1


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval:\n  top_k: 3\n  topk: 4\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown RetrievalConfig keys: topk"):
        load_config(path)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("eval_k: 3\nnope: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="nope"):
        load_config(path)


def test_partial_yaml_keeps_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("agent:\n  max_steps: 3\n", encoding="utf-8")
    config = load_config(path)
    assert config.agent.max_steps == 3
    assert config.retrieval.top_k == RetrievalConfig().top_k


def test_chunk_config_validation() -> None:
    with pytest.raises(ConfigError):
        ChunkConfig(mode="paragraph")
    with pytest.raises(ConfigError):
        ChunkConfig(size=100, overlap=100)
    with pytest.raises(ConfigError):
        ChunkConfig(overlap=-1)
    with pytest.raises(ConfigError):
        ChunkConfig(overlap_sentences=9)
