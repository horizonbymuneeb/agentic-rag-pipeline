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
