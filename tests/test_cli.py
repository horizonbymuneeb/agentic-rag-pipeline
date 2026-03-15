"""CLI wiring: pipeline assembly, ask/stats/eval payloads and file output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ragpipe.cli import Pipeline, build_pipeline, main, summarize_pipeline
from src.ragpipe.config import PipelineConfig


def test_build_pipeline_from_default_config() -> None:
    pipeline = build_pipeline(PipelineConfig())
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.documents) == 8
    assert len(pipeline.chunks) >= 8
    assert len(pipeline.index) == len(pipeline.chunks)
    assert pipeline.agent.index is pipeline.index


def test_summarize_pipeline_keys() -> None:
    pipeline = build_pipeline(PipelineConfig())
    stats = summarize_pipeline(pipeline)
    assert set(stats) == {"corpus", "chunks", "index", "tools"}
    assert stats["corpus"]["documents"] == 8
    assert stats["chunks"]["chunks"] == len(pipeline.chunks)
    assert stats["index"]["chunks"] == len(pipeline.chunks)
    assert stats["tools"]["tools"] == ["retrieve", "lookup", "summarise"]


def test_pipeline_ask_payload() -> None:
    pipeline = build_pipeline(PipelineConfig())
    payload = pipeline.ask("How does reciprocal rank fusion combine ranked lists?")
    assert payload["question"].startswith("How does")
    assert payload["answer"]
    assert payload["citations"]
    assert payload["step_count"] >= 2
    assert payload["grounding_summary"]["method"].startswith("lexical")
    assert payload["tools"]["calls"] >= 1
