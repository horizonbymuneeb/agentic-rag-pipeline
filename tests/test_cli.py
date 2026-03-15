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


def test_cli_stats_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "stats.json"
    code = main(["--config", str(tmp_path / "missing.yaml"), "--stats", "--out", str(out)])
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["stats"]["corpus"]["documents"] == 8


def test_cli_query_writes_answers(tmp_path: Path) -> None:
    out = tmp_path / "answers.json"
    code = main(
        [
            "--query",
            "What does the k1 parameter control in BM25?",
            "--query",
            "What is recall at k measuring?",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["answers"]) == 2
    assert all(answer["answer"] for answer in payload["answers"])


def test_cli_eval_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    code = main(["--eval", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]
    assert evaluation["questions"] == 16
    assert 0.0 <= evaluation["recall_at_k"] <= 1.0
    assert 0.0 <= evaluation["answer_coverage"] <= 1.0
