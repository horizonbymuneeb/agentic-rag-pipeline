"""Tools and the registry: registration, validation, failure reporting, summaries."""

from __future__ import annotations

import pytest

from src.ragpipe.chunking import Chunk
from src.ragpipe.fusion import HybridIndex
from src.ragpipe.tools import (
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    build_default_tools,
    make_lookup_tool,
    make_retrieve_tool,
    make_summarise_tool,
    select_sentences,
    tool_summary,
)


def chunk(index: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"c{index}", doc_id="d", index=index, text=text, start=0, end=len(text))


def test_registry_register_lookup_and_duplicates() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="echo", description="echo", fn=lambda **kw: {"args": kw}))
    assert registry.names() == ["echo"]
    assert "echo" in registry
    assert len(registry) == 1
    assert registry.get("echo").name == "echo"
    with pytest.raises(ToolError, match="already registered"):
        registry.register(Tool(name="echo", description="dup", fn=lambda **kw: {}))
    with pytest.raises(ToolError, match="unknown tool"):
        registry.get("missing")
    assert registry.describe() == {"echo": "echo"}


def test_registry_records_success_and_failure() -> None:
    registry = ToolRegistry()

    def sometimes(**kwargs):
        if kwargs.get("fail"):
            raise ValueError("bad argument")
        return {"ok": True}

    registry.register(Tool(name="t", description="t", fn=sometimes))
    ok = registry.call("t", value=1)
    failed = registry.call("t", fail=True)
    assert isinstance(ok, ToolResult) and ok.ok is True
    assert failed.ok is False and "bad argument" in failed.message
    assert len(registry.calls) == 2
    assert registry.successful_calls == 1
    registry.reset_log()
    assert registry.calls == []


def test_tool_call_rejects_non_mapping_payload() -> None:
    tool = Tool(name="bad", description="bad", fn=lambda **kw: ["not", "a", "mapping"])  # type: ignore[return-value]
    result = tool.call()
    assert result.ok is False
    assert "non-mapping" in result.message


def test_retrieve_tool_returns_hits(hybrid_index: HybridIndex) -> None:
    tool = make_retrieve_tool(hybrid_index, default_k=2)
    result = tool.call(query="reciprocal rank fusion")
    assert result.ok is True
    payload = result.payload
    assert payload["query"] == "reciprocal rank fusion"
    assert 1 <= len(payload["hits"]) <= 2
    assert all({"chunk_id", "score", "sources"} == set(hit) for hit in payload["hits"])
    assert all(isinstance(item, Chunk) for item in payload["chunks"])


def test_retrieve_tool_validation(hybrid_index: HybridIndex) -> None:
    tool = make_retrieve_tool(hybrid_index)
    assert tool.call(query="   ").ok is False
    assert tool.call(query="alpha", k=0).ok is False


def test_lookup_tool_exact_term_and_snippets() -> None:
    chunks = [chunk(0, "BM25 uses k1 for saturation"), chunk(1, "unrelated material here")]
    tool = make_lookup_tool(chunks)
    result = tool.call(term="k1")
    assert result.ok is True
    assert result.payload["match_count"] == 1
    match = result.payload["matches"][0]
    assert match["chunk_id"] == "c0"
    assert "k1" in match["snippet"]


def test_lookup_tool_rejects_multi_term_and_empty() -> None:
    tool = make_lookup_tool([chunk(0, "alpha beta")])
    assert tool.call(term="alpha beta").ok is False
    assert tool.call(term="   ").ok is False


def test_summarise_tool_returns_sentences() -> None:
    chunks = [
        chunk(0, "BM25 saturates term frequency with k1. Length normalisation uses b."),
        chunk(1, "Reciprocal rank fusion damps the top ranks."),
    ]
    tool = make_summarise_tool(chunks, max_sentences=2)
    result = tool.call(query="what does k1 control", chunk_ids=["c0"])
    assert result.ok is True
    assert result.payload["chunk_ids"] == ["c0"]
    assert len(result.payload["sentences"]) <= 2
    assert all(isinstance(sentence, str) for sentence in result.payload["sentences"])
