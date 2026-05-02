"""The tools the agent may call, and a registry that validates their arguments.

Three tools, each with a narrow job:

* ``retrieve`` -- hybrid search over the chunk corpus.
* ``lookup``   -- does an exact term occur, and where?  Useful when a question
  hinges on one identifier that ranking buried.
* ``summarise``-- extractive summary of given chunks, ranked by query overlap.

A tool returns a :class:`ToolResult` instead of raising on bad input, because the
agent loop must be able to observe a failure and continue; programming errors still
raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .bm25 import tokenize
from .chunking import Chunk, split_sentences
from .fusion import HybridIndex


class ToolError(ValueError):
    """Raised when a tool is unknown or its arguments are structurally invalid."""


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome of one tool call."""

    name: str
    ok: bool
    payload: Mapping[str, Any]
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "message": self.message, "payload": dict(self.payload)}


@dataclass(frozen=True)
class Tool:
    """A callable tool plus the metadata the agent loop logs."""

    name: str
    description: str
    fn: Callable[..., Mapping[str, Any]]

    def call(self, **kwargs: Any) -> ToolResult:
        try:
            payload = self.fn(**kwargs)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(name=self.name, ok=False, payload={}, message=str(exc))
        if not isinstance(payload, Mapping):  # pragma: no cover - defensive
            return ToolResult(name=self.name, ok=False, payload={}, message="tool returned a non-mapping payload")
        return ToolResult(name=self.name, ok=True, payload=payload)


class ToolRegistry:
    """Name -> :class:`Tool` mapping with a call counter and a call log."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._calls: list[ToolResult] = []

    def register(self, tool: Tool) -> "ToolRegistry":
        """Add a tool, rejecting duplicate names."""
        if tool.name in self._tools:
            raise ToolError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return self

    def names(self) -> list[str]:
        """Registered tool names in registration order."""
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        """Look up a tool, raising for unknown names."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool {name!r}; registered: {', '.join(self._tools) or 'none'}") from exc

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        """Call ``name`` with ``kwargs``, recording the result."""
        result = self.get(name).call(**kwargs)
        self._calls.append(result)
        return result

    @property
    def calls(self) -> list[ToolResult]:
        """Every recorded call, in order."""
        return list(self._calls)

    @property
    def successful_calls(self) -> int:
        """Number of calls that returned ``ok=True``."""
        return sum(1 for result in self._calls if result.ok)

    def reset_log(self) -> None:
        """Clear the call log (keeps the tools registered)."""
        self._calls.clear()

    def describe(self) -> dict[str, str]:
        """Tool names mapped to their descriptions, for prompt-like logs."""
        return {name: tool.description for name, tool in self._tools.items()}


def make_retrieve_tool(index: HybridIndex, default_k: int = 4) -> Tool:
    """Build the ``retrieve`` tool over a hybrid index."""

    def retrieve(query: str, k: int | None = None) -> dict[str, Any]:
        if not query or not query.strip():
            raise ToolError("retrieve needs a non-empty query")
        limit = default_k if k is None else int(k)
        if limit <= 0:
            raise ToolError(f"k must be positive, got {k}")
        hits = index.search(query, limit)
        return {
            "query": query,
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 6),
                    "sources": list(hit.sources),
                }
                for hit in hits
            ],
            "chunks": [index.chunk_by_id(hit.chunk_id) for hit in hits],
        }

    return Tool(
        name="retrieve",
        description="Hybrid (BM25 + hashing dense, RRF-fused) retrieval over the chunk corpus.",
        fn=retrieve,
    )


def make_lookup_tool(chunks: Sequence[Chunk]) -> Tool:
    """Build the ``lookup`` tool: exact token presence with quoted context."""

    def lookup(term: str, limit: int = 3) -> dict[str, Any]:
        bare = term.strip().lower()
        if not bare:
            raise ToolError("lookup needs a non-empty term")
        if len(bare.split()) != 1:
            raise ToolError(f"lookup expects a single term, got {term!r}")
        matches: list[dict[str, Any]] = []
        for chunk in chunks:
            if bare in tokenize(chunk.text):
                position = chunk.text.lower().find(bare)
                snippet = chunk.text[max(0, position - 40) : position + 60].replace("\n", " ")
                matches.append({"chunk_id": chunk.chunk_id, "snippet": snippet})
        return {"term": bare, "match_count": len(matches), "matches": matches[:limit]}

    return Tool(
        name="lookup",
        description="Find an exact term in the corpus and return short snippets.",
        fn=lookup,
    )


def make_summarise_tool(chunks: Sequence[Chunk], max_sentences: int = 3) -> Tool:
    """Build the ``summarise`` tool: extractive summary of selected chunks."""

    def summarise(query: str, chunk_ids: Sequence[str]) -> dict[str, Any]:
        if not chunk_ids:
            raise ToolError("summarise needs at least one chunk id")
        wanted = [chunk for chunk in chunks if chunk.chunk_id in set(chunk_ids)]
        if not wanted:
            raise ToolError(f"no known chunk ids in {list(chunk_ids)!r}")
        sentences = select_sentences(query, wanted, max_sentences)
        return {"chunk_ids": [chunk.chunk_id for chunk in wanted], "sentences": sentences}

    return Tool(
        name="summarise",
        description="Extract the sentences from the given chunks that best match the query.",
        fn=summarise,
    )


def select_sentences(query: str, chunks: Sequence[Chunk], limit: int = 3) -> list[str]:
    """Rank sentences from ``chunks`` by query-token overlap and return the top ``limit``.

    Ties keep corpus order so the summary is deterministic.  Sentences shorter than
    four tokens are skipped: they rarely carry an answer and they inflate overlap
    scores.
    """
    if limit <= 0:
        raise ToolError(f"limit must be positive, got {limit}")
    query_terms = set(tokenize(query))
    scored: list[tuple[int, float, str]] = []
    for chunk_index, chunk in enumerate(chunks):
        for sentence_index, sentence in enumerate(split_sentences(chunk.text)):
            terms = set(tokenize(sentence))
            if len(terms) < 3:
                continue
            overlap = len(terms & query_terms) / len(terms)
            scored.append((chunk_index * 1000 + sentence_index, overlap, sentence))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[1], item[0]))
    seen: set[str] = set()
    out: list[str] = []
    for _, _, sentence in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        out.append(sentence)
        if len(out) >= limit:
            break
    return out


def build_default_tools(index: HybridIndex, max_summary_sentences: int = 3) -> ToolRegistry:
    """The tool set the agent uses: retrieve, lookup and summarise."""
    registry = ToolRegistry()
    registry.register(make_retrieve_tool(index))
    registry.register(make_lookup_tool(index.chunks()))
    registry.register(make_summarise_tool(index.chunks(), max_summary_sentences))
    return registry


def tool_summary(registry: ToolRegistry) -> dict[str, Any]:
    """Counts for the CLI's trace printout."""
    return {
        "tools": registry.names(),
        "calls": len(registry.calls),
        "successful": registry.successful_calls,
        "failed": len(registry.calls) - registry.successful_calls,
    }
