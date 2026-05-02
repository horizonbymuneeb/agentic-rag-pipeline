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
