"""Command line entry points: build the pipeline, answer a question, run the eval.

Kept thin on purpose -- it wires configuration to the library modules and prints
JSON, so the same code paths are covered by the tests without shelling out.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .agent import RAGAgent
from .chunking import Chunker, Chunk, chunk_statistics
from .config import PipelineConfig, load_config
from .corpus import Document, build_corpus, corpus_statistics, load_corpus
from .evaluation import EvalResult, build_eval_set, evaluate_pipeline
from .fusion import HybridIndex
from .grounding import grounding_summary
from .tools import build_default_tools, tool_summary


@dataclass
class Pipeline:
    """The assembled objects for one run, so callers do not rebuild them per query."""

    config: PipelineConfig
    documents: list[Document]
    chunks: list[Chunk]
    index: HybridIndex
    agent: RAGAgent

    def ask(self, question: str) -> dict[str, Any]:
        """Answer one question and return the compact trace plus grounding numbers."""
        trace = self.agent.run(question)
        payload = trace.as_dict()
        if trace.grounding is not None:
            payload["grounding_summary"] = grounding_summary(trace.grounding)
        payload["tools"] = tool_summary(self.agent.tools)
        return payload

    def evaluate(self) -> EvalResult:
        """Run the offline evaluation set."""
        return evaluate_pipeline(
            self.index, self.agent, build_eval_set(self.chunks), k=self.config.eval_k
        )


def build_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """Assemble corpus -> chunks -> index -> agent from a configuration."""
    cfg = config or load_config()
    if cfg.corpus_path:
        documents = load_corpus(cfg.corpus_path)
    else:
        documents = build_corpus(cfg.corpus_seed)
    chunks = Chunker(cfg.chunking).split_documents(documents)
    index = HybridIndex(cfg.retrieval).build(chunks)
    agent = RAGAgent(index, cfg, tools=build_default_tools(index, cfg.agent.max_summary_sentences))
    return Pipeline(config=cfg, documents=documents, chunks=chunks, index=index, agent=agent)
