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
