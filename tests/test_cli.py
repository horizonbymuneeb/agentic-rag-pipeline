"""CLI wiring: pipeline assembly, ask/stats/eval payloads and file output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ragpipe.cli import Pipeline, build_pipeline, main, summarize_pipeline
from src.ragpipe.config import PipelineConfig
