"""Shared fixtures: small hand-written corpora so every expected value is exact."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
