"""Typed configuration for the retrieval pipeline, the rewriting loop and the agent.

Every tunable is a frozen dataclass so a run can be validated up front and logged
afterwards.  ``load_config`` reads ``config.yaml`` and rejects unknown keys, because
a silently ignored typo in a retrieval parameter is very hard to notice from the
answer quality alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ConfigError(ValueError):
    """Raised when configuration values are missing, unknown or out of range."""


def _check(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ConfigError(f"{name}={value!r} is outside [{low}, {high}]")


def _check_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
