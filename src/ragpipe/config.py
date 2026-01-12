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
