"""Documents and the offline corpus the pipeline runs on.

Two sources: a deterministic synthetic corpus that ships with the repository (so
tests and demos need no downloads) and plain-text files read from a local
directory.  Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
