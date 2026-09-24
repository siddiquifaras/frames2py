"""The implementation under test.

The v1 behavioural suite reaches the implementation only through ``impl``: the
``frames2py`` package by default, or the module named by
``FRAMES2PY_CONTRACT_IMPL``. Missing names fail the test that uses them instead
of breaking collection, so the suite runs against an incomplete implementation
and reports what is missing.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

import numpy as np

impl: ModuleType = importlib.import_module(os.environ.get("FRAMES2PY_CONTRACT_IMPL", "frames2py"))

EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])
"""The canonical layout, written out independently of the implementation."""


def accumulated(stats: Any) -> int:
    """Events accumulated so far: ``events_ingested - events_out_of_bounds``."""
    return int(stats.events_ingested) - int(stats.events_out_of_bounds)
