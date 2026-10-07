"""PyTorch in tests: only the PyTorch recipe's tests and snippet use it, and only the separate torch CI job installs it.

``require_torch()`` returns the module, or skips when PyTorch isn't installed. PyTorch is not an extra, so
``FRAMES2PY_REQUIRE_EXTRAS`` doesn't cover it: set ``FRAMES2PY_REQUIRE_TORCH=1`` to fail instead of skipping.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

REQUIRE_ENV = "FRAMES2PY_REQUIRE_TORCH"


def require_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        reason = f"torch is not installed: {exc}"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{reason} ({REQUIRE_ENV}=1)", pytrace=False)
        pytest.skip(reason, allow_module_level=True)
