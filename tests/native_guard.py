"""Guard for native-vs-NumPy parity tests.

A parity test that runs without the compiled module compares NumPy with NumPy
and proves nothing. Every parity test calls ``require_native()`` first: it
returns the compiled module, or skips the test. Set ``FRAMES2PY_REQUIRE_NATIVE=1``
to fail instead of skip, so a build that should have the module can't pass
without it.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
from types import ModuleType

import pytest

REQUIRE_ENV = "FRAMES2PY_REQUIRE_NATIVE"


def _unavailable(reason: str) -> None:
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{reason} ({REQUIRE_ENV}=1)")
    pytest.skip(reason)


def require_native(module_name: str) -> ModuleType:
    """Import *module_name* and check it is a compiled extension module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        _unavailable(f"native module {module_name} is not importable: {exc}")
        raise AssertionError("unreachable") from exc
    origin = getattr(module, "__file__", None) or ""
    if not origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        _unavailable(f"{module_name} is not a compiled extension (loaded from {origin or 'nowhere'})")
    return module
