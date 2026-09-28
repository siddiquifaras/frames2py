"""Optional adapter backends in tests.

A test that needs an adapter's backend calls ``require_backend()``: it returns the module,
or skips the test when the extra isn't installed. Set ``FRAMES2PY_REQUIRE_EXTRAS=1`` to fail
instead, so an environment that should have every extra can't pass without one.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

REQUIRE_ENV = "FRAMES2PY_REQUIRE_EXTRAS"


def require_backend(*modules: str) -> list[ModuleType]:
    loaded = []
    for name in modules:
        try:
            loaded.append(importlib.import_module(name))
        except ImportError as exc:
            reason = f"{name} is not installed: {exc}"
            if os.environ.get(REQUIRE_ENV) == "1":
                pytest.fail(f"{reason} ({REQUIRE_ENV}=1)")
            pytest.skip(reason)
    return loaded
