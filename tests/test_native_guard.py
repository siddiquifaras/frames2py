"""The parity guard refuses anything that isn't a compiled extension."""

from __future__ import annotations

import pytest

from tests.native_guard import REQUIRE_ENV, require_native


def test_missing_module_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_native("_frames2py_no_such_module")


def test_missing_module_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        require_native("_frames2py_no_such_module")


def test_pure_python_module_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Python fallback importable under the native name must not pass for native.
    monkeypatch.setenv(REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        require_native("json")


def test_compiled_extension_is_returned() -> None:
    module = require_native("numpy._core._multiarray_umath")
    assert module.__name__ == "numpy._core._multiarray_umath"
