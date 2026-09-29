"""Shared pytest options and markers for the frames2py test suite."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from hypothesis import settings

import frames2py

# Property tests: the same examples on every run, so CI results reproduce and a red run is a
# regression, not a lucky draw. No example database is written into the checkout.
# `pytest --hypothesis-profile explore` searches many more random examples.
settings.register_profile("default", derandomize=True, database=None, deadline=None, print_blob=True)
settings.register_profile("explore", max_examples=3000, database=None, deadline=None, print_blob=True)
settings.load_profile("default")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--recordings",
        action="store_true",
        help="run the opt-in tests on real recordings (fetch them with: uv run python -m tests.recordings download)",
    )
    parser.addoption(
        "--display",
        action="store_true",
        help="run the opt-in tests that open a window (needs a display and the viewer extra)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "recordings: needs downloaded real recordings; runs only with --recordings")
    config.addinivalue_line("markers", "display: opens a window; runs only with --display")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, option, reason in (
        ("recordings", "--recordings", "opt-in real-recording test: pass --recordings"),
        ("display", "--display", "opt-in window test: pass --display"),
    ):
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


def _gil_enabled() -> bool:
    return bool(getattr(sys, "_is_gil_enabled", lambda: True)())


@pytest.fixture(scope="session", autouse=True)
def _runtime_properties(record_testsuite_property: Callable[[str, object], None]) -> Iterator[None]:
    """Record the runtime that ran the suite in the JUnit XML, so CI can check it."""
    record_testsuite_property("python", sys.version.split()[0])
    record_testsuite_property("numpy", np.__version__)
    record_testsuite_property("frames2py_file", frames2py.__file__)
    record_testsuite_property("gil_enabled_at_start", _gil_enabled())
    yield
    record_testsuite_property("gil_enabled_at_end", _gil_enabled())
