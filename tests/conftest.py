"""Shared pytest options and markers for the frames2py test suite."""

from __future__ import annotations

import pytest


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
