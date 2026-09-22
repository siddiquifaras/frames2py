"""Shared pytest fixtures for the frames2py test suite."""

from __future__ import annotations

import numpy as np
import pytest

from frames2py.core.types import EVENT_DTYPE


@pytest.fixture
def sensor_size() -> tuple[int, int]:
    """Standard test sensor: 64x48 (small for fast tests)."""
    return (64, 48)


@pytest.fixture
def small_events() -> np.ndarray:
    """100 deterministic events within a 64x48 sensor."""
    rng = np.random.default_rng(42)
    n = 100
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(0, n * 10, 10, dtype=np.uint64)
    events["x"] = rng.integers(0, 64, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, 48, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


@pytest.fixture
def large_events() -> np.ndarray:
    """50,000 deterministic events for stress tests."""
    rng = np.random.default_rng(123)
    n = 50_000
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(0, n, dtype=np.uint64)
    events["x"] = rng.integers(0, 640, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, 480, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


@pytest.fixture
def make_events():
    """Factory fixture: create n events for a given sensor size."""

    def _make(
        n: int,
        sensor_size: tuple[int, int] = (64, 48),
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        w, h = sensor_size
        events = np.empty(n, dtype=EVENT_DTYPE)
        events["t"] = np.arange(0, n, dtype=np.uint64)
        events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
        events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
        events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
        return events

    return _make
