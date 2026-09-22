"""Tests for native (C++) kernel wrappers with NumPy fallback parity.

When the C++ module is not compiled, these kernels automatically fall
back to their NumPy equivalents.  The tests verify identical output
regardless of which backend is active.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from frames2py.core.types import EVENT_DTYPE, SnapshotMeta
from frames2py.kernels.base import get_kernel


def _events_at(positions: list[tuple[int, int, int, int]]) -> np.ndarray:
    n = len(positions)
    events = np.empty(n, dtype=EVENT_DTYPE)
    for i, (t, x, y, p) in enumerate(positions):
        events[i] = (t, x, y, p)
    return events


def _random_events(n: int, w: int = 64, h: int = 48, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(n, dtype=np.uint64)
    events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


class TestNativeEventCountParity:
    """Compare native_event_count against event_count (NumPy)."""

    def test_identical_output(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_k = get_kernel("event_count")
            nat_k = get_kernel("native_event_count")

        events = _random_events(5000)
        sensor = (64, 48)

        np_state = np_k.init_state(sensor)
        nat_state = nat_k.init_state(sensor)

        np_k.accumulate(events, np_state)
        nat_k.accumulate(events, nat_state)

        np_out = np.zeros((48, 64), dtype=np.float32)
        nat_out = np.zeros((48, 64), dtype=np.float32)

        np_k.snapshot(np_state, np_out)
        nat_k.snapshot(nat_state, nat_out)

        np.testing.assert_allclose(nat_out, np_out, rtol=1e-5)

    def test_reset_parity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_k = get_kernel("event_count")
            nat_k = get_kernel("native_event_count")

        events = _random_events(100)
        sensor = (64, 48)

        np_state = np_k.init_state(sensor)
        nat_state = nat_k.init_state(sensor)

        np_k.accumulate(events, np_state)
        nat_k.accumulate(events, nat_state)

        np_k.reset(np_state)
        nat_k.reset(nat_state)

        assert np_state.buf.sum() == 0.0
        assert nat_state.buf.sum() == 0.0


class TestNativePolarityParity:
    """Compare native_polarity against polarity (NumPy)."""

    def test_identical_output(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_k = get_kernel("polarity")
            nat_k = get_kernel("native_polarity")

        events = _random_events(5000)
        sensor = (64, 48)

        np_state = np_k.init_state(sensor)
        nat_state = nat_k.init_state(sensor)

        np_k.accumulate(events, np_state)
        nat_k.accumulate(events, nat_state)

        np_out = np.zeros((48, 64, 2), dtype=np.float32)
        nat_out = np.zeros((48, 64, 2), dtype=np.float32)

        np_k.snapshot(np_state, np_out)
        nat_k.snapshot(nat_state, nat_out)

        np.testing.assert_allclose(nat_out, np_out, rtol=1e-5)


class TestNativeTimeSurfaceParity:
    """Compare native_time_surface against time_surface (NumPy)."""

    def test_identical_output(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_k = get_kernel("time_surface")
            nat_k = get_kernel("native_time_surface")

        events = _random_events(5000)
        sensor = (64, 48)

        np_state = np_k.init_state(sensor)
        nat_state = nat_k.init_state(sensor)

        np_k.accumulate(events, np_state)
        nat_k.accumulate(events, nat_state)

        np_out = np.zeros((48, 64), dtype=np.float64)
        nat_out = np.zeros((48, 64), dtype=np.float64)

        np_k.snapshot(np_state, np_out)
        nat_k.snapshot(nat_state, nat_out)

        np.testing.assert_allclose(nat_out, np_out, rtol=1e-10)

    def test_non_monotonic_timestamps(self):
        """Time surface handles out-of-order timestamps correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_k = get_kernel("time_surface")
            nat_k = get_kernel("native_time_surface")

        events = _events_at([
            (500, 3, 5, 0),
            (100, 3, 5, 1),  # earlier timestamp at same pixel
            (300, 3, 5, 0),
        ])
        sensor = (10, 8)

        np_state = np_k.init_state(sensor)
        nat_state = nat_k.init_state(sensor)

        np_k.accumulate(events, np_state)
        nat_k.accumulate(events, nat_state)

        np_out = np.zeros((8, 10), dtype=np.float64)
        nat_out = np.zeros((8, 10), dtype=np.float64)

        np_k.snapshot(np_state, np_out)
        nat_k.snapshot(nat_state, nat_out)

        # Both should keep the max timestamp (500)
        assert np_out[5, 3] == 500.0
        assert nat_out[5, 3] == 500.0


class TestAllNativeKernelsRegister:
    """Verify all native kernels are accessible via get_kernel."""

    @pytest.mark.parametrize("name", [
        "native_event_count",
        "native_polarity",
        "native_time_surface",
    ])
    def test_registered(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            k = get_kernel(name)
        assert k.name == name
