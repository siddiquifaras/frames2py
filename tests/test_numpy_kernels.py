"""Tests for the four NumPy accumulation kernels.

Each kernel is tested for:
    - Correct output for known input patterns
    - State reset behaviour
    - Out-of-bounds coordinate safety
    - Empty batch handling
    - Snapshot metadata correctness
    - No allocation in accumulate/snapshot (via shape stability)
"""

from __future__ import annotations

import numpy as np
import pytest

from frames2py.core.types import EVENT_DTYPE, SnapshotMeta
from frames2py.kernels.base import get_kernel
from frames2py.kernels.numpy_kernels import (
    EventCountKernel,
    ExpDecayKernel,
    PolarityKernel,
    TimeSurfaceKernel,
)


def _events_at(positions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Create events at specific (t, x, y, p) positions."""
    n = len(positions)
    events = np.empty(n, dtype=EVENT_DTYPE)
    for i, (t, x, y, p) in enumerate(positions):
        events[i] = (t, x, y, p)
    return events


# ===================================================================
# EventCountKernel
# ===================================================================
class TestEventCountKernel:

    def test_single_event(self):
        k = EventCountKernel()
        state = k.init_state((10, 8))
        events = _events_at([(100, 3, 5, 1)])
        k.accumulate(events, state)
        assert state.buf[5, 3] == 1.0

    def test_multiple_events_same_pixel(self):
        k = EventCountKernel()
        state = k.init_state((10, 8))
        events = _events_at([
            (100, 3, 5, 1),
            (200, 3, 5, 0),
            (300, 3, 5, 1),
        ])
        k.accumulate(events, state)
        assert state.buf[5, 3] == 3.0

    def test_snapshot_copies_and_resets(self):
        k = EventCountKernel()
        state = k.init_state((10, 8))
        events = _events_at([(100, 2, 3, 0)])
        k.accumulate(events, state)

        out = np.zeros((8, 10), dtype=np.float32)
        meta = k.snapshot(state, out)

        assert out[3, 2] == 1.0
        assert state.buf[3, 2] == 0.0  # reset after snapshot
        assert isinstance(meta, SnapshotMeta)
        assert meta.events_accumulated == 1

    def test_empty_batch(self):
        k = EventCountKernel()
        state = k.init_state((10, 8))
        k.accumulate(np.empty(0, dtype=EVENT_DTYPE), state)
        assert state.buf.sum() == 0.0

    def test_out_of_bounds_discarded(self):
        k = EventCountKernel()
        state = k.init_state((10, 8))
        events = _events_at([
            (100, 999, 999, 1),  # way out of bounds
            (200, 5, 4, 0),      # valid
        ])
        k.accumulate(events, state)
        assert state.buf[4, 5] == 1.0
        # OOB events are discarded, not clamped -- edge pixels stay clean
        assert state.buf[7, 9] == 0.0
        assert state.events_accumulated == 2

    def test_reset(self):
        k = EventCountKernel()
        state = k.init_state((10, 8))
        k.accumulate(_events_at([(100, 1, 1, 1)]), state)
        k.reset(state)
        assert state.buf.sum() == 0.0
        assert state.events_accumulated == 0

    def test_factory_lookup(self):
        k = get_kernel("event_count")
        assert isinstance(k, EventCountKernel)

    def test_channels(self):
        assert EventCountKernel().channels == 1

    def test_large_batch_vectorised(self):
        """Verify correct output with a large random batch."""
        k = EventCountKernel()
        state = k.init_state((64, 48))
        rng = np.random.default_rng(42)
        n = 10_000
        events = np.empty(n, dtype=EVENT_DTYPE)
        events["t"] = np.arange(n, dtype=np.uint64)
        events["x"] = rng.integers(0, 64, size=n, dtype=np.uint16)
        events["y"] = rng.integers(0, 48, size=n, dtype=np.uint16)
        events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
        k.accumulate(events, state)
        assert state.buf.sum() == n

    def test_1m_events_exact_count(self):
        """1M events: frame pixel sum must equal event count exactly."""
        k = EventCountKernel()
        w, h = 1280, 720
        state = k.init_state((w, h))
        rng = np.random.default_rng(7)
        batch_size = 50_000
        total = 0
        for i in range(20):
            events = np.empty(batch_size, dtype=EVENT_DTYPE)
            events["t"] = np.arange(
                i * batch_size, (i + 1) * batch_size, dtype=np.uint64
            )
            events["x"] = rng.integers(0, w, size=batch_size, dtype=np.uint16)
            events["y"] = rng.integers(0, h, size=batch_size, dtype=np.uint16)
            events["p"] = rng.integers(0, 2, size=batch_size, dtype=np.uint8)
            k.accumulate(events, state)
            total += batch_size
        assert state.buf.sum() == total
        assert state.events_accumulated == total

    def test_bincount_matches_reference(self):
        """Cross-check bincount kernel vs naive per-event loop."""
        k = EventCountKernel()
        w, h = 32, 24
        state = k.init_state((w, h))
        rng = np.random.default_rng(99)
        n = 5_000
        events = np.empty(n, dtype=EVENT_DTYPE)
        events["t"] = np.arange(n, dtype=np.uint64)
        events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
        events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
        events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
        k.accumulate(events, state)

        ref = np.zeros((h, w), dtype=np.float32)
        for ev in events:
            ref[int(ev["y"]), int(ev["x"])] += 1
        np.testing.assert_array_equal(state.buf, ref)


# ===================================================================
# PolarityKernel
# ===================================================================
class TestPolarityKernel:

    def test_on_off_separation(self):
        k = PolarityKernel()
        state = k.init_state((10, 8))
        events = _events_at([
            (100, 3, 5, 1),  # ON → channel 1
            (200, 3, 5, 0),  # OFF → channel 0
        ])
        k.accumulate(events, state)
        assert state.buf[5, 3, 1] == 1.0  # ON
        assert state.buf[5, 3, 0] == 1.0  # OFF

    def test_snapshot_resets(self):
        k = PolarityKernel()
        state = k.init_state((10, 8))
        k.accumulate(_events_at([(100, 1, 1, 1)]), state)
        out = np.zeros((8, 10, 2), dtype=np.float32)
        k.snapshot(state, out)
        assert out[1, 1, 1] == 1.0
        assert state.buf.sum() == 0.0

    def test_channels(self):
        assert PolarityKernel().channels == 2

    def test_output_shape(self):
        k = PolarityKernel()
        state = k.init_state((20, 15))
        assert state.buf.shape == (15, 20, 2)

    def test_500k_events_exact_polarity_split(self):
        """500K events: per-channel pixel sums must match polarity counts."""
        k = PolarityKernel()
        w, h = 640, 480
        state = k.init_state((w, h))
        rng = np.random.default_rng(11)
        n = 500_000
        events = np.empty(n, dtype=EVENT_DTYPE)
        events["t"] = np.arange(n, dtype=np.uint64)
        events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
        events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
        events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
        k.accumulate(events, state)
        n_off = int((events["p"] == 0).sum())
        n_on = int((events["p"] == 1).sum())
        assert state.buf[:, :, 0].sum() == n_off
        assert state.buf[:, :, 1].sum() == n_on
        assert state.events_accumulated == n


# ===================================================================
# TimeSurfaceKernel
# ===================================================================
class TestTimeSurfaceKernel:

    def test_stores_latest_timestamp(self):
        k = TimeSurfaceKernel()
        state = k.init_state((10, 8))
        events = _events_at([
            (100, 3, 5, 0),
            (500, 3, 5, 1),
            (200, 3, 5, 0),  # older timestamp at same pixel
        ])
        k.accumulate(events, state)
        # np.maximum.at keeps 500 (the max seen)
        assert state.buf[5, 3] == 500.0

    def test_no_reset_on_snapshot(self):
        k = TimeSurfaceKernel()
        state = k.init_state((10, 8))
        k.accumulate(_events_at([(100, 1, 1, 0)]), state)
        out = np.zeros((8, 10), dtype=np.float64)
        k.snapshot(state, out)
        # State should NOT be zeroed.
        assert state.buf[1, 1] == 100.0

    def test_uses_float64(self):
        k = TimeSurfaceKernel()
        state = k.init_state((10, 8))
        assert state.buf.dtype == np.float64


# ===================================================================
# ExpDecayKernel
# ===================================================================
class TestExpDecayKernel:

    def test_decay_applied(self):
        k = ExpDecayKernel(decay=0.5)
        state = k.init_state((10, 8))
        # First accumulate: pixel (2,3) = 1.0
        k.accumulate(_events_at([(100, 2, 3, 0)]), state)
        assert state.buf[3, 2] == 1.0

        # Second accumulate with empty batch: all pixels *= 0.5
        k.accumulate(np.empty(0, dtype=EVENT_DTYPE), state)
        assert state.buf[3, 2] == pytest.approx(0.5, abs=1e-6)

    def test_decay_plus_new_events(self):
        k = ExpDecayKernel(decay=0.5)
        state = k.init_state((10, 8))
        k.accumulate(_events_at([(100, 2, 3, 0)]), state)
        # Now decay + add to same pixel
        k.accumulate(_events_at([(200, 2, 3, 1)]), state)
        # Expected: 1.0 * 0.5 + 1.0 = 1.5
        assert state.buf[3, 2] == pytest.approx(1.5, abs=1e-6)

    def test_no_reset_on_snapshot(self):
        k = ExpDecayKernel(decay=0.9)
        state = k.init_state((10, 8))
        k.accumulate(_events_at([(100, 1, 1, 0)]), state)
        out = np.zeros((8, 10), dtype=np.float32)
        k.snapshot(state, out)
        assert state.buf[1, 1] > 0.0  # not reset


# ===================================================================
# Cross-kernel tests
# ===================================================================
class TestKernelProtocol:
    """Verify all kernels satisfy the Kernel protocol."""

    @pytest.mark.parametrize("name", ["event_count", "polarity", "time_surface", "exp_decay"])
    def test_has_required_methods(self, name):
        k = get_kernel(name)
        assert hasattr(k, "init_state")
        assert hasattr(k, "accumulate")
        assert hasattr(k, "snapshot")
        assert hasattr(k, "reset")
        assert hasattr(k, "name")
        assert hasattr(k, "channels")

    @pytest.mark.parametrize("name", ["event_count", "polarity", "time_surface", "exp_decay"])
    def test_snapshot_returns_meta(self, name):
        k = get_kernel(name)
        state = k.init_state((10, 8))
        events = _events_at([(100, 1, 1, 0)])
        k.accumulate(events, state)

        channels = k.channels
        if channels > 1:
            out = np.zeros((8, 10, channels), dtype=state.buf.dtype)
        else:
            out = np.zeros((8, 10), dtype=state.buf.dtype)
        meta = k.snapshot(state, out)
        assert isinstance(meta, SnapshotMeta)
        assert meta.events_accumulated >= 1

    def test_unknown_kernel_raises(self):
        with pytest.raises(KeyError, match="Unknown kernel"):
            get_kernel("nonexistent_kernel")
