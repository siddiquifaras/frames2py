"""Tests for the Viewer consumer with headless backend."""

from __future__ import annotations

import time

import numpy as np
import pytest

from frames2py.core.engine import Engine
from frames2py.core.types import EVENT_DTYPE
from frames2py.consumers.viewer import Viewer, _normalize_frame


def _make_events(n: int, w: int = 64, h: int = 48, seed: int = 42):
    rng = np.random.default_rng(seed)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(n, dtype=np.uint64)
    events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


class TestNormalizeFrame:
    """Test frame normalization for display."""

    def test_scalar_frame_to_bgr(self):
        frame = np.random.rand(48, 64).astype(np.float32)
        result = _normalize_frame(frame, colormap=None)
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    def test_polarity_frame_to_bgr(self):
        frame = np.random.rand(48, 64, 2).astype(np.float32)
        result = _normalize_frame(frame, colormap=None)
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    def test_zero_frame(self):
        frame = np.zeros((48, 64), dtype=np.float32)
        result = _normalize_frame(frame, colormap=None)
        assert result.shape == (48, 64, 3)
        np.testing.assert_array_equal(result, 0)

    def test_constant_frame(self):
        frame = np.full((48, 64), 42.0, dtype=np.float32)
        result = _normalize_frame(frame, colormap=None)
        assert result.shape == (48, 64, 3)


class TestViewerHeadless:
    """Test Viewer with headless backend (no GUI)."""

    @pytest.mark.timeout(10)
    def test_start_stop(self):
        engine = Engine(sensor_size=(64, 48))
        viewer = Viewer(engine, backend="headless", fps=60)
        viewer.start()
        time.sleep(0.1)
        viewer.stop()

    @pytest.mark.timeout(10)
    def test_shows_frames(self):
        engine = Engine(sensor_size=(64, 48))
        viewer = Viewer(engine, backend="headless", fps=100)
        viewer.start()

        for i in range(20):
            engine.ingest(_make_events(100, seed=i))
            time.sleep(0.01)

        time.sleep(0.2)
        viewer.stop()
        assert viewer.frames_shown > 0

    @pytest.mark.timeout(10)
    def test_respects_engine_stop(self):
        engine = Engine(sensor_size=(64, 48))
        viewer = Viewer(engine, backend="headless", fps=60)
        viewer.start()
        time.sleep(0.1)
        engine.stop()
        time.sleep(0.2)
        viewer.stop()

    def test_invalid_backend_raises(self):
        engine = Engine(sensor_size=(64, 48))
        with pytest.raises(ValueError, match="Unknown viewer backend"):
            Viewer(engine, backend="nonexistent")

    @pytest.mark.timeout(10)
    def test_overlay_called(self):
        engine = Engine(sensor_size=(64, 48))
        viewer = Viewer(engine, backend="headless", fps=100)

        draw_count = {"n": 0}

        class CountOverlay:
            def draw(self, frame, meta):
                draw_count["n"] += 1

        viewer.add_overlay(CountOverlay())
        viewer.start()

        for i in range(10):
            engine.ingest(_make_events(100, seed=i))
            time.sleep(0.02)

        time.sleep(0.3)
        viewer.stop()
        assert draw_count["n"] > 0
