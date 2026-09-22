"""Tests for the Telemetry consumer."""

from __future__ import annotations

import time

import numpy as np
import pytest

from frames2py.consumers.telemetry import Telemetry, TelemetrySample
from frames2py.core.engine import Engine
from frames2py.core.types import EVENT_DTYPE


def _make_events(n: int, w: int = 64, h: int = 48, seed: int = 42):
    rng = np.random.default_rng(seed)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(n, dtype=np.uint64)
    events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


class TestTelemetryCollection:
    """Verify telemetry samples are collected correctly."""

    @pytest.mark.timeout(10)
    def test_collects_samples(self):
        engine = Engine(sensor_size=(64, 48))
        tel = Telemetry(engine, poll_interval_ms=50).start()

        for i in range(20):
            engine.ingest(_make_events(100, seed=i))
            time.sleep(0.02)

        time.sleep(0.2)
        tel.stop()

        history = tel.history
        assert len(history) > 0
        for sample in history:
            assert isinstance(sample, TelemetrySample)
            assert sample.wall_time_ns > 0

    @pytest.mark.timeout(10)
    def test_latest_sample(self):
        engine = Engine(sensor_size=(64, 48))
        tel = Telemetry(engine, poll_interval_ms=50).start()

        engine.ingest(_make_events(500))
        time.sleep(0.2)
        tel.stop()

        latest = tel.latest
        assert latest is not None
        assert latest.events_ingested == 500
        assert latest.snapshots_published > 0

    @pytest.mark.timeout(10)
    def test_latency_percentiles_populated(self):
        engine = Engine(sensor_size=(64, 48))
        tel = Telemetry(engine, poll_interval_ms=50).start()

        for i in range(30):
            engine.ingest(_make_events(200, seed=i))
            time.sleep(0.01)

        time.sleep(0.2)
        tel.stop()

        latest = tel.latest
        assert latest is not None
        assert latest.accumulate_ms_p50 >= 0
        assert latest.accumulate_ms_p99 >= latest.accumulate_ms_p50

    @pytest.mark.timeout(10)
    def test_start_stop_idempotent(self):
        engine = Engine(sensor_size=(64, 48))
        tel = Telemetry(engine, poll_interval_ms=100)
        tel.start()
        tel.start()  # second start is no-op
        time.sleep(0.15)
        tel.stop()
        tel.stop()  # second stop is no-op

    def test_history_is_thread_safe_copy(self):
        engine = Engine(sensor_size=(64, 48))
        tel = Telemetry(engine, poll_interval_ms=50).start()
        engine.ingest(_make_events(100))
        time.sleep(0.15)
        h1 = tel.history
        h2 = tel.history
        tel.stop()
        # Returns copies, not the internal list.
        assert h1 is not h2


class TestTelemetrySample:
    """Verify TelemetrySample dataclass."""

    def test_default_values(self):
        s = TelemetrySample()
        assert s.events_ingested == 0
        assert s.viewer_frames_shown is None
        assert s.accumulate_ms_p50 == 0.0

    def test_is_frozen(self):
        s = TelemetrySample(events_ingested=100)
        with pytest.raises(AttributeError):
            s.events_ingested = 200  # type: ignore
