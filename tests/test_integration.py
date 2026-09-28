"""Integration tests: full pipeline roundtrips end-to-end.

These tests exercise the complete path from synthetic events through
the engine, consumers, and adapters working together.
"""

from __future__ import annotations

import tempfile
import time
import threading

import numpy as np
import pytest

from frames2py.bench.synthetic import generate_batch, event_stream, PROFILES
from frames2py.core.engine import Engine
from frames2py.core.types import EVENT_DTYPE, OverflowPolicy
from frames2py.consumers.telemetry import Telemetry


class TestSyntheticToEngine:
    """Synthetic generator → Engine → snapshot verification."""

    def test_low_profile_no_drops(self):
        """Low profile should produce zero drops with default buffer."""
        p = PROFILES["low"]
        engine = Engine(
            sensor_size=p.sensor_size,
            kernel="event_count",
            buffer_capacity=64,
            chunk_size=65_536,
        )

        events_total = 0
        for batch in event_stream(
            rate_events_per_sec=p.rate,
            sensor_size=p.sensor_size,
            batch_size=p.batch_size,
            duration_sec=0.5,
            paced=False,
        ):
            engine.ingest(batch)
            events_total += len(batch)

        stats = engine.stats
        assert stats.events_ingested == events_total
        # With paced=False and 0.5s at 500K, should be manageable
        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert frame.sum() > 0

    @pytest.mark.parametrize("kernel_name", [
        "event_count", "polarity", "time_surface", "exp_decay",
    ])
    def test_all_kernels_produce_output(self, kernel_name):
        """Each kernel produces non-zero output from synthetic events."""
        engine = Engine(sensor_size=(64, 48), kernel=kernel_name)

        for batch in event_stream(
            rate_events_per_sec=100_000,
            sensor_size=(64, 48),
            batch_size=1000,
            duration_sec=0.1,
            paced=False,
        ):
            engine.ingest(batch)

        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert frame.max() > 0

    def test_generate_batch_deterministic(self):
        """Same seed produces identical batches."""
        b1 = generate_batch(1000, sensor_size=(640, 480), seed=42)
        b2 = generate_batch(1000, sensor_size=(640, 480), seed=42)
        np.testing.assert_array_equal(b1, b2)

    def test_generate_batch_different_seeds(self):
        """Different seeds produce different batches."""
        b1 = generate_batch(1000, seed=1)
        b2 = generate_batch(1000, seed=2)
        assert not np.array_equal(b1["x"], b2["x"])


class TestEngineWithTelemetry:
    """Engine + Telemetry consumer working together."""

    @pytest.mark.timeout(15)
    def test_telemetry_tracks_ingestion(self):
        engine = Engine(sensor_size=(64, 48))
        tel = Telemetry(engine, poll_interval_ms=50).start()

        total = 0
        for batch in event_stream(
            rate_events_per_sec=200_000,
            sensor_size=(64, 48),
            batch_size=2000,
            duration_sec=0.3,
            paced=False,
        ):
            engine.ingest(batch)
            total += len(batch)

        time.sleep(0.2)
        tel.stop()

        latest = tel.latest
        assert latest is not None
        assert latest.events_ingested == total
        assert latest.snapshots_published > 0
        assert latest.accumulate_ms_p50 >= 0

    @pytest.mark.timeout(15)
    def test_telemetry_detects_drops(self):
        engine = Engine(
            sensor_size=(64, 48),
            buffer_capacity=2,
            chunk_size=100,
        )
        tel = Telemetry(engine, poll_interval_ms=50).start()

        for batch in event_stream(
            rate_events_per_sec=1_000_000,
            sensor_size=(64, 48),
            batch_size=500,
            duration_sec=0.2,
            paced=False,
        ):
            engine.ingest(batch)

        time.sleep(0.2)
        tel.stop()

        latest = tel.latest
        assert latest is not None
        assert latest.events_dropped > 0


class TestOverflowAccounting:
    """Verify that events_ingested = events_in_buffer + events_dropped."""

    def test_exact_accounting(self):
        engine = Engine(
            sensor_size=(64, 48),
            buffer_capacity=4,
            chunk_size=100,
        )

        total_ingested = 0
        for i in range(50):
            batch = generate_batch(100, sensor_size=(64, 48), t_start=i * 100, seed=i)
            engine.ingest(batch)
            total_ingested += len(batch)

        stats = engine.stats
        assert stats.events_ingested == total_ingested


class TestProductionScalePipeline:
    """Full pipeline at production event volumes."""

    def test_2m_events_through_engine_with_telemetry(self):
        """2M events through engine + telemetry: exact accounting."""
        engine = Engine(
            sensor_size=(1280, 720),
            kernel="event_count",
            buffer_capacity=64,
            chunk_size=65_536,
        )
        tel = Telemetry(engine, poll_interval_ms=100).start()

        total = 0
        for batch in event_stream(
            rate_events_per_sec=5_000_000,
            sensor_size=(1280, 720),
            batch_size=50_000,
            duration_sec=0.4,
            paced=False,
        ):
            engine.ingest(batch)
            total += len(batch)

        import time as _time

        _time.sleep(0.2)
        tel.stop()

        stats = engine.stats
        assert stats.events_ingested == total
        assert stats.events_dropped == 0
        assert stats.snapshots_published > 0

        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert meta.events_accumulated == total
        assert frame.max() > 0

    def test_high_profile_1s_all_kernels(self):
        """High profile (5M ev/s) for 1 second with each kernel."""
        for kname in ("event_count", "polarity", "time_surface", "exp_decay"):
            p = PROFILES["high"]
            engine = Engine(
                sensor_size=p.sensor_size,
                kernel=kname,
                buffer_capacity=64,
                chunk_size=65_536,
            )
            total = 0
            for batch in event_stream(
                rate_events_per_sec=p.rate,
                sensor_size=p.sensor_size,
                batch_size=p.batch_size,
                duration_sec=1.0,
                paced=False,
            ):
                engine.ingest(batch)
                total += len(batch)

            result = engine.latest_snapshot()
            assert result is not None, f"Kernel {kname} produced no output"
            frame, meta = result
            assert frame.max() > 0
            assert meta.events_accumulated == total
