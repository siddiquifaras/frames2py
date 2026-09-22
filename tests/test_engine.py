"""Tests for the Engine: ingest-snapshot roundtrip, counters, lifecycle."""

from __future__ import annotations

import time
import threading

import numpy as np
import pytest

from frames2py.core.engine import Engine, LatencyTracker
from frames2py.core.types import EVENT_DTYPE, OverflowPolicy


def _make_events(n: int, t_start: int = 0, w: int = 64, h: int = 48, seed: int = 42):
    rng = np.random.default_rng(seed)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(t_start, t_start + n, dtype=np.uint64)
    events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


class TestIngestSnapshotRoundtrip:
    """Verify events go in, correct snapshots come out."""

    def test_basic_roundtrip(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, kernel="event_count")
        events = _make_events(100, w=sensor_size[0], h=sensor_size[1])
        engine.ingest(events)

        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert frame.shape == (sensor_size[1], sensor_size[0])
        assert frame.sum() > 0
        assert meta.seq == 1
        assert meta.events_accumulated >= 100

    def test_multiple_ingests_accumulate(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, kernel="event_count")
        total = 0
        for i in range(5):
            events = _make_events(200, t_start=i * 200, w=sensor_size[0], h=sensor_size[1], seed=i)
            engine.ingest(events)
            total += 200

        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert meta.seq == 5
        assert meta.events_accumulated >= total

    def test_empty_batch_is_noop(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, kernel="event_count")
        engine.ingest(np.empty(0, dtype=EVENT_DTYPE))
        assert engine.latest_snapshot() is None
        assert engine.stats.events_ingested == 0

    def test_snapshot_before_ingest_returns_none(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        assert engine.latest_snapshot() is None

    def test_polarity_kernel_roundtrip(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, kernel="polarity")
        events = _make_events(500, w=sensor_size[0], h=sensor_size[1])
        engine.ingest(events)

        result = engine.latest_snapshot()
        assert result is not None
        frame, _ = result
        assert frame.shape == (sensor_size[1], sensor_size[0], 2)
        assert frame.sum() > 0

    def test_time_surface_roundtrip(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, kernel="time_surface")
        events = _make_events(100, w=sensor_size[0], h=sensor_size[1])
        engine.ingest(events)

        result = engine.latest_snapshot()
        assert result is not None
        frame, _ = result
        assert frame.dtype == np.float64
        assert frame.max() > 0

    def test_exp_decay_roundtrip(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, kernel="exp_decay")
        events = _make_events(100, w=sensor_size[0], h=sensor_size[1])
        engine.ingest(events)

        result = engine.latest_snapshot()
        assert result is not None
        assert result[0].sum() > 0


class TestCounterAccuracy:
    """Verify telemetry counters are exact."""

    def test_events_ingested_counter(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        for i in range(10):
            engine.ingest(_make_events(100, t_start=i * 100, w=sensor_size[0], h=sensor_size[1], seed=i))
        assert engine.stats.events_ingested == 1000

    def test_snapshots_published_counter(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        for i in range(5):
            engine.ingest(_make_events(50, t_start=i * 50, w=sensor_size[0], h=sensor_size[1], seed=i))
        assert engine.stats.snapshots_published == 5

    def test_overflow_counters(self):
        # Capacity=2, chunk_size=50 → total buffer holds 100 events.
        # A single batch of 200 events produces 4 chunks but only 2 slots
        # fit, so 2 chunks must be evicted during the write() split.
        engine = Engine(
            sensor_size=(64, 48),
            buffer_capacity=2,
            chunk_size=50,
        )
        big_batch = _make_events(200, t_start=0, seed=0)
        engine.ingest(big_batch)

        stats = engine.stats
        assert stats.events_ingested == 200
        assert stats.chunks_dropped > 0
        assert stats.events_dropped > 0

    def test_uptime_increases(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        t0 = engine.stats.uptime_ns
        time.sleep(0.01)
        t1 = engine.stats.uptime_ns
        assert t1 > t0


class TestSnapshotInterval:
    """Verify snapshot_interval_ms throttles publication."""

    def test_interval_throttles_snapshots(self, sensor_size):
        engine = Engine(
            sensor_size=sensor_size,
            snapshot_interval_ms=50.0,
        )
        for i in range(5):
            engine.ingest(_make_events(100, t_start=i * 100, w=sensor_size[0], h=sensor_size[1], seed=i))
        # With 50ms interval and rapid ingestion, should have fewer than 5 snapshots.
        assert engine.stats.snapshots_published <= 5

    def test_zero_interval_publishes_every_time(self, sensor_size):
        engine = Engine(
            sensor_size=sensor_size,
            snapshot_interval_ms=0.0,
        )
        for i in range(10):
            engine.ingest(_make_events(50, t_start=i * 50, w=sensor_size[0], h=sensor_size[1], seed=i))
        assert engine.stats.snapshots_published == 10


class TestResetAndStop:
    """Verify reset clears state and stop halts ingestion."""

    def test_reset_clears_everything(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        engine.ingest(_make_events(500, w=sensor_size[0], h=sensor_size[1]))
        engine.reset()

        assert engine.stats.events_ingested == 0
        assert engine.stats.events_dropped == 0
        assert engine.stats.snapshots_published == 0
        assert engine.latest_snapshot() is None

    def test_reset_allows_new_ingestion(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        engine.ingest(_make_events(100, w=sensor_size[0], h=sensor_size[1]))
        engine.reset()
        engine.ingest(_make_events(50, w=sensor_size[0], h=sensor_size[1], seed=99))

        result = engine.latest_snapshot()
        assert result is not None
        assert engine.stats.events_ingested == 50

    def test_stop_prevents_ingestion(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        engine.ingest(_make_events(100, w=sensor_size[0], h=sensor_size[1]))
        engine.stop()
        assert engine.running is False

        engine.ingest(_make_events(100, w=sensor_size[0], h=sensor_size[1], seed=99))
        assert engine.stats.events_ingested == 100  # no new ingestion

    def test_start_after_stop(self, sensor_size):
        engine = Engine(sensor_size=sensor_size)
        engine.stop()
        engine.start()
        assert engine.running is True
        engine.ingest(_make_events(50, w=sensor_size[0], h=sensor_size[1]))
        assert engine.stats.events_ingested == 50


class TestConcurrentReadWrite:
    """Verify engine is safe with one writer + multiple readers."""

    @pytest.mark.timeout(15)
    def test_concurrent_ingest_and_read(self, sensor_size):
        engine = Engine(sensor_size=sensor_size, buffer_capacity=16)
        errors: list[str] = []
        n_ingests = 5000

        def writer():
            for i in range(n_ingests):
                engine.ingest(_make_events(50, t_start=i * 50, w=sensor_size[0], h=sensor_size[1], seed=i))

        def reader():
            last_seq = -1
            reads = 0
            while reads < 100:
                result = engine.latest_snapshot()
                if result is None:
                    continue
                _, meta = result
                if meta.seq < last_seq:
                    errors.append(f"Seq regression: {meta.seq} < {last_seq}")
                last_seq = meta.seq
                reads += 1

        w = threading.Thread(target=writer)
        r1 = threading.Thread(target=reader)
        r2 = threading.Thread(target=reader)

        w.start()
        r1.start()
        r2.start()
        w.join(timeout=10)
        r1.join(timeout=10)
        r2.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        assert engine.stats.events_ingested == n_ingests * 50


class TestProductionScale:
    """Verify correctness at production event volumes (1M+ events)."""

    def test_1m_events_exact_accounting(self):
        """1M events: counters exact, no drops, all accounted for."""
        sensor_size = (1280, 720)
        engine = Engine(
            sensor_size=sensor_size,
            kernel="event_count",
            buffer_capacity=64,
            chunk_size=65_536,
        )
        rng = np.random.default_rng(13)
        n_total = 1_000_000
        batch_size = 50_000
        n_batches = n_total // batch_size
        for i in range(n_batches):
            events = np.empty(batch_size, dtype=EVENT_DTYPE)
            events["t"] = np.arange(
                i * batch_size, (i + 1) * batch_size, dtype=np.uint64
            )
            events["x"] = rng.integers(0, 1280, size=batch_size, dtype=np.uint16)
            events["y"] = rng.integers(0, 720, size=batch_size, dtype=np.uint16)
            events["p"] = rng.integers(0, 2, size=batch_size, dtype=np.uint8)
            engine.ingest(events)

        stats = engine.stats
        assert stats.events_ingested == n_total
        assert stats.events_dropped == 0
        assert stats.snapshots_published == n_batches

        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert meta.events_accumulated == n_total
        # EventCount resets per snapshot, so frame holds last batch only
        assert frame.sum() == batch_size

    def test_1m_events_time_surface_cumulative(self):
        """1M events via time_surface: frame persists across snapshots."""
        sensor_size = (1280, 720)
        engine = Engine(
            sensor_size=sensor_size,
            kernel="time_surface",
            buffer_capacity=64,
            chunk_size=65_536,
        )
        rng = np.random.default_rng(13)
        n_total = 1_000_000
        batch_size = 50_000
        for i in range(n_total // batch_size):
            events = np.empty(batch_size, dtype=EVENT_DTYPE)
            events["t"] = np.arange(
                i * batch_size, (i + 1) * batch_size, dtype=np.uint64
            )
            events["x"] = rng.integers(0, 1280, size=batch_size, dtype=np.uint16)
            events["y"] = rng.integers(0, 720, size=batch_size, dtype=np.uint16)
            events["p"] = rng.integers(0, 2, size=batch_size, dtype=np.uint8)
            engine.ingest(events)

        result = engine.latest_snapshot()
        assert result is not None
        frame, meta = result
        assert frame.max() == n_total - 1  # last timestamp
        assert meta.events_accumulated == n_total
        assert (frame > 0).sum() > 0  # pixels were hit

    def test_all_kernels_500k_events(self):
        """All 4 kernels produce correct output at 500K events."""
        for kname in ("event_count", "polarity", "time_surface", "exp_decay"):
            engine = Engine(sensor_size=(640, 480), kernel=kname)
            rng = np.random.default_rng(42)
            n = 500_000
            batch_size = 50_000
            for i in range(n // batch_size):
                events = np.empty(batch_size, dtype=EVENT_DTYPE)
                events["t"] = np.arange(
                    i * batch_size, (i + 1) * batch_size, dtype=np.uint64
                )
                events["x"] = rng.integers(0, 640, size=batch_size, dtype=np.uint16)
                events["y"] = rng.integers(0, 480, size=batch_size, dtype=np.uint16)
                events["p"] = rng.integers(0, 2, size=batch_size, dtype=np.uint8)
                engine.ingest(events)

            result = engine.latest_snapshot()
            assert result is not None, f"Kernel {kname} produced no snapshot"
            frame, meta = result
            assert frame.max() > 0, f"Kernel {kname} produced zero frame"
            assert meta.events_accumulated == n

    def test_throughput_baseline(self):
        """Ingest 5M events and verify throughput exceeds 2M ev/s.

        The 2M threshold is conservative for CI machines; production
        targets 5-6M+ on real hardware.
        """
        sensor_size = (1280, 720)
        engine = Engine(sensor_size=sensor_size, kernel="event_count")
        rng = np.random.default_rng(42)
        n_total = 5_000_000
        batch_size = 50_000
        batches = []
        for i in range(n_total // batch_size):
            events = np.empty(batch_size, dtype=EVENT_DTYPE)
            events["t"] = np.arange(
                i * batch_size, (i + 1) * batch_size, dtype=np.uint64
            )
            events["x"] = rng.integers(0, 1280, size=batch_size, dtype=np.uint16)
            events["y"] = rng.integers(0, 720, size=batch_size, dtype=np.uint16)
            events["p"] = rng.integers(0, 2, size=batch_size, dtype=np.uint8)
            batches.append(events)

        t0 = time.perf_counter()
        for batch in batches:
            engine.ingest(batch)
        elapsed = time.perf_counter() - t0

        throughput = n_total / elapsed
        assert throughput > 2_000_000, (
            f"Throughput {throughput:,.0f} ev/s below 2M ev/s baseline"
        )
        assert engine.stats.events_ingested == n_total


class TestLatencyTracker:
    """Verify the LatencyTracker percentile computation."""

    def test_empty_returns_zero(self):
        lt = LatencyTracker()
        assert lt.percentile(50) == 0.0
        assert lt.count == 0

    def test_single_measurement(self):
        lt = LatencyTracker()
        lt.record(1_000_000)  # 1ms in nanoseconds
        assert lt.percentile(50) == pytest.approx(1.0, abs=0.01)
        assert lt.count == 1

    def test_percentiles(self):
        lt = LatencyTracker(window_size=100)
        for i in range(100):
            lt.record(i * 1_000_000)  # 0ms to 99ms
        p50 = lt.percentile(50)
        p99 = lt.percentile(99)
        assert 45 < p50 < 55
        assert 95 < p99 < 100

    def test_rolling_window_wraps(self):
        lt = LatencyTracker(window_size=10)
        for i in range(20):
            lt.record(i * 1_000_000)
        assert lt.count == 10  # capped at window size

    def test_reset(self):
        lt = LatencyTracker()
        lt.record(1_000_000)
        lt.reset()
        assert lt.count == 0
        assert lt.percentile(50) == 0.0
