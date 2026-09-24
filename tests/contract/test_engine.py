"""Engine: snapshots, publication cadence, windowed and running kernels, lifecycle, stats."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from tests.contract.api import EVENT_DTYPE, accumulated, impl
from tests.contract.helpers import ALL, KERNELS, SENSOR, assert_matches, events, random_events

HOUR_MS = 3_600_000.0


def _publish(oracle: Any, windowed: bool) -> None:
    if windowed:
        oracle.close_window()


class TestSnapshots:
    @pytest.mark.parametrize("kernel", ALL)
    def test_none_before_the_first_publication(self, kernel: str) -> None:
        assert KERNELS[kernel].engine().snapshot() is None

    @pytest.mark.parametrize("kernel", ALL)
    def test_frame_and_metadata_match_the_oracle(self, kernel: str) -> None:
        case = KERNELS[kernel]
        engine = case.engine(interval_ms=0.0)
        oracle = case.oracle()
        for seed in range(4):
            batch = random_events(seed, 40)
            engine.ingest(batch)
            oracle.accumulate(batch)
            frame, meta = engine.snapshot()
            assert_matches(frame, oracle)
            assert meta.watermark == oracle.watermark
            _publish(oracle, case.windowed)

    def test_reads_are_non_destructive_copies(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0), (2, 1, 1, 0)))
        first, meta = engine.snapshot()
        expected = first.copy()
        first[...] = 99
        for _ in range(3):
            frame, again = engine.snapshot()
            np.testing.assert_array_equal(frame, expected)
            assert again.sequence == meta.sequence

    def test_sequence_increases_with_every_publication(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        seen = []
        for t in range(5):
            engine.ingest(events((t, 1, 1, 0)))
            seen.append(engine.snapshot()[1].sequence)
        assert all(a < b for a, b in zip(seen, seen[1:]))

    def test_watermark_is_none_when_nothing_was_accumulated(self) -> None:
        engine = KERNELS["time_surface"].engine(interval_ms=0.0)
        engine.ingest(events((50, SENSOR[0], 0, 0)))
        frame, meta = engine.snapshot()
        assert meta.watermark is None
        assert not frame.any()

    def test_timestamp_decay_snapshot_is_evaluated_at_its_watermark(self) -> None:
        case = KERNELS["timestamp_decay"]
        engine = case.engine(interval_ms=0.0)
        oracle = case.oracle()
        for batch in (events((0, 1, 1, 0), (30, 2, 2, 1)), events((5, 1, 1, 1)), events((9_999, SENSOR[0], 0, 0))):
            engine.ingest(batch)
            oracle.accumulate(batch)
            frame, meta = engine.snapshot()
            assert meta.watermark == 30
            assert_matches(frame, oracle)


class TestCadence:
    def test_interval_zero_publishes_on_every_ingest(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        for n in range(1, 6):
            engine.ingest(events((n, 1, 1, 0)))
            assert engine.stats.snapshots_published == n

    def test_empty_ingest_still_publishes_at_interval_zero(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.ingest(events())
        assert engine.stats.snapshots_published == 2
        assert not engine.snapshot()[0].any()

    def test_first_ingest_publishes_then_at_most_once_per_interval(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=HOUR_MS)
        engine.ingest(events((1, 1, 1, 0)))
        assert engine.stats.snapshots_published == 1
        for t in range(2, 10):
            engine.ingest(events((t, 1, 1, 0)))
        assert engine.stats.snapshots_published == 1
        assert int(engine.snapshot()[0].sum()) == 1

    def test_first_ingest_after_the_interval_publishes(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=5.0)
        engine.ingest(events((1, 1, 1, 0)))
        deadline = time.monotonic() + 0.050
        while time.monotonic() < deadline:  # elapsed time is the input here, not a sync
            pass
        engine.ingest(events((2, 2, 2, 0)))
        assert engine.stats.snapshots_published == 2


class TestWindowedAndRunning:
    @pytest.mark.parametrize("kernel", ALL)
    def test_publication_closes_only_windowed_kernels(self, kernel: str) -> None:
        case = KERNELS[kernel]
        engine = case.engine(interval_ms=0.0)
        oracle = case.oracle()
        for batch in (events((1, 1, 1, 1)), events((2, 3, 2, 0)), events((3, 1, 1, 0))):
            engine.ingest(batch)
            oracle.accumulate(batch)
            assert_matches(engine.snapshot()[0], oracle)
            _publish(oracle, case.windowed)

    def test_time_surface_timestamps_stay_exact(self) -> None:
        engine = KERNELS["time_surface"].engine(interval_ms=0.0)
        engine.ingest(events((20_000_001, 1, 1, 0), (2**63 - 1, 2, 1, 1)))
        frame = engine.snapshot()[0]
        assert [int(frame[1, 1]), int(frame[1, 2])] == [20_000_001, 2**63 - 1]
        assert frame.dtype == np.uint64

    def test_exp_decay_decays_once_per_ingest_however_large(self) -> None:
        batch = np.zeros(200_000, dtype=EVENT_DTYPE)
        batch["x"], batch["y"] = 1, 1
        engine = impl.Engine(SENSOR, impl.ExpDecay(0.5), snapshot_interval_ms=0.0)
        engine.ingest(batch)
        assert engine.snapshot()[0][1, 1] == np.float32(200_000.0)

    def test_exp_decay_decays_once_per_ingest(self) -> None:
        engine = impl.Engine(SENSOR, impl.ExpDecay(0.5), snapshot_interval_ms=0.0)
        for _ in range(3):
            engine.ingest(events((0, 1, 1, 0)))
        assert engine.snapshot()[0][1, 1] == np.float32(1.75)


class TestLifecycle:
    @pytest.mark.parametrize("kernel", ALL)
    def test_stop_publishes_the_pending_window(self, kernel: str) -> None:
        case = KERNELS[kernel]
        engine = case.engine(interval_ms=HOUR_MS)
        oracle = case.oracle()
        first, pending = events((1, 1, 1, 0)), events((5, 2, 2, 1), (6, 3, 1, 0))
        engine.ingest(first)
        oracle.accumulate(first)
        _publish(oracle, case.windowed)
        engine.ingest(pending)
        oracle.accumulate(pending)
        engine.stop()
        assert engine.stats.snapshots_published == 2
        frame, meta = engine.snapshot()
        assert_matches(frame, oracle)
        assert meta.watermark == 6

    def test_stop_with_nothing_accumulated_since_the_last_publication(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=HOUR_MS)
        engine.ingest(events((1, 1, 1, 0)))
        engine.ingest(events((2, SENSOR[0], 0, 0)))  # out of bounds: nothing accumulated
        engine.stop()
        assert engine.stats.snapshots_published == 1
        assert int(engine.snapshot()[0][1, 1]) == 1

    def test_stop_before_any_ingest_publishes_nothing(self) -> None:
        engine = KERNELS["event_count"].engine()
        engine.stop()
        assert engine.snapshot() is None
        assert engine.stats.snapshots_published == 0

    def test_stopped_ingest_is_a_no_op_and_the_snapshot_stays(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.stop()
        frame, meta = engine.snapshot()
        stats = engine.stats
        engine.ingest(events((2, 2, 2, 0)))
        engine.ingest(events((2**63, 1, 1, 0)))  # the lifecycle check comes first
        after = engine.stats
        assert (after.events_ingested, after.events_out_of_bounds, after.snapshots_published) == (
            stats.events_ingested, stats.events_out_of_bounds, stats.snapshots_published)
        again, again_meta = engine.snapshot()
        np.testing.assert_array_equal(again, frame)
        assert again_meta.sequence == meta.sequence

    def test_start_resumes_ingestion(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.stop()
        engine.start()
        engine.ingest(events((2, 2, 2, 0)))
        frame, meta = engine.snapshot()
        assert int(frame[2, 2]) == 1
        assert meta.watermark == 2

    @pytest.mark.parametrize("kernel", ["event_count", "polarity"])
    def test_watermark_spans_windows_until_reset(self, kernel: str) -> None:
        engine = KERNELS[kernel].engine(interval_ms=0.0)
        engine.ingest(events((100, 1, 1, 1)))
        assert engine.snapshot()[1].watermark == 100
        engine.ingest(events())  # publishes an empty window
        frame, meta = engine.snapshot()
        assert not frame.any()
        assert meta.watermark == 100
        engine.reset()
        assert engine.snapshot() is None
        engine.ingest(events((200, 2, 2, 0)))
        assert engine.snapshot()[1].watermark == 200

    def test_first_ingest_after_reset_publishes(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=HOUR_MS)
        engine.ingest(events((1, 1, 1, 0)))
        engine.reset()
        engine.ingest(events((2, 2, 2, 0)))
        assert engine.stats.snapshots_published == 1
        frame, meta = engine.snapshot()
        assert int(frame[2, 2]) == 1 and meta.watermark == 2

    def test_sequence_keeps_increasing_across_reset(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.ingest(events((2, 1, 1, 0)))
        before = engine.snapshot()[1].sequence
        engine.reset()
        engine.ingest(events((3, 1, 1, 0)))
        assert engine.snapshot()[1].sequence > before

    def test_uptime_keeps_counting_across_reset(self) -> None:
        engine = KERNELS["event_count"].engine()
        while engine.stats.uptime_ns < 2_000_000:  # elapsed time is the input here
            pass
        before = engine.stats.uptime_ns
        engine.reset()
        assert engine.stats.uptime_ns >= before

    @pytest.mark.parametrize("kernel", ALL)
    def test_reset_clears_state_counters_and_snapshot(self, kernel: str) -> None:
        case = KERNELS[kernel]
        engine = case.engine(interval_ms=0.0)
        engine.ingest(random_events(4, 60, t_max=10_000))
        engine.reset()
        assert engine.snapshot() is None
        stats = engine.stats
        assert (stats.events_ingested, stats.events_out_of_bounds, stats.snapshots_published) == (0, 0, 0)
        oracle = case.oracle()
        batch = events((3, 1, 1, 1))
        engine.ingest(batch)
        oracle.accumulate(batch)
        frame, meta = engine.snapshot()
        assert_matches(frame, oracle)
        assert meta.watermark == 3


class TestStats:
    def test_fields(self) -> None:
        stats = KERNELS["event_count"].engine().stats
        for name in ("events_ingested", "events_out_of_bounds", "snapshots_published", "uptime_ns"):
            assert isinstance(getattr(stats, name), int), name
        for removed in ("events_dropped", "chunks_dropped", "buffer_fill_ratio"):
            assert not hasattr(stats, removed), removed

    def test_frozen_view(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        stats = engine.stats
        with pytest.raises(AttributeError):
            stats.events_ingested = 0
        engine.ingest(events((2, 1, 1, 0), (3, SENSOR[0], 0, 0)))
        assert stats.snapshots_published == 1
        assert engine.stats.snapshots_published == 2

    def test_out_of_bounds_count_and_accumulated_events(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        oracle = KERNELS["event_count"].oracle()
        for seed in range(5):
            batch = random_events(seed, 70)
            engine.ingest(batch)
            oracle.accumulate(batch)
        stats = engine.stats
        assert stats.events_out_of_bounds == oracle.out_of_bounds
        assert accumulated(stats) == oracle.accumulated

    def test_one_large_call_loses_no_events(self) -> None:
        batch = np.zeros(5_000_000, dtype=EVENT_DTYPE)
        batch["t"] = np.arange(len(batch), dtype=np.uint64)
        batch["x"] = np.arange(len(batch)) % SENSOR[0]
        batch["y"] = (np.arange(len(batch)) // SENSOR[0]) % SENSOR[1]
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(batch)
        assert int(engine.snapshot()[0].sum(dtype=np.uint64)) == len(batch)
        assert accumulated(engine.stats) == len(batch)

    def test_uptime_does_not_go_backwards(self) -> None:
        engine = KERNELS["event_count"].engine()
        first = engine.stats.uptime_ns
        engine.ingest(events((1, 1, 1, 0)))
        assert engine.stats.uptime_ns >= first >= 0
