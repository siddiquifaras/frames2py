"""``frames2py.replay.paced``: timing against a fake clock, discontinuities, identity, errors."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frames2py import EVENT_DTYPE
from frames2py.replay import paced

DATA = Path(__file__).resolve().parent / "data"

pytestmark = pytest.mark.timeout(20)


class FakeClock:
    """A clock that only moves when slept on; records the time of every yield."""

    def __init__(self) -> None:
        self.now = 5_000_000_000
        self.sleeps: list[float] = []

    def __call__(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += round(seconds * 1e9)


def batch(*timestamps: int) -> np.ndarray:
    events = np.zeros(len(timestamps), dtype=EVENT_DTYPE)
    events["t"] = timestamps
    return events


def yield_times(batches: list[np.ndarray], speed: float = 1, work_ns: int = 0) -> tuple[list[int], FakeClock]:
    """Clock readings (ns after the first nonempty batch arrives) at which each batch is yielded."""
    clock = FakeClock()
    start = clock.now
    times = []
    for got, expected in zip(paced(batches, speed, clock=clock, sleep=clock.sleep), batches, strict=True):
        assert got is expected
        times.append(clock.now - start)
        clock.now += work_ns
    return times, clock


class TestTiming:
    def test_each_batch_waits_for_its_largest_timestamp(self) -> None:
        times, _ = yield_times([batch(100, 150), batch(160, 400), batch(900)])
        assert times == [50_000, 300_000, 800_000]  # (M - t0) µs, as ns

    @pytest.mark.parametrize("speed", [0.5, 2, 10, 3.0])
    def test_speed_scales_the_waits(self, speed: float) -> None:
        times, _ = yield_times([batch(0), batch(1_000), batch(4_000)], speed=speed)
        assert times == [0, math.ceil(1_000_000 / speed), math.ceil(4_000_000 / speed)]

    def test_time_spent_by_the_consumer_counts(self) -> None:
        times, clock = yield_times([batch(0), batch(10_000), batch(20_000)], work_ns=4_000_000)
        assert times == [0, 10_000_000, 20_000_000]
        assert clock.sleeps == [0.006, 0.006]

    def test_a_late_consumer_gets_batches_at_once_and_nothing_is_skipped(self) -> None:
        batches = [batch(0), batch(1_000), batch(2_000), batch(30_000)]
        times, _ = yield_times(batches, work_ns=10_000_000)
        assert times == [0, 10_000_000, 20_000_000, 30_000_000]

    def test_a_sleep_that_returns_early_is_repeated(self) -> None:
        clock = FakeClock()

        def short_sleep(seconds: float) -> None:
            clock.sleeps.append(seconds)
            clock.now += round(seconds * 1e9 / 2) + 1

        start = clock.now
        list(paced([batch(0), batch(8_000)], clock=clock, sleep=short_sleep))
        assert clock.now - start >= 8_000_000 and len(clock.sleeps) > 1

    def test_one_batch_waits_for_its_own_span(self) -> None:
        assert yield_times([batch(7, 5, 2_007)])[0] == [2_002_000]


class TestDiscontinuities:
    def test_a_backward_jump_is_due_at_once_and_keeps_the_maximum(self) -> None:
        # The clock restarts near 0 after 1 s; M stays at 1 s until the restarted clock passes it.
        times, _ = yield_times([batch(0), batch(1_000_000), batch(10), batch(500_000), batch(1_200_000)])
        assert times == [0, 1_000_000_000, 1_000_000_000, 1_000_000_000, 1_200_000_000]

    def test_out_of_order_within_and_across_batches(self) -> None:
        times, _ = yield_times([batch(500, 100, 300), batch(200, 450), batch(900, 50)])
        assert times == [400_000, 400_000, 800_000]

    def test_a_forward_spike_waits_and_later_batches_wait_behind_it(self) -> None:
        times, _ = yield_times([batch(0), batch(3_600_000_000), batch(1_000)])
        assert times == [0, 3_600_000_000_000, 3_600_000_000_000]


class TestBatches:
    def test_no_batches(self) -> None:
        clock = FakeClock()
        assert list(paced([], clock=clock, sleep=clock.sleep)) == [] and clock.sleeps == []

    def test_empty_batches_are_yielded_at_once_and_set_nothing(self) -> None:
        empty = np.empty(0, dtype=EVENT_DTYPE)
        times, _ = yield_times([empty, batch(1_000), empty, batch(3_000), empty])
        assert times == [0, 0, 0, 2_000_000, 2_000_000]

    def test_the_same_objects_are_yielded_unchanged(self) -> None:
        batches = [batch(1, 2), batch(3)]
        before = [b.copy() for b in batches]
        clock = FakeClock()
        out = list(paced(iter(batches), clock=clock, sleep=clock.sleep))
        assert all(a is b for a, b in zip(out, batches)) and len(out) == 2
        assert all(np.array_equal(a, b) for a, b in zip(batches, before))

    def test_a_reader_is_replayed_batch_for_batch(self) -> None:
        from frames2py.adapters import evt

        with evt.open(DATA / "active_marker_head.evt3.raw") as reader:
            unpaced = [b.copy() for b in reader]
        clock = FakeClock()
        with evt.open(DATA / "active_marker_head.evt3.raw") as reader:
            replayed = list(paced(reader, 1_000, clock=clock, sleep=clock.sleep))
        assert [b.tobytes() for b in replayed] == [b.tobytes() for b in unpaced]
        span_us = int(np.concatenate(unpaced)["t"].max()) - int(unpaced[0]["t"].min())
        assert clock.now - 5_000_000_000 == pytest.approx(span_us * 1000 / 1_000, abs=1)

    def test_a_malformed_batch_raises_when_reached(self) -> None:
        clock = FakeClock()
        replay = paced([batch(1), [1, 2, 3]], clock=clock, sleep=clock.sleep)  # type: ignore[list-item]
        assert len(next(replay)) == 1
        with pytest.raises(TypeError):
            next(replay)


class TestArguments:
    @pytest.mark.parametrize("speed", [0, -1, math.inf, -math.inf, math.nan, True, "2", None])
    def test_speed_must_be_finite_and_positive_on_the_call(self, speed: Any) -> None:
        with pytest.raises(ValueError, match="speed"):
            paced([batch(1)], speed)

    def test_no_thread_is_started(self) -> None:
        clock = FakeClock()
        before = threading.active_count()
        list(paced([batch(0), batch(1_000)], clock=clock, sleep=clock.sleep))
        assert threading.active_count() == before
