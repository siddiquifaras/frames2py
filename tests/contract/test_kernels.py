"""Kernel semantics through ``Accumulator``, against the independent oracle."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import numpy as np
import pytest

from tests.contract.api import EVENT_DTYPE, impl
from tests.contract.helpers import (
    ALL,
    BACKWARD_JUMP,
    FORWARD_SPIKE,
    KERNELS,
    ORDER_INVARIANT,
    SENSOR,
    assert_matches,
    events,
    partitions,
    random_events,
)
from tests.oracle import ReferenceAccumulator

SLOW = os.environ.get("FRAMES2PY_SLOW_TESTS") == "1"

_OVERFLOW_EDGE = 2.0**63 / sys.float_info.max
"""Below this tau, a timestamp difference of 2**63 divided by tau exceeds float64's range."""


def _one_pixel(n: int, t0: int = 0, p: int = 1) -> np.ndarray:
    batch = np.zeros(n, dtype=EVENT_DTYPE)
    batch["t"] = np.arange(t0, t0 + n, dtype=np.uint64)
    batch["x"] = 2
    batch["y"] = 1
    batch["p"] = p
    return batch


class TestAgainstTheOracle:
    @pytest.mark.parametrize("kernel", ALL)
    def test_output_shape_and_dtype(self, kernel: str) -> None:
        case = KERNELS[kernel]
        acc = case.accumulator()
        empty = acc.read()
        acc.accumulate(events((3, 1, 1, 1)))
        for frame in (empty, acc.read()):
            assert frame.shape == case.shape(SENSOR)
            assert frame.dtype == case.dtype

    @pytest.mark.parametrize("kernel", ALL)
    @pytest.mark.parametrize("seed", range(6))
    def test_sequence_of_calls(self, kernel: str, seed: int) -> None:
        acc = KERNELS[kernel].accumulator()
        oracle = KERNELS[kernel].oracle()
        for call in range(5):
            batch = random_events(100 * seed + call, 30)
            acc.accumulate(batch)
            oracle.accumulate(batch)
            assert_matches(acc.read(), oracle)
            assert acc.watermark == oracle.watermark
            assert acc.events_out_of_bounds == oracle.out_of_bounds

    @pytest.mark.parametrize("kernel", ORDER_INVARIANT)
    @pytest.mark.parametrize("seed", range(10))
    def test_arrival_order_and_call_partition_do_not_matter(self, kernel: str, seed: int) -> None:
        batch = random_events(seed, 200)
        oracle = KERNELS[kernel].oracle()
        oracle.accumulate(batch)
        acc = KERNELS[kernel].accumulator()
        for part in partitions(batch, seed):
            acc.accumulate(part)
        assert_matches(acc.read(), oracle)
        assert acc.watermark == oracle.watermark

    @pytest.mark.parametrize("kernel", ALL)
    def test_empty_calls_are_accepted(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        oracle = KERNELS[kernel].oracle()
        for batch in (events(), events((4, 1, 1, 1)), events()):
            acc.accumulate(batch)
            oracle.accumulate(batch)
        assert_matches(acc.read(), oracle)

    @pytest.mark.parametrize("kernel", ALL)
    def test_read_is_a_repeatable_copy(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(random_events(1, 50, out_of_bounds=False))
        first = acc.read()
        snapshot = first.copy()
        first[...] = 0
        np.testing.assert_array_equal(acc.read(), snapshot)
        np.testing.assert_array_equal(acc.read(), acc.read())

    @pytest.mark.parametrize("kernel", ALL)
    def test_reset_clears_everything(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(random_events(2, 50))
        acc.reset()
        assert acc.watermark is None
        assert acc.events_out_of_bounds == 0
        fresh = KERNELS[kernel].oracle()
        assert_matches(acc.read(), fresh)
        batch = events((7, 1, 1, 0))
        acc.accumulate(batch)
        fresh.accumulate(batch)
        assert_matches(acc.read(), fresh)  # exp_decay: no decay steps survive reset


class TestCounts:
    @pytest.mark.parametrize("kernel", ["event_count", "polarity"])
    def test_counts_stay_exact_past_2_to_the_24(self, kernel: str) -> None:
        # float32 counting stops at 2**24.
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(_one_pixel(2**24 + 3))
        frame = acc.read()
        count = frame[1, 2] if kernel == "event_count" else frame[1, 2, 1]
        assert int(count) == 2**24 + 3

    @pytest.mark.skipif(not SLOW, reason="accumulates 2**32 events; set FRAMES2PY_SLOW_TESTS=1")
    @pytest.mark.parametrize("kernel", ["event_count", "polarity"])
    def test_counts_wrap_modulo_2_to_the_32(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        chunk = _one_pixel(2**24)
        for _ in range(2**8):
            acc.accumulate(chunk)
        acc.accumulate(_one_pixel(5))
        frame = acc.read()
        count = frame[1, 2] if kernel == "event_count" else frame[1, 2, 1]
        assert int(count) == 5


class TestTimeSurface:
    def test_large_timestamps_are_exact(self) -> None:
        acc = KERNELS["time_surface"].accumulator()
        acc.accumulate(events((2**63 - 1, 0, 0, 0), (2**53 + 1, 1, 0, 0), (20_000_001, 2, 0, 1)))
        frame = acc.read()
        assert [int(v) for v in frame[0, :3]] == [2**63 - 1, 2**53 + 1, 20_000_001]

    def test_event_at_t0_reads_as_no_event(self) -> None:
        acc = KERNELS["time_surface"].accumulator()
        acc.accumulate(events((0, 1, 1, 1)))
        assert not acc.read().any()


class TestExpDecay:
    def test_decay_is_once_per_call_however_large_the_call(self) -> None:
        acc = impl.Accumulator(SENSOR, impl.ExpDecay(0.5))
        acc.accumulate(_one_pixel(200_000))
        assert acc.read()[1, 2] == np.float32(200_000.0)

    def test_results_depend_on_call_boundaries(self) -> None:
        batch = events((0, 1, 1, 0), (1, 1, 1, 0))
        whole = impl.Accumulator(SENSOR, impl.ExpDecay(0.5))
        whole.accumulate(batch)
        split = impl.Accumulator(SENSOR, impl.ExpDecay(0.5))
        split.accumulate(batch[:1].copy())
        split.accumulate(batch[1:].copy())
        assert whole.read()[1, 1] == np.float32(2.0)
        assert split.read()[1, 1] == np.float32(1.5)

    def test_hot_pixel_over_many_calls(self) -> None:
        # Thousands of increments into one pixel: storage that rounds each addition to
        # float32 drifts several ULP from the true value; the output must not.
        sensor = (32, 24)
        acc = impl.Accumulator(sensor, impl.ExpDecay(0.999))
        oracle = ReferenceAccumulator("exp_decay", sensor, decay=0.999)
        rng = np.random.default_rng(11)
        for call in range(3_000):
            flat = np.concatenate([np.full(60, 5), rng.integers(0, sensor[0] * sensor[1], 20)])
            batch = np.zeros(len(flat), dtype=EVENT_DTYPE)
            batch["t"], batch["x"], batch["y"] = call, flat % sensor[0], flat // sensor[0]
            acc.accumulate(batch)
            oracle.accumulate(batch)
            if call % 750 == 749:
                assert_matches(acc.read(), oracle)

    def test_across_renormalisation(self) -> None:
        # 400 calls at decay 0.5 take the global scale to 2**-400, far past float32 and
        # float64 exponent range, so any correct lazy scale must renormalise.
        acc = impl.Accumulator(SENSOR, impl.ExpDecay(0.5))
        oracle = ReferenceAccumulator("exp_decay", SENSOR, decay=0.5)
        for call in range(400):
            rows = [(call, 1, 1, 0)] + ([(call, 4, 2, 1)] if call < 10 else [])
            acc.accumulate(events(*rows))
            oracle.accumulate(events(*rows))
            if call % 50 == 49:
                assert_matches(acc.read(), oracle)
        assert acc.read()[2, 4] == np.float32(0.0)


class TestTimestampDecay:
    def _pair(self, tau: float = 10.0) -> tuple[Any, ReferenceAccumulator]:
        return impl.Accumulator(SENSOR, impl.TimestampDecay(tau)), ReferenceAccumulator(
            "timestamp_decay", SENSOR, tau_us=tau
        )

    def _feed(self, acc: Any, oracle: ReferenceAccumulator, *batches: np.ndarray) -> None:
        for batch in batches:
            acc.accumulate(batch)
            oracle.accumulate(batch)

    def test_multiple_events_per_pixel_at_the_watermark(self) -> None:
        acc, oracle = self._pair()
        self._feed(acc, oracle, events((0, 1, 1, 0), (10, 1, 1, 1), (10, 1, 1, 7), (5, 3, 2, 0)))
        assert acc.watermark == 10
        assert_matches(acc.read(), oracle)

    def test_idle_calls_do_not_change_the_surface(self) -> None:
        acc, oracle = self._pair()
        self._feed(acc, oracle, events((10, 1, 1, 0)))
        first = acc.read()
        self._feed(acc, oracle, events(), events((99_999, SENSOR[0], 0, 0)))
        np.testing.assert_array_equal(acc.read(), first)
        assert acc.watermark == 10

    def test_a_newer_event_decays_every_pixel(self) -> None:
        acc, oracle = self._pair()
        self._feed(acc, oracle, events((0, 1, 1, 0)), events((20, 3, 2, 0)))
        assert_matches(acc.read(), oracle)

    def test_older_events_arriving_late(self) -> None:
        acc, oracle = self._pair(tau=1_000_000.0)
        self._feed(acc, oracle, events((1_000_000, 1, 1, 0)), events((0, 1, 1, 0), (500_000, 2, 1, 1)))
        assert_matches(acc.read(), oracle)

    def test_differences_taken_exactly_near_2_to_the_53(self) -> None:
        acc, oracle = self._pair(tau=1.0)
        self._feed(acc, oracle, events((2**53, 1, 1, 0), (2**53 + 1, 2, 1, 0)))
        assert_matches(acc.read(), oracle)

    @pytest.mark.parametrize("seed", range(4))
    def test_rebasing_over_long_event_time_spans(self, seed: int) -> None:
        # (T - T0) / tau passes 665 many times over; recent contributions must survive.
        acc, oracle = self._pair(tau=1_000.0)
        for call in range(40):
            batch = random_events(1_000 * seed + call, 25, t_max=50_000)
            batch["t"] += np.uint64(call * 50_000)
            self._feed(acc, oracle, batch)
            assert_matches(acc.read(), oracle)
        assert acc.watermark == oracle.watermark

    def test_one_call_spanning_past_float64_exp_range(self) -> None:
        # The first call spans 1000 tau; exp(1000) overflows float64.
        acc, oracle = self._pair(tau=1.0)
        self._feed(acc, oracle, events((0, 1, 1, 0), (1_000, 2, 1, 0), (999, 2, 1, 1)))
        assert_matches(acc.read(), oracle)

    @pytest.mark.parametrize(
        "tau", [5e-324, 1e-300, _OVERFLOW_EDGE * (1 - 2**-40), _OVERFLOW_EDGE * (1 + 2**-40), 1e-280]
    )
    def test_tiny_tau_across_the_whole_timestamp_range(self, tau: float) -> None:
        # The oracle can't evaluate these: every event older than the watermark contributes
        # 0, each event at the watermark 1. No warning, so none can raise mid-call.
        acc = impl.Accumulator(SENSOR, impl.TimestampDecay(tau))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            acc.accumulate(events((0, 1, 1, 0), (2**63 - 1, 2, 1, 0)))
            acc.accumulate(events((5, 3, 1, 0), (2**63 - 1, 2, 1, 1)))
            frame = acc.read()
        expected = np.zeros_like(frame)
        expected[1, 2] = 2.0
        np.testing.assert_array_equal(frame, expected)

    def test_extreme_timestamp_then_ordinary_ones(self) -> None:
        acc, oracle = self._pair()
        self._feed(acc, oracle, events((5, 1, 1, 0)), events((2**63 - 1, 2, 2, 0)), events((100, 1, 1, 0)))
        assert_matches(acc.read(), oracle)

    @pytest.mark.parametrize("seed", range(6))
    def test_order_and_partition_with_wide_timestamps(self, seed: int) -> None:
        batch = random_events(seed, 300, t_max=10**12)
        acc, oracle = self._pair(tau=5e9)
        oracle.accumulate(batch)
        for part in partitions(batch, seed, parts=7):
            acc.accumulate(part)
        assert_matches(acc.read(), oracle)


@pytest.mark.filterwarnings("error")
class TestTimestampDiscontinuities:
    def _feed(self, kernel: str, batches: list[np.ndarray], one_call: bool) -> Any:
        acc = KERNELS[kernel].accumulator()
        oracle = KERNELS[kernel].oracle()
        for batch in [np.concatenate(batches)] if one_call else batches:
            acc.accumulate(batch)
            oracle.accumulate(batch)
        assert_matches(acc.read(), oracle)
        assert acc.watermark == oracle.watermark
        return acc

    def _timestamps_do_not_matter(self, acc: dict[str, Any], batches: list[np.ndarray], one_call: bool) -> None:
        zeroed = [batch.copy() for batch in batches]
        for batch in zeroed:
            batch["t"] = 0
        for kernel in ("event_count", "polarity", "exp_decay"):
            np.testing.assert_array_equal(acc[kernel].read(), self._feed(kernel, zeroed, one_call).read())

    @pytest.mark.parametrize("one_call", [False, True])
    def test_backward_jump(self, one_call: bool) -> None:
        batches = list(BACKWARD_JUMP)
        acc = {kernel: self._feed(kernel, batches, one_call) for kernel in ALL}
        assert {a.watermark for a in acc.values()} == {10**12}
        self._timestamps_do_not_matter(acc, batches, one_call)
        assert int(acc["event_count"].read()[1, 1]) == 2
        # Per pixel: the restart is hidden where an older-clock timestamp is larger.
        surface = acc["time_surface"].read()
        assert (int(surface[1, 1]), int(surface[2, 3])) == (10**12, 7)
        # Global: every restarted-clock event is evaluated at the unchanged watermark.
        decay = acc["timestamp_decay"].read()
        assert (decay[1, 1], decay[2, 3]) == (np.float32(1.0), np.float32(0.0))

    @pytest.mark.parametrize("one_call", [False, True])
    def test_forward_spike(self, one_call: bool) -> None:
        batches = list(FORWARD_SPIKE)
        acc = {kernel: self._feed(kernel, batches, one_call) for kernel in ALL}
        assert {a.watermark for a in acc.values()} == {2**62}
        self._timestamps_do_not_matter(acc, batches, one_call)
        assert int(acc["event_count"].read()[1, 1]) == 2
        # Per pixel: only the spike's pixel holds the spike.
        surface = acc["time_surface"].read()
        assert (int(surface[1, 1]), int(surface[2, 3]), int(surface[3, 5])) == (1_010, 1_002, 2**62)
        # Global: everything but the spike, before it and after it, reads 0.
        decay = acc["timestamp_decay"].read()
        assert decay[3, 5] == np.float32(1.0)
        assert np.count_nonzero(decay) == 1

    @pytest.mark.parametrize("batches", [BACKWARD_JUMP, FORWARD_SPIKE], ids=["backward", "forward"])
    @pytest.mark.parametrize("kernel", ALL)
    def test_reset_recovers(self, kernel: str, batches: tuple[np.ndarray, ...]) -> None:
        acc = self._feed(kernel, list(batches), one_call=False)
        acc.reset()
        fresh = KERNELS[kernel].oracle()
        clean = events((20, 1, 1, 0), (25, 2, 3, 1))
        acc.accumulate(clean)
        fresh.accumulate(clean)
        assert_matches(acc.read(), fresh)
        assert acc.watermark == 25
