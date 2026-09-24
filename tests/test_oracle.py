"""The reference oracle against hand-computed cases and contract properties.

Expected values here are worked out by hand from each kernel's definition, not
from the oracle's own code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frames2py import EVENT_DTYPE

from tests.oracle import (
    KERNELS,
    ReferenceAccumulator,
    counts_to_uint32,
    float32_ulp_distance,
)

SENSOR = (4, 3)  # width 4, height 3: frames are (3, 4)
PARAMS = {"exp_decay": {"decay": 0.5}, "timestamp_decay": {"tau_us": 10.0}}
OUTPUT = {
    "event_count": ((3, 4), np.uint32),
    "polarity": ((3, 4, 2), np.uint32),
    "time_surface": ((3, 4), np.uint64),
    "exp_decay": ((3, 4), np.float32),
    "timestamp_decay": ((3, 4), np.float32),
}


def events(*rows: tuple[int, int, int, int]) -> np.ndarray:
    return np.array(list(rows), dtype=EVENT_DTYPE)


def make(kernel: str, sensor: tuple[int, int] = SENSOR) -> ReferenceAccumulator:
    return ReferenceAccumulator(kernel, sensor, **PARAMS.get(kernel, {}))


def random_events(seed: int, n: int, sensor: tuple[int, int] = SENSOR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=EVENT_DTYPE)
    out["t"] = rng.integers(0, 200, size=n)
    out["x"] = rng.integers(0, sensor[0] + 1, size=n)  # some out of bounds
    out["y"] = rng.integers(0, sensor[1] + 1, size=n)
    out["p"] = rng.integers(0, 256, size=n)
    return out


@pytest.mark.parametrize("kernel", KERNELS)
def test_output_shape_and_dtype(kernel: str) -> None:
    acc = make(kernel)
    shape, dtype = OUTPUT[kernel]
    empty = acc.read()
    acc.accumulate(events((5, 1, 2, 1)))
    for frame in (empty, acc.read()):
        assert frame.shape == shape
        assert frame.dtype == dtype


class TestEventCount:
    def test_counts_per_pixel_with_x_as_column(self) -> None:
        acc = make("event_count")
        acc.accumulate(events((0, 3, 0, 0), (1, 3, 0, 1), (2, 0, 2, 0)))
        expected = np.zeros((3, 4), dtype=np.uint32)
        expected[0, 3] = 2
        expected[2, 0] = 1
        np.testing.assert_array_equal(acc.read(), expected)

    def test_out_of_bounds_counted_not_accumulated(self) -> None:
        acc = make("event_count")
        acc.accumulate(events((0, 4, 0, 0), (1, 0, 3, 0), (2, 1, 1, 0)))
        assert acc.out_of_bounds == 2
        assert acc.accumulated == 1
        assert int(acc.read().sum()) == 1

    def test_closing_the_window_starts_from_zero(self) -> None:
        acc = make("event_count")
        acc.accumulate(events((0, 1, 1, 0)))
        acc.close_window()
        assert int(acc.read().sum()) == 0
        acc.accumulate(events((1, 1, 1, 0)))
        assert int(acc.read()[1, 1]) == 1

    def test_read_is_repeatable(self) -> None:
        acc = make("event_count")
        acc.accumulate(events((0, 1, 1, 0)))
        np.testing.assert_array_equal(acc.read(), acc.read())


class TestPolarity:
    @pytest.mark.parametrize("p", range(256))
    def test_every_p_value_lands_on_its_own_pixel(self, p: int) -> None:
        acc = make("polarity")
        acc.accumulate(events((0, 2, 1, p)))
        frame = acc.read()
        expected = np.zeros((3, 4, 2), dtype=np.uint32)
        expected[1, 2, 0 if p == 0 else 1] = 1
        np.testing.assert_array_equal(frame, expected)

    def test_windowed(self) -> None:
        acc = make("polarity")
        acc.accumulate(events((0, 0, 0, 1)))
        acc.close_window()
        assert int(acc.read().sum()) == 0


def test_uint32_counts_wrap_instead_of_saturating() -> None:
    exact = np.array([0, 2**32 - 1, 2**32, 2**32 + 7, 3 * 2**32 + 1], dtype=object)
    np.testing.assert_array_equal(
        counts_to_uint32(exact), np.array([0, 2**32 - 1, 0, 7, 1], dtype=np.uint32)
    )


class TestTimeSurface:
    def test_keeps_the_maximum_not_the_last_timestamp(self) -> None:
        acc = make("time_surface")
        acc.accumulate(events((100, 1, 1, 0), (5, 1, 1, 0)))
        assert int(acc.read()[1, 1]) == 100
        assert acc.watermark == 100

    def test_large_timestamps_stay_exact(self) -> None:
        acc = make("time_surface")
        t = 2**63 - 1
        acc.accumulate(events((t, 0, 0, 0), (20_000_001, 1, 0, 0)))
        assert int(acc.read()[0, 0]) == t
        assert int(acc.read()[0, 1]) == 20_000_001

    def test_event_at_t0_looks_like_no_event(self) -> None:
        acc = make("time_surface")
        acc.accumulate(events((0, 1, 1, 0)))
        assert int(acc.read()[1, 1]) == 0

    def test_running_across_window_close(self) -> None:
        acc = make("time_surface")
        acc.accumulate(events((7, 1, 1, 0)))
        acc.close_window()
        assert int(acc.read()[1, 1]) == 7


class TestExpDecay:
    def test_decay_once_per_call_then_add_one_per_event(self) -> None:
        acc = make("exp_decay")  # decay 0.5
        acc.accumulate(events((0, 1, 1, 0)))
        assert acc.read_exact()[1, 1] == 1.0
        acc.accumulate(events((1, 1, 1, 0), (2, 1, 1, 0)))
        assert acc.read_exact()[1, 1] == 0.5 + 2.0
        acc.accumulate(events())
        assert acc.read_exact()[1, 1] == 1.25

    def test_a_call_with_only_out_of_bounds_events_still_decays(self) -> None:
        acc = make("exp_decay")
        acc.accumulate(events((0, 1, 1, 0)))
        acc.accumulate(events((1, 9, 9, 0)))
        assert acc.read_exact()[1, 1] == 0.5

    def test_result_depends_on_call_boundaries(self) -> None:
        batch = events((0, 1, 1, 0), (1, 1, 1, 0))
        whole = make("exp_decay")
        whole.accumulate(batch)
        split = make("exp_decay")
        split.accumulate(batch[:1])
        split.accumulate(batch[1:])
        assert whole.read_exact()[1, 1] == 2.0
        assert split.read_exact()[1, 1] == 1.5

    def test_running_across_window_close(self) -> None:
        acc = make("exp_decay")
        acc.accumulate(events((0, 1, 1, 0)))
        acc.close_window()
        assert acc.read_exact()[1, 1] == 1.0


class TestTimestampDecay:
    def test_sum_evaluated_at_the_watermark(self) -> None:
        acc = make("timestamp_decay")  # tau 10 us
        acc.accumulate(events((0, 1, 1, 0), (10, 1, 1, 1)))
        assert acc.watermark == 10
        assert acc.read_exact()[1, 1] == pytest.approx(1 + math.exp(-1), rel=1e-15)

    def test_weight_is_one_per_event_whatever_the_polarity(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((10, 1, 1, 0), (10, 1, 1, 255)))
        assert acc.read_exact()[1, 1] == 2.0

    def test_a_newer_event_elsewhere_decays_every_pixel(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((0, 1, 1, 0)))
        acc.accumulate(events((20, 3, 2, 0)))
        assert acc.read_exact()[1, 1] == pytest.approx(math.exp(-2), rel=1e-15)
        assert acc.read_exact()[2, 3] == 1.0

    def test_out_of_bounds_events_do_not_move_the_watermark(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((10, 1, 1, 0)))
        before = acc.read_exact()
        acc.accumulate(events((1_000, 4, 0, 0), (2_000, 0, 3, 1)))
        assert acc.watermark == 10
        assert acc.out_of_bounds == 2
        np.testing.assert_array_equal(acc.read_exact(), before)

    def test_idle_reads_do_not_change(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((10, 1, 1, 0)))
        first = acc.read_exact()
        acc.accumulate(events())
        acc.close_window()
        np.testing.assert_array_equal(acc.read_exact(), first)

    def test_out_of_order_arrival_uses_the_maximum(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((30, 1, 1, 0), (10, 1, 1, 0)))
        assert acc.watermark == 30
        assert acc.read_exact()[1, 1] == pytest.approx(1 + math.exp(-2), rel=1e-15)

    def test_differences_are_exact_near_2_to_the_53(self) -> None:
        # As float64, 2**53 + 1 rounds to 2**53, so a float-first difference is 0.
        acc = ReferenceAccumulator("timestamp_decay", SENSOR, tau_us=1.0)
        acc.accumulate(events((2**53, 1, 1, 0), (2**53 + 1, 2, 1, 0)))
        assert acc.read_exact()[1, 1] == math.exp(-1)
        assert acc.read_exact()[1, 2] == 1.0

    def test_extreme_timestamp_underflows_earlier_contributions(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((0, 1, 1, 0)))
        acc.accumulate(events((2**63 - 1, 2, 2, 0)))
        assert acc.read_exact()[1, 1] == 0.0
        acc.accumulate(events((100, 1, 1, 0)))
        assert acc.read_exact()[1, 1] == 0.0
        assert acc.read_exact()[2, 2] == 1.0

    @pytest.mark.parametrize("tau", [None, 0.0, -1.0, math.inf, math.nan])
    def test_tau_must_be_finite_and_positive(self, tau: float | None) -> None:
        with pytest.raises(ValueError):
            ReferenceAccumulator("timestamp_decay", SENSOR, tau_us=tau)

    def test_float32_output_is_the_rounded_exact_value(self) -> None:
        acc = make("timestamp_decay")
        acc.accumulate(events((0, 1, 1, 0), (7, 1, 1, 0)))
        exact = 1 + math.exp(-0.7)
        assert acc.read()[1, 1] == np.float32(exact)


class TestInvariance:
    @pytest.mark.parametrize("kernel", ["event_count", "polarity", "time_surface", "timestamp_decay"])
    @pytest.mark.parametrize("seed", range(20))
    def test_arrival_order_and_call_partition_do_not_matter(self, kernel: str, seed: int) -> None:
        batch = random_events(seed, 60)
        rng = np.random.default_rng(1_000 + seed)
        reference = make(kernel)
        reference.accumulate(batch)
        shuffled = batch[rng.permutation(len(batch))]
        cuts = np.sort(rng.choice(np.arange(1, len(batch)), size=5, replace=False))
        other = make(kernel)
        for part in np.split(shuffled, cuts):
            other.accumulate(np.ascontiguousarray(part))
        np.testing.assert_array_equal(other.read(), reference.read())
        assert other.watermark == reference.watermark
        assert (other.accumulated, other.out_of_bounds) == (reference.accumulated, reference.out_of_bounds)


@pytest.mark.parametrize("kernel", KERNELS)
class TestAccumulatorRules:
    def test_timestamp_range_rejects_the_whole_call_unchanged(self, kernel: str) -> None:
        acc = make(kernel)
        acc.accumulate(events((5, 1, 1, 1)))
        before = (acc.read(), acc.watermark, acc.accumulated, acc.out_of_bounds)
        # The offending event is out of bounds: the check still covers it.
        bad = events((6, 2, 2, 0), (2**63, 99, 99, 0))
        with pytest.raises(ValueError):
            acc.accumulate(bad)
        np.testing.assert_array_equal(acc.read(), before[0])
        assert (acc.watermark, acc.accumulated, acc.out_of_bounds) == before[1:]
        if kernel == "exp_decay":
            acc.accumulate(events())
            assert acc.read_exact()[1, 1] == 0.5  # the rejected call applied no decay

    def test_only_out_of_bounds_events_leave_the_watermark(self, kernel: str) -> None:
        acc = make(kernel)
        acc.accumulate(events((5, 1, 1, 1)))
        acc.accumulate(events((50, 4, 1, 0), (60, 1, 3, 0)))
        assert acc.watermark == 5
        assert acc.out_of_bounds == 2

    def test_watermark_is_none_until_an_in_bounds_event(self, kernel: str) -> None:
        acc = make(kernel)
        assert acc.watermark is None
        acc.accumulate(events((9, 7, 7, 0)))
        assert acc.watermark is None

    def test_reset_clears_everything(self, kernel: str) -> None:
        acc = make(kernel)
        acc.accumulate(events((5, 1, 1, 1), (6, 9, 9, 0)))
        acc.reset()
        fresh = make(kernel)
        np.testing.assert_array_equal(acc.read(), fresh.read())
        assert (acc.watermark, acc.accumulated, acc.out_of_bounds) == (None, 0, 0)
        if kernel == "exp_decay":
            acc.accumulate(events((7, 1, 1, 0)))
            assert acc.read_exact()[1, 1] == 1.0  # no decay steps survive reset

    def test_extra_fields_are_ignored(self, kernel: str) -> None:
        wide = np.dtype(EVENT_DTYPE.descr + [("extra", "<f8")])
        batch = np.zeros(2, dtype=wide)
        batch[["t", "x", "y", "p"]] = events((3, 1, 1, 0), (4, 2, 0, 1))
        batch["extra"] = 123.0
        plain = make(kernel)
        plain.accumulate(events((3, 1, 1, 0), (4, 2, 0, 1)))
        extended = make(kernel)
        extended.accumulate(batch)
        np.testing.assert_array_equal(extended.read(), plain.read())


def test_ulp_distance() -> None:
    one = np.float32(1.0)
    up = np.nextafter(one, np.float32(2.0))
    down = np.nextafter(one, np.float32(0.0))
    assert float32_ulp_distance(np.array([one]), np.array([one]))[0] == 0
    assert float32_ulp_distance(np.array([up]), np.array([one]))[0] == 1
    assert float32_ulp_distance(np.array([down]), np.array([up]))[0] == 2
    assert float32_ulp_distance(np.array([np.float32(-0.0)]), np.array([np.float32(0.0)]))[0] == 0
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    assert float32_ulp_distance(np.array([-tiny]), np.array([tiny]))[0] == 2
