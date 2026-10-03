"""The temporal-kernel oracle against hand-computed cases and properties of the definitions.

Expected values come from ``tests/temporal_cases.py``, worked out by hand, or from
properties stated independently of the oracle's code (for example, which events lie in a
frame's span).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frames2py import EVENT_DTYPE

from tests.oracle import ReferenceAccumulator
from tests.temporal_cases import CASES, RESET, SENSOR, WINDOW_CASES, Case, WindowCase
from tests.temporal_oracle import (
    BIN_US_LIMIT,
    TEMPORAL_KERNELS,
    TemporalReference,
    correctly_rounded_float32,
    signed_reading,
    voxel_value,
    windows,
)

WIDTH, HEIGHT = SENSOR


def array(rows: Any) -> np.ndarray:
    return np.array(list(rows), dtype=EVENT_DTYPE)


def dense(kernel: str, bins: int, bin_us: int, sparse: dict[tuple[int, ...], int]) -> np.ndarray:
    """A sparse hand-computed frame as the kernel's observable output."""
    if kernel == "stacked_histogram":
        frame = np.zeros((2, bins, HEIGHT, WIDTH), dtype=np.uint32)
        for (channel, j, x), count in sparse.items():
            frame[channel, j, 0, x] = count
        return frame
    if kernel == "voxel_grid":
        frame = np.zeros((bins, HEIGHT, WIDTH), dtype=np.float32)
        for (j, x), numerator in sparse.items():
            frame[j, 0, x] = voxel_value(numerator, bin_us)
        return frame
    assert kernel == "event_count"
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    for (x,), count in sparse.items():
        frame[0, x] = count
    return frame


def run_case(case: Case) -> TemporalReference:
    ref = TemporalReference(case.kernel, SENSOR, bins=case.bins, bin_us=case.bin_us)
    for call in case.calls:
        if call == RESET:
            ref.reset()
        else:
            ref.accumulate(array(call))
    return ref


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool((a.view(np.uint8) == b.view(np.uint8)).all())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_hand_computed_cases(case: Case) -> None:
    ref = run_case(case)
    assert ref.watermark == case.watermark
    assert bit_equal(ref.read(), dense(case.kernel, case.bins, case.bin_us, case.expected))


def make_reference(name: str, params: dict[str, int]) -> Any:
    if name in TEMPORAL_KERNELS:
        return lambda: TemporalReference(name, SENSOR, **params)
    return lambda: ReferenceAccumulator(name, SENSOR, **params)


@pytest.mark.parametrize("case", WINDOW_CASES, ids=lambda c: c.name)
def test_hand_computed_window_cases(case: WindowCase) -> None:
    name, params = case.kernel
    got = windows([array(b) for b in case.batches], SENSOR, make_reference(name, params), case.every_us)
    assert [t for t, _ in got] == [t for t, _ in case.expected]
    bins, bin_us = params.get("bins", 0), params.get("bin_us", 1)
    for (_, frame), (_, sparse) in zip(got, case.expected):
        assert bit_equal(frame, dense(name, bins, bin_us, sparse))


def test_windows_reads_timestamp_decay_at_the_boundary() -> None:
    got = windows([array([(0, 0, 0, 1), (15, 0, 0, 1)])], SENSOR,
                  make_reference("timestamp_decay", {"tau_us": 10.0}), 10)
    assert [t for t, _ in got] == [10]
    assert got[0][1][0, 0] == np.float32(math.exp(-1.0))  # (10 - 0) / tau, not (0 - 0) / tau


def test_windows_rejects_exp_decay() -> None:
    with pytest.raises(TypeError):
        windows([array([(3, 0, 0, 1), (12, 0, 0, 1)])], SENSOR, make_reference("exp_decay", {"decay": 0.5}), 10)


def test_windows_rejects_a_batch_before_accumulating_any_of_it() -> None:
    batches = [array([(3, 0, 0, 1)]), array([(25, 0, 0, 1), (2**63, 0, 0, 1)])]
    with pytest.raises(ValueError):
        windows(batches, SENSOR, make_reference("event_count", {}), 10)


# --- the numerator's signed reading and the float32 output ----------------------------------


@pytest.mark.parametrize(
    ("numerator", "reading"),
    [(0, 0), (5, 5), (-5, -5), (2**63 - 1, 2**63 - 1), (-(2**63), -(2**63)),
     (2**63, -(2**63)), (2**64 - 1, -1), (2**64 + 7, 7), (-(2**63) - 1, 2**63 - 1)],
)
def test_signed_reading_wraps_modulo_2_64(numerator: int, reading: int) -> None:
    assert signed_reading(numerator) == reading


def test_voxel_value_uses_the_signed_reading() -> None:
    assert voxel_value(2**64 + 30, 10) == np.float32(3.0)
    assert voxel_value(2**63, 1) == np.float32(-(2.0**63))


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-3e38, max_value=3e38)
       .filter(lambda v: v == 0 or abs(v) >= 2.0**-126))
def test_correct_rounding_agrees_with_a_single_rounding_of_a_double(value: float) -> None:
    # A float64 is an exact rational; NumPy rounds it to float32 once, ties to even.
    assert correctly_rounded_float32(Fraction(value)) == np.float32(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(Fraction(2**24 + 1), 2.0**24), (Fraction(2**24 + 3), 2.0**24 + 4), (Fraction(-(2**24) - 1), -(2.0**24)),
     (Fraction(1, 3), float(np.float32(1 / 3))), (Fraction(2**25 + 2), 2.0**25)],
)
def test_correct_rounding_breaks_ties_to_even(value: Fraction, expected: float) -> None:
    assert correctly_rounded_float32(value) == np.float32(expected)


numerators = st.integers(-(2**53), 2**53)
bin_widths = st.integers(1, BIN_US_LIMIT - 1)


@given(numerators, bin_widths)
@settings(max_examples=2000)
def test_output_is_correctly_rounded_while_the_numerator_is_within_2_53(numerator: int, bin_us: int) -> None:
    assert voxel_value(numerator, bin_us) == correctly_rounded_float32(Fraction(numerator, bin_us))


@pytest.mark.parametrize("seed", range(4))
def test_output_is_correctly_rounded_next_to_float32_midpoints(seed: int) -> None:
    # Numerators within 1 of a float32 midpoint times bin_us: where double rounding would show.
    rng = np.random.default_rng(seed)
    for _ in range(3000):
        bin_us = int(rng.integers(1, BIN_US_LIMIT))
        mantissa = int(rng.integers(2**23, 2**24))
        exponent = int(rng.integers(-20, 30))
        midpoint = Fraction(2 * mantissa + 1, 2**24) * Fraction(2) ** exponent
        target = math.floor(midpoint * bin_us)
        for numerator in (target - 1, target, target + 1):
            if 0 < numerator <= 2**53:
                assert voxel_value(numerator, bin_us) == correctly_rounded_float32(Fraction(numerator, bin_us))


# --- properties of the definitions ----------------------------------------------------------

@st.composite
def streams(draw: st.DrawFn) -> tuple[str, int, int, list[tuple[int, int, int, int]]]:
    """Events in bins 0 ... bins + 2, so frames are rarely empty; 3 in 16 out of bounds (in x, y or both)."""
    kernel = draw(st.sampled_from(TEMPORAL_KERNELS))
    bins = draw(st.integers(2 if kernel == "voxel_grid" else 1, 6))
    bin_us = draw(st.integers(1, 40))
    drawn = draw(st.lists(st.tuples(st.integers(0, bins + 2), st.integers(0, bin_us - 1), st.integers(0, WIDTH - 1),
                                    st.integers(0, HEIGHT - 1), st.integers(0, 255), st.integers(0, 15)),
                          min_size=3, max_size=60))
    rows = [(k * bin_us + r, WIDTH if where in (0, 2) else x, HEIGHT if where in (1, 2) else y, p)
            for k, r, x, y, p, where in drawn]
    return kernel, bins, bin_us, rows


def reference(kernel: str, bins: int, bin_us: int, calls: list[list[Any]]) -> TemporalReference:
    ref = TemporalReference(kernel, SENSOR, bins=bins, bin_us=bin_us)
    for call in calls:
        ref.accumulate(array(call))
    return ref


@given(streams(), st.data())
def test_order_and_partition_dont_matter(stream: Any, data: st.DataObject) -> None:
    kernel, bins, bin_us, rows = stream
    order = data.draw(st.permutations(rows))
    cuts = sorted(data.draw(st.lists(st.integers(0, len(rows)), max_size=6)))
    calls = [list(order[a:b]) for a, b in zip([0, *cuts], [*cuts, len(rows)])]
    assert bit_equal(reference(kernel, bins, bin_us, calls).read(), reference(kernel, bins, bin_us, [rows]).read())


def in_frame(t: int, kernel: str, bins: int, bin_us: int, watermark: int) -> bool:
    """Whether an event at t lies in the frame's temporal extent, from the extents alone."""
    end = (watermark // bin_us) * bin_us
    extent = bins * bin_us if kernel == "stacked_histogram" else (bins - 1) * bin_us
    return end - extent <= t < end


@given(streams())
def test_every_event_in_the_extent_counts_once_and_no_other(stream: Any) -> None:
    kernel, bins, bin_us, rows = stream
    ref = reference(kernel, bins, bin_us, [rows])
    inside = [r for r in rows if r[1] < WIDTH and r[2] < HEIGHT]
    for x in range(WIDTH):
        mine = [r for r in inside if r[1] == x and ref.watermark is not None
                and in_frame(r[0], kernel, bins, bin_us, ref.watermark)]
        if kernel == "stacked_histogram":
            assert int(ref.counts()[:, :, 0, x].sum()) == len(mine)
        else:
            signed = sum(1 if r[3] else -1 for r in mine)
            assert int(ref.numerators()[:, 0, x].sum()) == bin_us * signed


@given(streams(), st.data())
def test_the_in_progress_bin_never_shows(stream: Any, data: st.DataObject) -> None:
    kernel, bins, bin_us, rows = stream
    ref = reference(kernel, bins, bin_us, [rows])
    if ref.watermark is None:
        return
    start = (ref.watermark // bin_us) * bin_us
    extra = data.draw(st.lists(st.tuples(st.integers(start, ref.watermark), st.integers(0, WIDTH - 1),
                                         st.integers(0, HEIGHT - 1), st.integers(0, 255)), max_size=10))
    before = ref.read()
    ref.accumulate(array(extra))
    assert bit_equal(ref.read(), before)


@given(streams())
def test_out_of_bounds_events_change_nothing(stream: Any) -> None:
    kernel, bins, bin_us, rows = stream
    inside = [r for r in rows if r[1] < WIDTH and r[2] < HEIGHT]
    assert bit_equal(reference(kernel, bins, bin_us, [rows]).read(), reference(kernel, bins, bin_us, [inside]).read())


@given(streams(), streams())
def test_reset_equals_a_fresh_reference(first: Any, second: Any) -> None:
    kernel, bins, bin_us, rows = second
    ref = reference(kernel, bins, bin_us, [first[3]])
    ref.reset()
    ref.accumulate(array(rows))
    assert bit_equal(ref.read(), reference(kernel, bins, bin_us, [rows]).read())


@given(streams(), st.integers(1, 50), st.data())
def test_windows_dont_depend_on_batches(stream: Any, every_us: int, data: st.DataObject) -> None:
    kernel, bins, bin_us, rows = stream
    cuts = sorted(data.draw(st.lists(st.integers(0, len(rows)), max_size=6)))
    batches = [array(rows[a:b]) for a, b in zip([0, *cuts], [*cuts, len(rows)])]
    make = make_reference(kernel, {"bins": bins, "bin_us": bin_us})
    one = windows([array(rows)], SENSOR, make, every_us)
    many = windows(batches, SENSOR, make, every_us)
    assert [t for t, _ in many] == [t for t, _ in one]
    assert all(bit_equal(a, b) for (_, a), (_, b) in zip(many, one))
