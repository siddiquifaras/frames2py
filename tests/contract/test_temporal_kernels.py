"""The temporal kernels through ``Accumulator`` and ``Engine``, against the independent oracle.

Skipped while the implementation under test has no ``VoxelGrid`` or ``StackedHistogram``.
"""

from __future__ import annotations

import os
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.contract.api import EVENT_DTYPE, impl
from tests.contract.helpers import BACKWARD_JUMP, FORWARD_SPIKE
from tests.contract.helpers import SENSOR as WIDE_SENSOR
from tests.temporal_cases import CASES, RESET, SENSOR, Case
from tests.temporal_oracle import (
    BIN_US_LIMIT,
    TEMPORAL_KERNELS,
    TIMESTAMP_LIMIT,
    TemporalReference,
    correctly_rounded_float32,
    voxel_value,
)

pytestmark = pytest.mark.skipif(
    not (hasattr(impl, "VoxelGrid") and hasattr(impl, "StackedHistogram")),
    reason="the implementation under test has no temporal kernels",
)

SLOW = os.environ.get("FRAMES2PY_SLOW_TESTS") == "1"
WIDTH, HEIGHT = SENSOR
MIN_BINS = {"stacked_histogram": 1, "voxel_grid": 2}


def make(kernel: str, bins: Any, bin_us: Any) -> Any:
    cls = impl.StackedHistogram if kernel == "stacked_histogram" else impl.VoxelGrid
    return cls(bins=bins, bin_us=bin_us)


def array(rows: Any) -> np.ndarray:
    return np.array(list(rows), dtype=EVENT_DTYPE)


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool((a.view(np.uint8) == b.view(np.uint8)).all())


def expected_frame(case: Case) -> np.ndarray:
    ref = TemporalReference(case.kernel, SENSOR, bins=case.bins, bin_us=case.bin_us)
    frame = ref.read()  # all zeros: the oracle with nothing accumulated gives the shape and dtype
    for key, value in case.expected.items():
        if case.kernel == "stacked_histogram":
            channel, j, x = key
            frame[channel, j, 0, x] = value
        else:
            j, x = key
            frame[j, 0, x] = np.float32(float(value) / float(case.bin_us))
    return frame


# --- parameters ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
def test_output_shape_dtype_and_name(kernel: str) -> None:
    acc = impl.Accumulator((5, 3), make(kernel, 4, 10))
    frame = acc.read()
    if kernel == "stacked_histogram":
        assert (frame.shape, frame.dtype) == ((2, 4, 3, 5), np.dtype(np.uint32))
    else:
        assert (frame.shape, frame.dtype) == ((4, 3, 5), np.dtype(np.float32))
    assert make(kernel, 4, 10).name == kernel


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
@pytest.mark.parametrize(
    ("bins", "bin_us"),
    [("min", 1), ("min", BIN_US_LIMIT - 1), (np.int64(3), np.uint32(10)), (2**20, 1)],
)
def test_valid_parameters(kernel: str, bins: Any, bin_us: Any) -> None:
    make(kernel, MIN_BINS[kernel] if bins == "min" else bins, bin_us)


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
@pytest.mark.parametrize(
    ("bins", "bin_us"),
    [
        ("below", 10), (-1, 10), (3, 0), (3, -5), (3, BIN_US_LIMIT), (3, 2**63),
        (True, 10), (3, True), (np.bool_(True), 10), (3.0, 10), (3, 10.0), ("3", 10), (None, 10), (3, None),
    ],
)
def test_invalid_parameters_raise_value_error(kernel: str, bins: Any, bin_us: Any) -> None:
    with pytest.raises(ValueError):
        make(kernel, MIN_BINS[kernel] - 1 if bins == "below" else bins, bin_us)


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
def test_parameters_are_keyword_only(kernel: str) -> None:
    cls = impl.StackedHistogram if kernel == "stacked_histogram" else impl.VoxelGrid
    with pytest.raises(TypeError):
        cls(3, 10)


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
def test_no_name_lookup(kernel: str) -> None:
    with pytest.raises(ValueError):
        impl.Accumulator(SENSOR, kernel)


# --- hand-computed cases --------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_hand_computed_cases_through_the_accumulator(case: Case) -> None:
    acc = impl.Accumulator(SENSOR, make(case.kernel, case.bins, case.bin_us))
    for call in case.calls:
        if call == RESET:
            acc.reset()
        else:
            acc.accumulate(array(call))
    assert acc.watermark == case.watermark
    assert bit_equal(acc.read(), expected_frame(case))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_hand_computed_cases_through_the_engine(case: Case) -> None:
    engine = impl.Engine(SENSOR, make(case.kernel, case.bins, case.bin_us), snapshot_interval_ms=0.0)
    for call in case.calls:
        if call == RESET:
            engine.reset()
            assert engine.snapshot() is None
        else:
            engine.ingest(array(call))
    snapshot = engine.snapshot()
    assert snapshot.meta.watermark == case.watermark
    assert bit_equal(snapshot.frame, expected_frame(case))


# --- generated streams against the oracle ---------------------------------------------------

Stream = tuple[str, int, int, list[Any]]


@st.composite
def streams(draw: st.DrawFn, max_size: int = 80) -> Stream:
    """A kernel configuration and events placed around the bins its frames show.

    Events fall in bins ``base - 2 ... base + bins + 1``, near t = 0 or near 2**63, so frames
    are rarely empty. In about one stream in five, up to two more land anywhere below 2**63
    (forward spikes, late events).
    """
    kernel = draw(st.sampled_from(TEMPORAL_KERNELS))
    bins = draw(st.integers(MIN_BINS[kernel], 5))
    bin_us = draw(st.one_of(st.integers(1, 60), st.integers(1, BIN_US_LIMIT - 1)))
    last_bin = (TIMESTAMP_LIMIT - 1) // bin_us
    base = draw(st.one_of(st.integers(0, 20), st.integers(last_bin - bins - 30, last_bin - bins - 2)))
    near = draw(st.lists(st.tuples(st.integers(-2, bins + 1), st.integers(0, bin_us - 1), st.integers(0, WIDTH - 1),
                                   st.integers(0, HEIGHT - 1), st.integers(0, 255), st.integers(0, 7)),
                         min_size=4, max_size=max_size))
    far_event = st.tuples(st.integers(0, TIMESTAMP_LIMIT - 1), st.integers(0, WIDTH), st.integers(0, HEIGHT),
                          st.integers(0, 255))
    far = draw(st.lists(far_event, max_size=2)) if draw(st.integers(0, 4)) == 0 else []
    # about one event in eight is out of bounds
    rows = [((base + k) * bin_us + r, x if inside else WIDTH, y, p) for k, r, x, y, p, inside in near if base + k >= 0]
    for row in far:
        rows.insert(draw(st.integers(0, len(rows))), row)
    return kernel, bins, bin_us, rows


def split(data: st.DataObject, rows: list[Any], max_calls: int = 8) -> list[list[Any]]:
    """*rows* in another order, cut into calls, empty ones included."""
    order = data.draw(st.permutations(rows))
    cuts = sorted(data.draw(st.lists(st.integers(0, len(rows)), max_size=max_calls)))
    return [list(order[a:b]) for a, b in zip([0, *cuts], [*cuts, len(rows)])]


def accumulate(kernel: str, bins: int, bin_us: int, calls: list[list[Any]]) -> Any:
    acc = impl.Accumulator(SENSOR, make(kernel, bins, bin_us))
    for call in calls:
        acc.accumulate(array(call))
    return acc


@given(streams(), st.data())
@settings(max_examples=300)
def test_matches_the_oracle(stream: Stream, data: st.DataObject) -> None:
    kernel, bins, bin_us, rows = stream
    calls = split(data, rows)
    oracle = TemporalReference(kernel, SENSOR, bins=bins, bin_us=bin_us)
    for call in calls:
        oracle.accumulate(array(call))
    acc = accumulate(kernel, bins, bin_us, calls)
    assert acc.watermark == oracle.watermark
    assert bit_equal(acc.read(), oracle.read())


@given(streams(), st.data())
def test_order_and_partition_dont_matter(stream: Stream, data: st.DataObject) -> None:
    kernel, bins, bin_us, rows = stream
    whole = accumulate(kernel, bins, bin_us, [rows]).read()
    assert bit_equal(accumulate(kernel, bins, bin_us, split(data, rows)).read(), whole)


@given(streams(max_size=60))
def test_voxel_output_is_correctly_rounded_within_2_53(generated: Stream) -> None:
    _, bins, bin_us, stream = generated
    bins = max(bins, 2)
    oracle = TemporalReference("voxel_grid", SENSOR, bins=bins, bin_us=bin_us)
    oracle.accumulate(array(stream))
    frame = accumulate("voxel_grid", bins, bin_us, [stream]).read()
    for value, numerator in zip(frame.flat, oracle.numerators().flat):
        assert abs(numerator) <= 2**53
        assert value == correctly_rounded_float32(Fraction(int(numerator), bin_us))


@st.composite
def wide_bins(draw: st.DrawFn) -> tuple[int, int, list[Any]]:
    """bin_us beyond float32's exact integers, with events spread over the bins a frame shows."""
    bins = draw(st.integers(2, 4))
    bin_us = draw(st.integers(2**24, BIN_US_LIMIT - 1))
    offsets = st.integers(0, bin_us - 1)
    stream = draw(st.lists(st.tuples(st.integers(0, bins + 1), offsets, st.integers(0, WIDTH - 1),
                                     st.integers(0, 255)), min_size=1, max_size=40))
    return bins, bin_us, [(k * bin_us + r, x, 0, p) for k, r, x, p in stream]


@given(wide_bins())
def test_voxel_output_is_correctly_rounded_with_wide_bins(case: tuple[int, int, list[Any]]) -> None:
    bins, bin_us, stream = case
    oracle = TemporalReference("voxel_grid", SENSOR, bins=bins, bin_us=bin_us)
    oracle.accumulate(array(stream))
    frame = accumulate("voxel_grid", bins, bin_us, [stream]).read()
    for value, numerator in zip(frame.flat, oracle.numerators().flat):
        assert value == correctly_rounded_float32(Fraction(int(numerator), bin_us))


@pytest.mark.parametrize("order", ["sorted", "interleaved"])
def test_voxel_accumulation_is_exact_for_many_events_at_one_pixel(order: str) -> None:
    # bins 2, bin_us 3, T=6 from pixel 1: knots at 3 and 6, span [3, 6). 20,000 ON events at t=4
    # and 10,000 OFF at t=5: knot 0 gets 20,000 * 2 - 10,000 * 1, knot 1 gets 20,000 * 1 - 10,000 * 2,
    # exactly 0.
    on, off = (4, 0, 0, 1), (5, 0, 0, 0)
    rows = [on] * 20_000 + [off] * 10_000 if order == "sorted" else [on, on, off] * 10_000
    frame = accumulate("voxel_grid", 2, 3, [rows + [(6, 1, 0, 1)]]).read()
    assert frame[0, 0, 0] == correctly_rounded_float32(Fraction(30_000, 3))
    assert frame[1, 0, 0] == 0.0


@given(st.lists(st.tuples(st.integers(0, 2_000), st.integers(0, WIDTH - 1), st.integers(0, HEIGHT - 1),
                          st.integers(0, 255)), max_size=80),
       st.integers(2, 6))
def test_every_event_in_the_extent_counts_once(stream: list[Any], bins: int) -> None:
    # bin_us = 16: every value N / 16 is exact in float32 here, so sums are exact.
    bin_us = 16
    for kernel in TEMPORAL_KERNELS:
        acc = accumulate(kernel, bins, bin_us, [stream])
        frame = acc.read().astype(np.float64)
        if acc.watermark is None:
            assert not frame.any()
            continue
        end = (acc.watermark // bin_us) * bin_us
        extent = bins * bin_us if kernel == "stacked_histogram" else (bins - 1) * bin_us
        for x in range(WIDTH):
            mine = [r for r in stream if r[1] == x and end - extent <= r[0] < end]
            if kernel == "stacked_histogram":
                assert frame[:, :, 0, x].sum() == len(mine)
            else:
                assert frame[:, 0, x].sum() == sum(1 if r[3] else -1 for r in mine)


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
def test_polarity_for_every_p(kernel: str) -> None:
    # One event per p at t=15, pixel 0; T=40 from pixel 1, in the in-progress bin.
    stream = [(15, 0, 0, p) for p in range(256)] + [(40, 1, 0, 1)]
    if kernel == "stacked_histogram":  # bins 1, 2, 3 shown; t=15 is bin 1, j=0
        frame = accumulate(kernel, 3, 10, [stream]).read()
        assert frame[0, 0, 0, 0] == 1 and frame[1, 0, 0, 0] == 255
        assert frame.sum() == 256
    else:  # knots at 10, 20, 30, 40; t=15 gives 5 to each of the first two, signed: 255 - 1
        frame = accumulate(kernel, 4, 10, [stream]).read()
        assert frame[0, 0, 0] == np.float32(127.0) and frame[1, 0, 0] == np.float32(127.0)
        assert not frame[2:].any()
    assert not frame[..., 1].any()


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
def test_a_rejected_call_changes_nothing(kernel: str) -> None:
    acc = accumulate(kernel, 3, 10, [[(5, 0, 0, 1), (35, 1, 0, 0)]])
    before = acc.read()
    with pytest.raises(ValueError):
        acc.accumulate(array([(45, 0, 0, 1), (TIMESTAMP_LIMIT, 0, 0, 1)]))
    assert acc.watermark == 35
    assert bit_equal(acc.read(), before)


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
def test_the_last_bin_before_2_63_is_never_shown(kernel: str) -> None:
    bin_us = BIN_US_LIMIT - 1
    acc = accumulate(kernel, 2, bin_us, [[(TIMESTAMP_LIMIT - 1, 0, 0, 1), (TIMESTAMP_LIMIT - 1, 1, 0, 0)]])
    assert not acc.read().any()


@pytest.mark.parametrize("kernel", TEMPORAL_KERNELS)
@pytest.mark.parametrize("calls", [BACKWARD_JUMP, FORWARD_SPIKE], ids=["backward jump", "forward spike"])
def test_timestamp_discontinuities(kernel: str, calls: Any) -> None:
    oracle = TemporalReference(kernel, WIDE_SENSOR, bins=3, bin_us=7)
    acc = impl.Accumulator(WIDE_SENSOR, make(kernel, 3, 7))
    for call in calls:
        oracle.accumulate(call)
        acc.accumulate(call)
        assert bit_equal(acc.read(), oracle.read())


@given(streams(max_size=40), streams(max_size=40))
def test_reset_equals_a_fresh_accumulator(first: Stream, second: Stream) -> None:
    before = first[3]
    kernel, bins, bin_us, after = second
    acc = accumulate(kernel, bins, bin_us, [before])
    acc.reset()
    acc.accumulate(array(after))
    assert acc.watermark == accumulate(kernel, bins, bin_us, [after]).watermark
    assert bit_equal(acc.read(), accumulate(kernel, bins, bin_us, [after]).read())


@given(streams(max_size=60), st.data(), st.sampled_from([0.0, 16.0]))
def test_engine_snapshots_are_the_representation_at_their_watermark(
    stream: Stream, data: st.DataObject, interval_ms: float
) -> None:
    kernel, bins, bin_us, rows = stream
    calls = split(data, rows, max_calls=5)
    engine = impl.Engine(SENSOR, make(kernel, bins, bin_us), snapshot_interval_ms=interval_ms)
    oracle = TemporalReference(kernel, SENSOR, bins=bins, bin_us=bin_us)
    for call in calls:
        engine.ingest(array(call))
        oracle.accumulate(array(call))
        snapshot = engine.snapshot()
        if interval_ms == 0.0:  # every ingest publishes; running kernels: publication changes no state
            assert snapshot.meta.watermark == oracle.watermark
            assert bit_equal(snapshot.frame, oracle.read())
    engine.stop()
    snapshot = engine.snapshot()
    if snapshot is not None:
        assert bit_equal(snapshot.frame, oracle.read(at=snapshot.meta.watermark))


def _same_event(n: int, t: int) -> np.ndarray:
    batch = np.zeros(n, dtype=EVENT_DTYPE)
    batch["t"], batch["p"] = t, 1
    return batch


@pytest.mark.skipif(not SLOW, reason="accumulates 2**32 events; set FRAMES2PY_SLOW_TESTS=1")
def test_histogram_counts_wrap_modulo_2_to_the_32() -> None:
    acc = impl.Accumulator(SENSOR, make("stacked_histogram", 1, 10))
    chunk = _same_event(2**24, 5)
    for _ in range(2**8):
        acc.accumulate(chunk)
    acc.accumulate(_same_event(7, 5))
    acc.accumulate(array([(10, 1, 0, 1)]))  # T=10: bin 0 is complete
    assert int(acc.read()[1, 0, 0, 0]) == 7


@pytest.mark.skipif(not SLOW, reason="accumulates about 3.4e10 events; set FRAMES2PY_SLOW_TESTS=1")
def test_voxel_numerator_wraps_modulo_2_to_the_64() -> None:
    # Each event at t = bin_us sits on knot 1 and adds bin_us. (2**35 + 2**24) of them make
    # N = 2**63 + 2**52 - 2**35 - 2**24, past int64's maximum: the stored value wraps.
    bin_us = BIN_US_LIMIT - 1
    acc = impl.Accumulator(SENSOR, make("voxel_grid", 2, bin_us))
    chunk = _same_event(2**24, bin_us)
    chunks = 2**11 + 1
    for _ in range(chunks):
        acc.accumulate(chunk)
    acc.accumulate(array([(2 * bin_us, 1, 0, 1)]))  # T = 2 * bin_us: knots at bin_us and 2 * bin_us
    numerator = chunks * 2**24 * bin_us
    assert numerator > 2**63 - 1
    assert acc.read()[0, 0, 0] == voxel_value(numerator, bin_us) < 0


@given(streams(max_size=40), st.data())
def test_read_at_a_later_watermark_without_changing_state(generated: Stream, data: st.DataObject) -> None:
    kernel_name, bins, bin_us, stream = generated
    later = data.draw(st.one_of(st.integers(0, (bins + 2) * bin_us), st.integers(0, TIMESTAMP_LIMIT - 1)))
    batch = array(stream)
    inside = batch[(batch["x"] < WIDTH) & (batch["y"] < HEIGHT)]
    if not len(inside):
        return
    oracle = TemporalReference(kernel_name, SENSOR, bins=bins, bin_us=bin_us)
    oracle.accumulate(inside)
    kernel = make(kernel_name, bins, bin_us)
    state = kernel.init_state(SENSOR)
    kernel.begin_call(state)
    kernel.accumulate(inside, state, oracle.watermark)
    shape, dtype = kernel.output_spec(SENSOR)
    at = min(oracle.watermark + later, TIMESTAMP_LIMIT - 1)

    def read(time: int) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        kernel.read(state, out, time)
        return out

    before = read(oracle.watermark)
    assert bit_equal(read(at), oracle.read(at=at))
    assert bit_equal(read(oracle.watermark), before)
