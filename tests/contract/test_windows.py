"""``replay.windows``: offline frames every N µs of event time, against the independent oracle."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.contract.api import EVENT_DTYPE, impl
from tests.oracle import ReferenceAccumulator, float32_ulp_distance
from tests.temporal_cases import SENSOR, WINDOW_CASES, WindowCase
from tests.temporal_oracle import TEMPORAL_KERNELS, TIMESTAMP_LIMIT, TemporalReference, windows as oracle_windows




def windows(*args: Any, **kwargs: Any) -> Any:
    """``replay.windows`` of the implementation under test, looked up on each call."""
    return importlib.import_module(f"{impl.__name__}.replay").windows(*args, **kwargs)

WIDTH, HEIGHT = SENSOR
KERNELS: dict[str, tuple[Any, dict[str, Any]]] = {
    "event_count": (lambda: impl.EventCount(), {}),
    "polarity": (lambda: impl.Polarity(), {}),
    "time_surface": (lambda: impl.TimeSurface(), {}),
    "timestamp_decay": (lambda: impl.TimestampDecay(10.0), {"tau_us": 10.0}),
    "stacked_histogram": (lambda: impl.StackedHistogram(bins=3, bin_us=7), {"bins": 3, "bin_us": 7}),
    "voxel_grid": (lambda: impl.VoxelGrid(bins=3, bin_us=7), {"bins": 3, "bin_us": 7}),
}


def array(rows: Any) -> np.ndarray:
    return np.array(list(rows), dtype=EVENT_DTYPE)


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool((a.view(np.uint8) == b.view(np.uint8)).all())


def instance(name: str, params: dict[str, int]) -> Any:
    if name == "stacked_histogram":
        return impl.StackedHistogram(**params)
    if name == "voxel_grid":
        return impl.VoxelGrid(**params)
    return KERNELS[name][0]()


def reference(name: str, params: dict[str, Any]) -> Any:
    if name in TEMPORAL_KERNELS:
        return lambda: TemporalReference(name, SENSOR, **params)
    return lambda: ReferenceAccumulator(name, SENSOR, **params)


def dense(name: str, params: dict[str, int], sparse: dict[tuple[int, ...], int]) -> np.ndarray:
    frame = reference(name, params)().read()  # zeros of the right shape and dtype
    for key, value in sparse.items():
        if name == "voxel_grid":
            j, x = key
            frame[j, 0, x] = np.float32(float(value) / float(params["bin_us"]))
        elif name == "stacked_histogram":
            channel, j, x = key
            frame[channel, j, 0, x] = value
        else:
            (x,) = key
            frame[0, x] = value
    return frame


def untouched() -> Iterator[np.ndarray]:
    raise AssertionError("the batches were read before the arguments were checked")
    yield  # pragma: no cover


@pytest.mark.parametrize("every_us", [True, np.bool_(False), 1.5, 10.0, "10", None])
def test_a_non_integer_interval_raises_type_error_on_the_call(every_us: Any) -> None:
    with pytest.raises(TypeError):
        windows(untouched(), SENSOR, impl.EventCount(), every_us=every_us)


@pytest.mark.parametrize("every_us", [0, -1, np.int64(0)])
def test_an_interval_below_1_raises_value_error_on_the_call(every_us: Any) -> None:
    with pytest.raises(ValueError):
        windows(untouched(), SENSOR, impl.EventCount(), every_us=every_us)


def test_the_interval_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        windows([], SENSOR, impl.EventCount(), 10)


def test_exp_decay_is_refused_on_the_call() -> None:
    class Subclass(impl.ExpDecay):
        pass

    for kernel in (impl.ExpDecay(0.5), Subclass(0.9)):
        with pytest.raises(TypeError, match="TimestampDecay"):
            windows(untouched(), SENSOR, kernel, every_us=10)


def test_accepts_numpy_integers_and_any_iterable() -> None:
    got = list(windows(iter([array([(3, 0, 0, 1), (12, 0, 0, 1)])]), SENSOR, impl.EventCount(),
                       every_us=np.int64(10)))
    assert [t for t, _ in got] == [10]


@pytest.mark.parametrize("case", WINDOW_CASES, ids=lambda c: c.name)
def test_hand_computed_cases(case: WindowCase) -> None:
    name, params = case.kernel
    got = list(windows([array(b) for b in case.batches], SENSOR, instance(name, params), every_us=case.every_us))
    assert [t for t, _ in got] == [t for t, _ in case.expected]
    for (_, frame), (_, sparse) in zip(got, case.expected):
        assert bit_equal(frame, dense(name, params, sparse))


def test_timestamp_decay_is_read_at_the_boundary() -> None:
    got = list(windows([array([(0, 0, 0, 1), (15, 0, 0, 1)])], SENSOR, impl.TimestampDecay(10.0), every_us=10))
    assert [t for t, _ in got] == [10]
    assert float32_ulp_distance(got[0][1][0, 0], np.float32(np.exp(-1.0))) <= 1


def test_a_forward_spike_gets_a_frame_per_boundary() -> None:
    got = list(windows([array([(3, 0, 0, 1), (1_000, 0, 0, 1)])], SENSOR, impl.EventCount(), every_us=10))
    assert [t for t, _ in got] == list(range(10, 1_001, 10))


def test_frames_are_new_writable_arrays() -> None:
    got = list(windows([array([(3, 0, 0, 1), (12, 1, 0, 1), (25, 0, 0, 0), (31, 1, 0, 1)])], SENSOR,
                       impl.StackedHistogram(bins=2, bin_us=5), every_us=10))
    frames = [frame for _, frame in got]
    assert len(frames) == 3
    for i, frame in enumerate(frames):
        assert frame.flags.writeable
        assert not any(np.shares_memory(frame, other) for other in frames[i + 1 :])


def test_a_batch_with_a_timestamp_at_2_63_yields_none_of_its_frames() -> None:
    first = array([(3, 0, 0, 1), (12, 0, 0, 1)])
    second = array([(25, 0, 0, 1), (TIMESTAMP_LIMIT, 0, 0, 1)])
    frames = []
    with pytest.raises(ValueError):
        for item in itertools.islice(windows([first, second], SENSOR, impl.EventCount(), every_us=10), 100):
            frames.append(item)
    assert [t for t, _ in frames] == [10]  # 20 would be completed by the second batch


def test_a_malformed_batch_raises_type_error_when_reached() -> None:
    bad = np.zeros(2, dtype=[("t", "<f8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])
    with pytest.raises(TypeError):
        list(windows([array([(3, 0, 0, 1)]), bad], SENSOR, impl.EventCount(), every_us=10))


@st.composite
def batched(draw: st.DrawFn) -> list[list[Any]]:
    """Events dense in t = 0 ... 80, 3 in 16 out of bounds (in x, y or both), cut into batches."""
    drawn = draw(st.lists(st.tuples(st.integers(0, 80), st.integers(0, WIDTH - 1), st.integers(0, HEIGHT - 1),
                                    st.integers(0, 255), st.integers(0, 15)), min_size=3, max_size=60))
    stream = [(t, WIDTH if where in (0, 2) else x, HEIGHT if where in (1, 2) else y, p)
              for t, x, y, p, where in drawn]
    cuts = sorted(draw(st.lists(st.integers(0, len(stream)), max_size=6)))
    return [stream[a:b] for a, b in zip([0, *cuts], [*cuts, len(stream)])]


@given(st.sampled_from(list(KERNELS)), batched(), st.integers(1, 40))
@settings(max_examples=300)
def test_matches_the_oracle(name: str, batches: list[list[Any]], every_us: int) -> None:
    make, params = KERNELS[name]
    got = list(windows([array(b) for b in batches], SENSOR, make(), every_us=every_us))
    expected = oracle_windows([array(b) for b in batches], SENSOR, reference(name, params), every_us)
    assert [t for t, _ in got] == [t for t, _ in expected]
    for (_, frame), (_, want) in zip(got, expected):
        if name == "timestamp_decay":
            assert frame.shape == want.shape and frame.dtype == want.dtype
            assert int(float32_ulp_distance(frame, want).max(initial=0)) <= 1
        else:
            assert bit_equal(frame, want)


@given(st.sampled_from(list(KERNELS)), batched(), st.integers(1, 40))
def test_batches_dont_matter(name: str, batches: list[list[Any]], every_us: int) -> None:
    make, _ = KERNELS[name]
    whole = [row for batch in batches for row in batch]
    one = list(windows([array(whole)], SENSOR, make(), every_us=every_us))
    many = list(windows([array(b) for b in batches], SENSOR, make(), every_us=every_us))
    assert [t for t, _ in many] == [t for t, _ in one]
    assert all(bit_equal(a, b) for (_, a), (_, b) in zip(many, one))
