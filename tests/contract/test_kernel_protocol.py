"""The public Kernel protocol: a kernel written outside the package works with
``Accumulator`` and ``Engine``, and receives the calls the protocol defines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.contract.api import impl
from tests.contract.helpers import ALL, KERNELS, SENSOR, assert_matches, events, random_events


class Recording:
    """A last-timestamp-per-pixel kernel that records every protocol call it receives."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def output_spec(self, sensor_size: tuple[int, int]) -> tuple[tuple[int, ...], np.dtype[Any]]:
        width, height = sensor_size
        return (height, width), np.dtype(np.int64)

    def init_state(self, sensor_size: tuple[int, int]) -> dict[str, Any]:
        width, height = sensor_size
        return {"surface": np.full((height, width), -1, dtype=np.int64)}

    def begin_call(self, state: dict[str, Any]) -> None:
        self.calls.append(("begin_call",))

    def accumulate(self, events: np.ndarray, state: dict[str, Any], watermark: int | None) -> None:
        self.calls.append(("accumulate", events["t"].tolist(), watermark))
        state["surface"][events["y"], events["x"]] = events["t"].astype(np.int64)

    def read(self, state: dict[str, Any], out: np.ndarray, watermark: int | None) -> None:
        self.calls.append(("read", watermark))
        out[...] = state["surface"]

    def close_window(self, state: dict[str, Any]) -> None:
        self.calls.append(("close_window",))
        state["surface"][...] = -1

    def reset(self, state: dict[str, Any]) -> None:
        self.calls.append(("reset",))
        state["surface"][...] = -1

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture
def kernel() -> Recording:
    return Recording()


def test_public_submodule_paths() -> None:
    import importlib

    kernels = importlib.import_module(f"{impl.__name__}.kernels")
    publish = importlib.import_module(f"{impl.__name__}.publish")
    assert set(kernels.__all__) == {"Kernel", "EventCount", "Polarity", "TimeSurface", "ExpDecay", "TimestampDecay"}
    assert hasattr(publish, "SnapshotPublisher")
    assert hasattr(publish, "ImmutablePublisher")


def test_accumulator_uses_a_custom_kernel(kernel: Recording) -> None:
    acc = impl.Accumulator(SENSOR, kernel)
    acc.accumulate(events((7, 1, 2, 0)))
    frame = acc.read()
    assert frame.shape == (4, 6) and frame.dtype == np.int64
    assert frame[2, 1] == 7
    assert (frame == -1).sum() == frame.size - 1


def test_begin_call_once_per_accepted_call_before_accumulate(kernel: Recording) -> None:
    acc = impl.Accumulator(SENSOR, kernel)
    acc.accumulate(events((5, 1, 1, 1), (6, 2, 2, 0)))
    acc.accumulate(events())
    acc.accumulate(events((9, SENSOR[0], 0, 0)))  # out of bounds only
    with pytest.raises(ValueError):
        acc.accumulate(events((2**63, 1, 1, 0)))
    with pytest.raises(TypeError):
        acc.accumulate(np.zeros(2, dtype=[("t", "<f8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")]))
    names = [n for n in kernel.names() if n in ("begin_call", "accumulate")]
    assert names.count("begin_call") == 3
    assert names[0] == "begin_call" and names[1] == "accumulate"


def test_accumulate_gets_only_in_bounds_events_and_the_new_watermark(kernel: Recording) -> None:
    acc = impl.Accumulator(SENSOR, kernel)
    acc.accumulate(events((5, 1, 1, 1), (50, SENSOR[0], 0, 0), (9, 0, SENSOR[1], 1), (8, 3, 2, 0)))
    acc.accumulate(events((6, 2, 2, 0)))
    accumulated = [call for call in kernel.calls if call[0] == "accumulate"]
    assert accumulated == [("accumulate", [5, 8], 8), ("accumulate", [6], 8)]


def test_read_gets_the_watermark(kernel: Recording) -> None:
    acc = impl.Accumulator(SENSOR, kernel)
    acc.read()
    acc.accumulate(events((12, 1, 1, 0)))
    acc.read()
    assert [call for call in kernel.calls if call[0] == "read"] == [("read", None), ("read", 12)]


def test_reset_resets_the_kernel(kernel: Recording) -> None:
    acc = impl.Accumulator(SENSOR, kernel)
    acc.accumulate(events((3, 1, 1, 0)))
    acc.reset()
    assert kernel.names()[-1] == "reset"
    assert (acc.read() == -1).all()


def test_engine_publishes_by_reading_then_closing_the_window(kernel: Recording) -> None:
    engine = impl.Engine(SENSOR, kernel, snapshot_interval_ms=0.0)
    engine.ingest(events((4, 2, 1, 1)))
    names = kernel.names()
    assert names[-2:] == ["read", "close_window"]
    snap = engine.snapshot()
    frame, meta = snap.frame, snap.meta
    assert frame[1, 2] == 4  # read before the window closed
    assert meta.watermark == 4


def test_stopped_engine_makes_no_kernel_calls(kernel: Recording) -> None:
    engine = impl.Engine(SENSOR, kernel, snapshot_interval_ms=0.0)
    engine.stop()
    before = len(kernel.calls)
    engine.ingest(events((4, 2, 1, 1)))
    assert len(kernel.calls) == before


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("later", [0, 1, 37, 10**6, 2**63 - 1], ids=lambda d: f"+{d}")
def test_builtin_kernels_read_at_a_later_watermark_without_changing_state(name: str, later: int) -> None:
    case = KERNELS[name]
    kernel = case.make()
    state = kernel.init_state(SENSOR)
    batch = random_events(7, 200, t_max=5_000, out_of_bounds=False)
    oracle = case.oracle()
    oracle.accumulate(batch)
    watermark = int(batch["t"].max())
    kernel.begin_call(state)
    kernel.accumulate(batch, state, watermark)
    shape, dtype = kernel.output_spec(SENSOR)
    at = min(watermark + later, 2**63 - 1)

    def read(time: int) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        kernel.read(state, out, time)
        return out

    before = read(watermark)
    assert_matches(read(at), oracle, at)
    np.testing.assert_array_equal(read(watermark), before)  # reading later changed nothing
