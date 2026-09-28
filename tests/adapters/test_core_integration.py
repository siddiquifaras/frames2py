"""Real fixture -> production adapter -> Accumulator and Engine, against the reference oracle.

The oracle is fed the very batches the adapter yielded, so these tests check that what an
adapter produces goes through the core with the core's semantics: counts, polarity, time
surface, watermark, statistics and publication.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frames2py import Accumulator, Engine
from tests.adapters import evt_words as w
from tests.adapters.backends import require_backend
from tests.oracle import ReferenceAccumulator

DATA = Path(__file__).resolve().parent.parent / "data"
KERNELS = ("event_count", "polarity", "time_surface")


@dataclasses.dataclass(frozen=True)
class Fixture:
    open: Callable[[], Any]
    count: int


def _evt(name: str, count: int, **kwargs: Any) -> Fixture:
    def opener() -> Any:
        from frames2py.adapters import evt

        return evt.open(DATA / name, batch_size=20_000, **kwargs)

    return Fixture(opener, count)


def _aedat4() -> Fixture:
    require_backend("dv_processing")

    def opener() -> Any:
        from frames2py.adapters import aedat4

        return aedat4.open(DATA / "sparklers_100k.aedat4")

    return Fixture(opener, 100_000)


def _hdf5() -> Fixture:
    require_backend("h5py", "hdf5plugin")

    def opener() -> Any:
        from frames2py.adapters import hdf5

        return hdf5.open(DATA / "sparklers_100k.h5", group="events", t_offset="t_offset", sensor_size=(640, 480),
                         batch_size=30_000)

    return Fixture(opener, 100_000)


FIXTURES: dict[str, Callable[[], Fixture]] = {
    "aedat4-sparklers": _aedat4,
    "hdf5-sparklers": _hdf5,
    "evt2-sparklers": lambda: _evt("sparklers_100k.evt2.raw", 100_000, sensor_size=(640, 480)),
    "evt3-active-marker": lambda: _evt("active_marker_head.evt3.raw", 46_893),
}


@pytest.fixture(params=sorted(FIXTURES))
def batches(request: pytest.FixtureRequest) -> tuple[tuple[int, int], list[np.ndarray]]:
    fixture = FIXTURES[request.param]()
    with fixture.open() as reader:
        out = list(reader)
        size = reader.sensor_size
    assert sum(len(b) for b in out) == fixture.count
    assert size is not None
    return size, out


@pytest.mark.parametrize("kernel", KERNELS)
def test_accumulator_matches_the_oracle(batches: tuple[tuple[int, int], list[np.ndarray]], kernel: str) -> None:
    size, parts = batches
    acc, ref = Accumulator(size, kernel), ReferenceAccumulator(kernel, size)
    for batch in parts:
        acc.accumulate(batch)
        ref.accumulate(batch)
    np.testing.assert_array_equal(acc.read(), ref.read())
    assert acc.read().dtype == ref.read().dtype
    assert acc.watermark == ref.watermark == max(int(b["t"].max()) for b in parts)
    assert acc.events_out_of_bounds == ref.out_of_bounds == 0


@pytest.mark.parametrize("kernel", KERNELS)
def test_engine_publishes_every_batch_at_interval_0(batches: tuple[tuple[int, int], list[np.ndarray]], kernel: str) -> None:
    size, parts = batches
    engine, ref = Engine(size, kernel, snapshot_interval_ms=0), ReferenceAccumulator(kernel, size)
    for batch in parts:
        engine.ingest(batch)
        ref.accumulate(batch)
        snapshot = engine.snapshot()
        assert snapshot is not None
        np.testing.assert_array_equal(snapshot.frame, ref.read())
        assert snapshot.meta.watermark == ref.watermark
        ref.close_window()
    stats = engine.stats
    assert stats.events_ingested == sum(len(b) for b in parts)
    assert stats.events_out_of_bounds == 0
    assert stats.snapshots_published == len(parts)


@pytest.mark.parametrize("kernel", KERNELS)
def test_engine_stop_publishes_the_pending_window(batches: tuple[tuple[int, int], list[np.ndarray]], kernel: str) -> None:
    size, parts = batches
    engine, ref = Engine(size, kernel, snapshot_interval_ms=1e9), ReferenceAccumulator(kernel, size)
    for i, batch in enumerate(parts):
        engine.ingest(batch)
        ref.accumulate(batch)
        if i == 0:
            ref.close_window()  # the first ingest() publishes
    engine.stop()
    snapshot = engine.snapshot()
    assert len(parts) > 1 and snapshot is not None
    assert snapshot.meta.sequence == 2 and engine.stats.snapshots_published == 2
    np.testing.assert_array_equal(snapshot.frame, ref.read())
    assert snapshot.meta.watermark == ref.watermark


def test_rows_beyond_the_height_reach_the_core_as_out_of_bounds(tmp_path: Path) -> None:
    from frames2py.adapters import evt

    # The out-of-bounds events carry the latest timestamps, so counting them would move the watermark.
    words = [w.evt3_time_high(1), w.evt3_y(3), w.evt3_x(10, on=True), w.evt3_x(11, on=False),
             w.evt3_time_high(2), w.evt3_time_low(5), w.evt3_y(479), w.evt3_x(639, on=True),
             w.evt3_time_high(3), w.evt3_y(480), w.evt3_x(12, on=True), w.evt3_base(20, on=True), w.evt3_vect8(0b111)]
    path = w.write_raw(tmp_path / "oob.raw", words, "3.0", geometry=(640, 480))
    with evt.open(path) as reader:
        parts = list(reader)
    decoded = np.concatenate(parts)
    assert int((decoded["y"] >= 480).sum()) == 4
    engine = Engine((640, 480), "event_count", snapshot_interval_ms=0)
    for batch in parts:
        engine.ingest(batch)
    assert engine.stats.events_ingested == 7
    assert engine.stats.events_out_of_bounds == 4
    snapshot = engine.snapshot()
    assert snapshot is not None
    assert snapshot.meta.watermark == (2 << 12) | 5
    ref = ReferenceAccumulator("event_count", (640, 480))
    ref.accumulate(decoded)
    np.testing.assert_array_equal(snapshot.frame, ref.read())
