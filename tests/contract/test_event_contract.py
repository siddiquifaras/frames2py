"""The event contract, through both entry points: ``Accumulator.accumulate`` and a
running ``Engine.ingest``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tests.contract.api import EVENT_DTYPE, accumulated
from tests.contract.helpers import ALL, KERNELS, SENSOR, assert_matches, events, random_events


@dataclass
class Entry:
    """One way into the core, with enough observation to prove nothing changed."""

    feed: Callable[[Any], None]
    observe: Callable[[], tuple[Any, ...]]


def _accumulator_entry(kernel: str) -> Entry:
    acc = KERNELS[kernel].accumulator()
    return Entry(
        feed=acc.accumulate,
        observe=lambda: (acc.read().tobytes(), acc.watermark, acc.events_out_of_bounds),
    )


def _engine_entry(kernel: str) -> Entry:
    engine = KERNELS[kernel].engine(interval_ms=0.0)

    def observe() -> tuple[Any, ...]:
        stats = engine.stats
        snap = engine.snapshot()
        published = None if snap is None else (snap.frame.tobytes(), snap.meta.sequence)
        return (stats.events_ingested, stats.events_out_of_bounds, stats.snapshots_published, published)

    return Entry(feed=engine.ingest, observe=observe)


ENTRIES = {"accumulator": _accumulator_entry, "engine": _engine_entry}


@pytest.fixture(params=list(ENTRIES))
def entry_kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _primed(entry_kind: str, kernel: str) -> Entry:
    entry = ENTRIES[entry_kind](kernel)
    entry.feed(events((5, 1, 1, 1), (7, 2, 3, 0)))
    return entry


FIELDS = ("t", "x", "y", "p")


def _with_field(name: str, dtype: str) -> np.ndarray:
    fields = [(n, dtype if n == name else EVENT_DTYPE[n]) for n in FIELDS]
    return np.zeros(3, dtype=fields)


STRUCTURALLY_INVALID: dict[str, Callable[[], Any]] = {
    "list": lambda: [(1, 1, 1, 1)],
    "plain array": lambda: np.zeros(4, dtype=np.uint64),
    "missing field": lambda: np.zeros(3, dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2")]),
    "2-D": lambda: np.zeros((2, 2), dtype=EVENT_DTYPE),
    "non-contiguous": lambda: np.zeros(6, dtype=EVENT_DTYPE)[::2],
}

WRONG_FIELD_DTYPES = {
    "float t": ("t", "<f8"),
    "signed t": ("t", "<i8"),
    "big-endian t": ("t", ">u8"),
    "int32 x": ("x", "<i4"),
    "uint32 y": ("y", "<u4"),
    "big-endian y": ("y", ">u2"),
    "bool p": ("p", "?"),
    "int16 p": ("p", "<i2"),
}


class TestStructuralValidation:
    @pytest.mark.parametrize("case", list(STRUCTURALLY_INVALID))
    def test_rejected_without_changing_anything(self, entry_kind: str, case: str) -> None:
        entry = _primed(entry_kind, "event_count")
        before = entry.observe()
        with pytest.raises(TypeError):
            entry.feed(STRUCTURALLY_INVALID[case]())
        assert entry.observe() == before

    @pytest.mark.parametrize("case", list(WRONG_FIELD_DTYPES))
    def test_wrong_field_dtype_is_a_type_error(self, entry_kind: str, case: str) -> None:
        entry = _primed(entry_kind, "event_count")
        before = entry.observe()
        with pytest.raises(TypeError):
            entry.feed(_with_field(*WRONG_FIELD_DTYPES[case]))
        assert entry.observe() == before

    def test_structure_is_checked_before_timestamp_values(self, entry_kind: str) -> None:
        bad = _with_field("x", "<i4")
        bad["t"] = 2**63
        with pytest.raises(TypeError):
            ENTRIES[entry_kind]("event_count").feed(bad)

    @pytest.mark.parametrize("kernel", ALL)
    def test_extra_fields_are_ignored(self, kernel: str) -> None:
        plain = random_events(3, 40)
        wide = np.zeros(len(plain), dtype=EVENT_DTYPE.descr + [("extra", "<f8"), ("tag", "u1")])
        for name in FIELDS:
            wide[name] = plain[name]
        wide["extra"] = np.nan
        wide["tag"] = 255
        a = KERNELS[kernel].accumulator()
        a.accumulate(wide)
        oracle = KERNELS[kernel].oracle()
        oracle.accumulate(plain)
        assert_matches(a.read(), oracle)


class TestTimestampRange:
    @pytest.mark.parametrize("kernel", ALL)
    @pytest.mark.parametrize("where", ["in bounds", "out of bounds"])
    def test_whole_call_rejected_atomically(self, entry_kind: str, kernel: str, where: str) -> None:
        entry = _primed(entry_kind, kernel)
        before = entry.observe()
        x = 1 if where == "in bounds" else SENSOR[0] + 5
        with pytest.raises(ValueError):
            entry.feed(events((9, 0, 0, 1), (2**63, x, 0, 0), (11, 3, 2, 0)))
        assert entry.observe() == before

    @pytest.mark.parametrize("kernel", ALL)
    def test_engine_state_is_untouched_by_a_rejected_call(self, kernel: str) -> None:
        # Hidden state (the watermark, pending accumulation) only shows at the next publication.
        case = KERNELS[kernel]
        engine = case.engine(interval_ms=0.0)
        oracle = case.oracle()
        good = events((5, 1, 1, 1), (7, 2, 3, 0))
        engine.ingest(good)
        oracle.accumulate(good)
        if case.windowed:
            oracle.close_window()
        with pytest.raises(ValueError):
            engine.ingest(events((9, 0, 0, 1), (2**63, 1, 0, 0), (11, 3, 2, 0)))
        engine.ingest(events())
        oracle.accumulate(events())
        snap = engine.snapshot()
        frame, meta = snap.frame, snap.meta
        assert_matches(frame, oracle)
        assert meta.watermark == oracle.watermark == 7

    @pytest.mark.parametrize("kernel", ALL)
    def test_largest_valid_timestamp_is_accepted(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(events((2**63 - 1, 1, 1, 0)))
        assert acc.watermark == 2**63 - 1

    def test_rejected_call_applies_no_decay_step(self) -> None:
        acc = KERNELS["exp_decay"].accumulator()  # decay 0.5
        acc.accumulate(events((1, 1, 1, 0)))
        with pytest.raises(ValueError):
            acc.accumulate(events((2**63, 1, 1, 0)))
        assert acc.read()[1, 1] == np.float32(1.0)


class TestBoundsAndWatermark:
    @pytest.mark.parametrize("kernel", ALL)
    def test_out_of_bounds_events_are_counted_and_skipped(self, kernel: str) -> None:
        w, h = SENSOR
        batch = events((1, w - 1, h - 1, 1), (2, w, 0, 1), (3, 0, h, 0), (4, 65_535, 65_535, 1))
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(batch)
        oracle = KERNELS[kernel].oracle()
        oracle.accumulate(batch)
        assert acc.events_out_of_bounds == 3
        assert acc.watermark == 1
        assert_matches(acc.read(), oracle)

    @pytest.mark.parametrize("kernel", ALL)
    def test_out_of_bounds_only_call_changes_only_the_count(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        oracle = KERNELS[kernel].oracle()
        for batch in (events((10, 1, 1, 1)), events((5_000, SENSOR[0], 0, 0), (6_000, 0, SENSOR[1], 1))):
            acc.accumulate(batch)
            oracle.accumulate(batch)
        assert acc.watermark == 10
        assert acc.events_out_of_bounds == 2
        assert_matches(acc.read(), oracle)  # exp_decay: the accepted call still decays once

    @pytest.mark.parametrize("kernel", ALL)
    def test_watermark_is_the_maximum_not_the_last(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(events((100, 1, 1, 0), (5, 2, 2, 0)))
        assert acc.watermark == 100
        acc.accumulate(events((50, 3, 3, 1)))
        assert acc.watermark == 100

    @pytest.mark.parametrize("kernel", ALL)
    def test_watermark_is_none_until_an_in_bounds_event(self, kernel: str) -> None:
        acc = KERNELS[kernel].accumulator()
        acc.accumulate(events((99, SENSOR[0], 0, 0)))
        assert acc.watermark is None

    def test_coordinates_x_is_the_column(self) -> None:
        acc = KERNELS["event_count"].accumulator()
        acc.accumulate(events((0, 5, 0, 0), (1, 0, 3, 0)))
        expected = np.zeros((4, 6), dtype=np.uint32)
        expected[0, 5] = 1
        expected[3, 0] = 1
        np.testing.assert_array_equal(acc.read(), expected)


@pytest.mark.parametrize("p", range(256))
def test_every_polarity_value_stays_on_its_pixel(p: int) -> None:
    acc = KERNELS["polarity"].accumulator()
    acc.accumulate(events((0, 5, 3, p)))  # the last pixel: raw p would run off the frame
    acc.accumulate(events((1, 0, 0, p)))
    expected = np.zeros((4, 6, 2), dtype=np.uint32)
    expected[3, 5, 0 if p == 0 else 1] = 1
    expected[0, 0, 0 if p == 0 else 1] = 1
    np.testing.assert_array_equal(acc.read(), expected)


def test_engine_accounting_every_event_accumulated_or_out_of_bounds() -> None:
    engine = KERNELS["event_count"].engine(interval_ms=0.0)
    oracle = KERNELS["event_count"].oracle()
    published = total = 0
    for seed in range(8):
        batch = random_events(seed, 50 + seed)
        engine.ingest(batch)
        oracle.accumulate(batch)
        published += int(engine.snapshot().frame.sum())
        total += len(batch)
    stats = engine.stats
    assert stats.events_ingested == total
    assert stats.events_out_of_bounds == oracle.out_of_bounds
    assert accumulated(stats) == published == oracle.accumulated
