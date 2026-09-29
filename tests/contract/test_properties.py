"""Contract properties over generated event streams, against the independent oracle.

Streams mix unordered timestamps (including near-2**63 spikes), every polarity value,
out-of-bounds coordinates, empty calls and, where a property is about rejection, calls holding
a timestamp at or above 2**63.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contract.api import EVENT_DTYPE
from tests.contract.helpers import ALL, KERNELS, ORDER_INVARIANT, SENSOR, assert_matches

TIMESTAMP_LIMIT = 2**63
WIDTH, HEIGHT = SENSOR

Row = tuple[int, int, int, int]

timestamps = st.one_of(st.integers(0, 20_000), st.integers(0, TIMESTAMP_LIMIT - 1))
"""Mostly within a few thousand ``tau_us`` of each other (so decay rebases), sometimes anywhere."""

rows = st.tuples(timestamps, st.integers(0, WIDTH + 1), st.integers(0, HEIGHT + 1), st.integers(0, 255))
calls = st.lists(st.lists(rows, max_size=30), max_size=8)
rejected_rows = st.tuples(st.integers(TIMESTAMP_LIMIT, 2**64 - 1), st.integers(0, WIDTH + 1),
                          st.integers(0, HEIGHT + 1), st.integers(0, 255))


def array(batch: list[Row]) -> Any:
    return np.array(batch, dtype=EVENT_DTYPE)


def in_bounds(row: Row) -> bool:
    return row[1] < WIDTH and row[2] < HEIGHT


@st.composite
def mixed_calls(draw: st.DrawFn) -> list[tuple[list[Row], bool]]:
    """Calls, each flagged ``True`` if it holds a timestamp the contract rejects (about one in four)."""
    out = []
    for batch in draw(calls):
        rejected = draw(st.sampled_from([False, False, False, True]))
        if rejected:
            batch = list(batch)
            batch.insert(draw(st.integers(0, len(batch))), draw(rejected_rows))
        out.append((batch, rejected))
    return out


@st.composite
def partitioned(draw: st.DrawFn) -> tuple[list[Row], list[list[Row]]]:
    """One stream, and the same events in another order cut into calls, empty ones included."""
    batch = draw(st.lists(rows, min_size=1, max_size=120))
    order = draw(st.permutations(range(len(batch))))
    cuts = sorted(draw(st.lists(st.integers(0, len(batch)), max_size=10)))
    reordered = [batch[i] for i in order]
    parts = [reordered[a:b] for a, b in zip([0, *cuts], [*cuts, len(batch)])]
    return batch, parts


@pytest.mark.parametrize("kernel", ALL)
@given(stream=mixed_calls())
def test_any_sequence_of_calls_matches_the_oracle_and_rejection_changes_nothing(
    kernel: str, stream: list[tuple[list[Row], bool]]
) -> None:
    acc, oracle = KERNELS[kernel].accumulator(), KERNELS[kernel].oracle()
    for batch, rejected in stream:
        before = (acc.read(), acc.watermark, acc.events_out_of_bounds)
        if rejected:
            with pytest.raises(ValueError):
                acc.accumulate(array(batch))
            np.testing.assert_array_equal(acc.read(), before[0])
            assert (acc.watermark, acc.events_out_of_bounds) == before[1:]
            continue
        acc.accumulate(array(batch))
        oracle.accumulate(array(batch))
        assert_matches(acc.read(), oracle)
        assert acc.watermark == oracle.watermark
        assert acc.events_out_of_bounds == oracle.out_of_bounds


@pytest.mark.parametrize("kernel", ORDER_INVARIANT)
@given(case=partitioned())
def test_order_and_call_partition_do_not_change_the_result(
    kernel: str, case: tuple[list[Row], list[list[Row]]]
) -> None:
    batch, parts = case
    oracle = KERNELS[kernel].oracle()
    oracle.accumulate(array(batch))
    acc = KERNELS[kernel].accumulator()
    for part in parts:
        acc.accumulate(array(part))
    assert_matches(acc.read(), oracle)
    assert acc.watermark == oracle.watermark
    assert acc.events_out_of_bounds == oracle.out_of_bounds


@pytest.mark.parametrize("kernel", ALL)
@given(stream=mixed_calls())
def test_the_engine_accounts_for_every_accepted_event_and_ignores_rejected_calls(
    kernel: str, stream: list[tuple[list[Row], bool]]
) -> None:
    engine = KERNELS[kernel].engine(interval_ms=0.0)
    twin = KERNELS[kernel].engine(interval_ms=0.0)  # sees only the accepted calls
    accepted: list[Row] = []
    for batch, rejected in stream:
        if rejected:
            with pytest.raises(ValueError):
                engine.ingest(array(batch))
            continue
        engine.ingest(array(batch))
        twin.ingest(array(batch))
        accepted += batch

    stats, twin_stats = engine.stats, twin.stats
    assert stats.events_ingested == twin_stats.events_ingested == len(accepted)
    assert stats.events_out_of_bounds == twin_stats.events_out_of_bounds == sum(not in_bounds(r) for r in accepted)
    assert stats.snapshots_published == twin_stats.snapshots_published
    snapshot, twin_snapshot = engine.snapshot(), twin.snapshot()
    if snapshot is None or twin_snapshot is None:
        assert snapshot is twin_snapshot is None
        return
    inside = [r[0] for r in accepted if in_bounds(r)]
    assert snapshot.meta.watermark == twin_snapshot.meta.watermark == (max(inside) if inside else None)
    assert snapshot.meta.sequence == twin_snapshot.meta.sequence
    np.testing.assert_array_equal(snapshot.frame, twin_snapshot.frame)
