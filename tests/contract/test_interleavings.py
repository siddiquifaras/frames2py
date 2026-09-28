"""Every bounded interleaving of lifecycle calls, ingest and snapshot reads on the real Engine.

The Engine's lifecycle lock and its publisher's slot are replaced with controlled stand-ins
(``tests/contract/interleave.py``), so the scheduler decides the order of every lock
acquire and release and every slot store and load. For each schedule the result is checked
against the same calls made one at a time, in the order the real run acquired the lock, on
an Engine that is not instrumented:

- each call's outcome (ownership: ``RuntimeError`` for a non-producer ``ingest()``)
- the final stats and snapshot
- no published frame changes after it is stored: a reader sees the bytes it was stored
  with, and still does after all threads finished
- every snapshot a reader got is a publication of that sequential run, frame and metadata
  together
- a read that starts after ``reset()`` returned gets ``None`` or a later publication
- readers (``snapshot()``, ``stats``) never take the lifecycle lock

Mutants (deliberately broken Engines and publishers) check that these checks can fail.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from tests.contract.api import impl
from tests.contract.helpers import SENSOR, SUPPORTED_RUNTIME, events
from tests.contract.interleave import ControlledLock, ControlledSlot, Scheduler, explore

pytestmark = [
    pytest.mark.skipif(not SUPPORTED_RUNTIME, reason="the Engine refuses this free-threaded runtime"),
    pytest.mark.timeout(300),
]

publish = importlib.import_module(f"{impl.__name__}.publish")

HOUR_MS = 3_600_000.0
Op = tuple[str, Any]  # (method, argument or None)


def batch(k: int) -> Any:
    """One event at a pixel of its own, at t = k."""
    return events((k, k % SENSOR[0], (k // SENSOR[0]) % SENSOR[1], 1))


class _RecordingSlot(ControlledSlot):
    """Also logs the stored snapshot's frame bytes at the moment it is stored."""

    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)
        if value is not None:
            self._s.log("stored", (value, value.frame.tobytes()))


def _instrument(engine: Any, scheduler: Scheduler) -> Any:
    engine._lifecycle = ControlledLock(scheduler)
    engine._publisher._slot = _RecordingSlot(scheduler, engine._publisher._slot[0])
    return engine


def _body(scheduler: Scheduler, engine: Any, name: str, ops: list[Op]) -> Callable[[], None]:
    def run() -> None:
        for i, (op, arg) in enumerate(ops):
            label = f"{name}.{i}"
            scheduler.log("begin", (label, op, arg))
            result: Any = None
            try:
                if op == "stats":
                    result = engine.stats
                elif arg is None:
                    result = getattr(engine, op)()
                else:
                    result = getattr(engine, op)(arg)
                outcome = "ok"
            except RuntimeError:
                outcome = "RuntimeError"
            read_bytes = result.frame.tobytes() if op == "snapshot" and result is not None else None
            scheduler.log("end", (label, op, outcome, (result, read_bytes)))

    return run


def _observe(engine: Any) -> tuple[Any, ...]:
    stats = engine.stats
    snap = engine.snapshot()
    published = None if snap is None else (snap.frame.tobytes(), snap.meta)
    return stats.events_ingested, stats.events_out_of_bounds, stats.snapshots_published, published


def _sequential(order: list[tuple[str, str, str, Any]], kernel: str, interval_ms: float) -> dict[str, Any]:
    """The same lifecycle calls, one at a time, in lock order, on a plain Engine."""
    engine = impl.Engine(SENSOR, kernel, snapshot_interval_ms=interval_ms)
    owner = None
    outcomes: dict[str, str] = {}
    publications: dict[int, tuple[bytes, Any]] = {}
    floors: dict[str, int] = {}
    for thread, label, op, arg in order:
        if op == "ingest":
            owner = owner or thread
            if thread != owner:
                outcomes[label] = "RuntimeError"
                continue
            engine.ingest(arg)
        else:
            getattr(engine, op)()
        outcomes[label] = "ok"
        snap = engine.snapshot()
        if snap is not None:
            publications[snap.meta.sequence] = (snap.frame.tobytes(), snap.meta)
        if op == "reset":
            floors[label] = max(publications, default=0)
    return {"outcomes": outcomes, "publications": publications, "floors": floors, "final": _observe(engine)}


def _check(scheduler: Scheduler, engine: Any, kernel: str, interval_ms: float, readers: set[str]) -> None:
    events_ = scheduler.events
    current: dict[str, tuple[str, str, Any]] = {}
    order: list[tuple[str, str, str, Any]] = []
    seen_ops: set[str] = set()
    begins: dict[str, int] = {}
    ends: dict[str, tuple[int, str, str, Any]] = {}
    stored: dict[int, bytes] = {}
    for e in events_:
        if e.kind == "begin":
            label, op, arg = e.data
            current[e.thread] = (label, op, arg)
            begins[label] = e.index
        elif e.kind == "end":
            label, op, outcome, result = e.data
            ends[label] = (e.index, op, outcome, result)
        elif e.kind == "stored":
            value, frame_bytes = e.data
            stored[id(value)] = frame_bytes
        elif e.kind == "acquire":
            assert e.thread not in readers, f"reader {e.thread} took the lifecycle lock"
            label, op, arg = current[e.thread]
            if label not in seen_ops:
                seen_ops.add(label)
                order.append((e.thread, label, op, arg))
    expected = _sequential(order, kernel, interval_ms)

    for thread, label, op, _ in order:
        assert ends[label][2] == expected["outcomes"][label], f"outcome: {label} {op}: {ends[label][2]}, sequential {expected['outcomes'][label]}"
    final = _observe(engine)
    for field, got, want in zip(("events_ingested", "events_out_of_bounds", "snapshots_published", "final snapshot"),
                                final, expected["final"]):
        assert got == want, f"{field} differs from the sequential run: {got!r} vs {want!r}"

    publications = expected["publications"]
    reset_ends = {label: ends[label][0] for _, label, op, _ in order if op == "reset"}
    reads = [(label, *ends[label][3]) for label in ends if ends[label][1] == "snapshot"]
    for label, result, read_bytes in reads:
        if result is None:
            continue
        assert id(result) in stored, f"unpublished: {label} got an object that was never stored"
        at_store = stored[id(result)]
        assert read_bytes == at_store and result.frame.tobytes() == at_store, (
            f"immutability: the frame {label} got changed after it was stored")
    for label, result, _ in reads:
        floor = max((expected["floors"][r] for r, at in reset_ends.items() if at < begins[label]), default=0)
        if result is None:
            continue
        sequence = result.meta.sequence
        assert sequence in publications, f"unpublished: {label} got sequence {sequence}, never published sequentially"
        assert (stored[id(result)], result.meta) == publications[sequence], (
            f"pairing: {label}'s frame and metadata don't match publication {sequence}")
        assert sequence > floor, f"after reset: {label} started after reset() returned but got pre-reset sequence {sequence}"


def scenario(
    threads: dict[str, list[Op]],
    kernel: str = "event_count",
    interval_ms: float = 0.0,
    engine_type: Any = None,
    publisher_type: Any = None,
) -> Callable[[Scheduler], tuple[dict[str, Callable[[], None]], Callable[[Scheduler], None]]]:
    readers = {name for name, ops in threads.items() if all(op in ("snapshot", "stats") for op, _ in ops)}

    def build(scheduler: Scheduler) -> tuple[dict[str, Callable[[], None]], Callable[[Scheduler], None]]:
        engine = (engine_type or impl.Engine)(SENSOR, kernel, snapshot_interval_ms=interval_ms)
        if publisher_type is not None:
            shape, dtype = engine._accumulator._output_spec
            engine._publisher = publisher_type(shape, dtype)
        _instrument(engine, scheduler)
        bodies = {name: _body(scheduler, engine, name, ops) for name, ops in threads.items()}
        return bodies, lambda s: _check(s, engine, kernel, interval_ms, readers)

    return build


SCENARIOS = {
    "ingest vs stop": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2))], "C": [("stop", None)], "R": [("snapshot", None)]},
        interval_ms=HOUR_MS,
    ),
    "ingest vs reset": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2))], "C": [("reset", None)],
                 "R": [("snapshot", None), ("snapshot", None)]},
    ),
    "stop vs reset": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2))], "C1": [("stop", None)], "C2": [("reset", None)]},
        interval_ms=HOUR_MS,
    ),
    "stop and start from another thread": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2)), ("ingest", batch(3))],
                 "C": [("stop", None), ("start", None)], "R": [("snapshot", None), ("stats", None)]},
    ),
    "producer ownership across reset": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(3))], "X": [("ingest", batch(2))],
                 "C": [("reset", None)]},
    ),
    "snapshot during publication, running kernel": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2)), ("reset", None), ("ingest", batch(3))],
                 "R1": [("snapshot", None), ("snapshot", None)], "R2": [("stats", None), ("snapshot", None)]},
        kernel="time_surface",
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_interleaving_matches_a_sequential_run(name: str) -> None:
    explored = explore(scenario(**SCENARIOS[name]))
    assert explored > 1


# ------------------------------------------------------------------ mutants


class _RunningCheckedBeforeTheLock(impl.Engine):
    def ingest(self, events: Any) -> None:
        caller = threading.get_ident()
        if not self._running:
            return
        with self._lifecycle:
            if self._producer is not None and caller != self._producer:
                raise RuntimeError("not the producer")
            self._ingest_locked(events, caller)

    def _ingest_locked(self, events: Any, caller: int) -> None:
        inside = self._accumulator._accumulate(events)
        self._producer = caller
        self._events_ingested += len(events)
        if inside:
            self._pending = True
        now = time.monotonic_ns()
        if self._last_published_ns is None or now - self._last_published_ns >= self._interval_ns:
            self._publish(now)


class _NoOwnershipCheck(_RunningCheckedBeforeTheLock):
    def ingest(self, events: Any) -> None:
        with self._lifecycle:
            if self._running:
                self._ingest_locked(events, threading.get_ident())


class _ResetInTwoSteps(impl.Engine):
    def reset(self) -> None:
        with self._lifecycle:
            self._publisher.reset()
        with self._lifecycle:
            self._accumulator.reset()
            self._events_ingested = 0
            self._snapshots_published = 0
            self._pending = False
            self._last_published_ns = None


class _SnapshotUnderTheLock(impl.Engine):
    def snapshot(self) -> Any:
        with self._lifecycle:
            return self._publisher.read()


class _MetadataStoredSeparately(publish.ImmutablePublisher):  # type: ignore[misc]
    """Stores the new frame with the previous metadata first, then the right pair."""

    def end_write(self, meta: Any) -> None:
        frame = self._writing
        self._writing = None
        frame.flags.writeable = False
        previous = self._slot[0]
        stale = previous.meta if previous is not None else meta
        self._slot[0] = publish.Snapshot(frame.view(), stale)
        self._slot[0] = publish.Snapshot(frame.view(), meta)


class _ReusesPublishedBuffers(publish.ImmutablePublisher):  # type: ignore[misc]
    """Writes each publication into the previously published buffer."""

    def begin_write(self) -> Any:
        previous = self._slot[0]
        if previous is None:
            return super().begin_write()
        base = previous.frame.base
        base.flags.writeable = True
        self._writing = base
        return base


MUTANTS = {
    # mutant: (scenario, overrides, the check that must catch it)
    "running checked before the lock": (
        "ingest vs stop", dict(engine_type=_RunningCheckedBeforeTheLock), r"events_ingested differs"),
    "no ownership check": (
        "producer ownership across reset", dict(engine_type=_NoOwnershipCheck), r"outcome: X\.0 ingest: ok, sequential RuntimeError"),
    "reset in two locked steps": (
        "ingest vs reset", dict(engine_type=_ResetInTwoSteps), r"(events_ingested|snapshots_published) differs"),
    "snapshot takes the lifecycle lock": (
        "stop and start from another thread", dict(engine_type=_SnapshotUnderTheLock), r"reader R took the lifecycle lock"),
    "metadata stored separately": (
        "ingest vs reset", dict(publisher_type=_MetadataStoredSeparately), r"pairing: "),
    "published buffers reused": (
        "snapshot during publication, running kernel", dict(publisher_type=_ReusesPublishedBuffers), r"immutability: "),
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_the_checks_catch_a_broken_ordering(name: str) -> None:
    base, override, check = MUTANTS[name]
    with pytest.raises(AssertionError, match=check):
        explore(scenario(**{**SCENARIOS[base], **override}))

