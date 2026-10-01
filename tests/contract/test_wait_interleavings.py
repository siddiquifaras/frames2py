"""Every bounded interleaving of ``wait_for_newer`` against publication, ``reset()`` and ``stop()``.

The real Engine runs with its lifecycle lock, its publisher's slot, its waiter registry, the
waiters' locks and the clock ``wait_for_newer`` reads replaced by controlled stand-ins
(``tests/contract/interleave.py``). The scheduler then decides the order of every lock
operation, slot load and store, registry operation, clock read and timeout. For each schedule:

- no thread is left blocked: a waiter without a timeout that never returns is a lost wakeup,
  reported by the harness as a deadlock
- a returned snapshot is newer than the sequence passed, was published, and is what the
  waiter's last read of the publisher returned (the latest at that read)
- ``None`` comes back only from a timed wait, and only when the waiter's last read came after
  the clock reached its deadline and found nothing newer
- the publishing thread never blocks on a lock a waiter holds
- a publication releases only waiters registered before it started releasing, and every
  schedule ends: a publication whose releasing never ends fails as unbounded
- no registration is left behind

Mutants (deliberately broken Engines) check that each of these can fail. The model is
sequentially consistent: it covers the orderings of these steps, not CPython's memory model,
so it can't show the store-buffering hazard an unlocked "any waiters?" check would have on
free-threaded builds; that is the TLA+ model's job (decisions.md 80).
"""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Callable
from typing import Any

import pytest

from tests.contract.api import impl
from tests.contract.helpers import SENSOR, SUPPORTED_RUNTIME, events
from tests.contract.interleave import (
    ControlledDeque,
    ControlledLock,
    ControlledSlot,
    ControlledTime,
    ControlledWaiterLock,
    Event,
    Scheduler,
    explore,
)

pytestmark = [
    pytest.mark.skipif(not SUPPORTED_RUNTIME, reason="the Engine refuses this free-threaded runtime"),
    pytest.mark.timeout(600),
]

engine_module = importlib.import_module(f"{impl.__name__}._engine")

HOUR_MS = 3_600_000.0
TIMEOUT = 1.0
"""Virtual seconds. Timed waits time out only when the scheduler picks them to."""

Op = tuple[str, Any]  # (call, argument): ("ingest", events), ("reset", None), ("wait", (sequence, timeout))
NEXT = "next"
"""As a wait's sequence: the sequence of the snapshot the same thread's previous wait returned."""


def batch(k: int) -> Any:
    return events((k, k % SENSOR[0], (k // SENSOR[0]) % SENSOR[1], 1))


class _Wiring:
    """Points the patched module names at the current schedule's scheduler."""

    def __init__(self) -> None:
        self.scheduler: Scheduler | None = None
        self._labels = itertools.count()

    def allocate_lock(self) -> ControlledWaiterLock:
        assert self.scheduler is not None
        return ControlledWaiterLock(self.scheduler, f"{self.scheduler.current()}#{next(self._labels)}")


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch) -> _Wiring:
    wiring = _Wiring()
    monkeypatch.setattr(engine_module, "_allocate_lock", wiring.allocate_lock)
    return wiring


def _body(scheduler: Scheduler, engine: Any, ops: list[Op]) -> Callable[[], None]:
    def run() -> None:
        previous: Any = None
        for i, (op, arg) in enumerate(ops):
            if op == "wait":
                sequence, timeout = arg
                if sequence == NEXT:
                    sequence = None if previous is None else previous.meta.sequence
                scheduler.log("call", (i, sequence, timeout))
                result = engine.wait_for_newer(sequence, timeout=timeout)
                scheduler.log("result", (i, result))
                previous = result if result is not None else previous
            elif arg is None:
                getattr(engine, op)()
            else:
                getattr(engine, op)(arg)

    return run


def _check(scheduler: Scheduler, engine: Any, waiters: set[str]) -> None:
    events_ = scheduler.events
    stored = {id(e.data) for e in events_ if e.kind == "store" and e.data is not None}

    # Results.
    calls: dict[tuple[str, int], list[Event]] = {}
    current: dict[str, tuple[str, int]] = {}
    for e in events_:
        if e.thread not in waiters:
            continue
        if e.kind == "call":
            current[e.thread] = (e.thread, e.data[0])
        if e.thread in current:
            calls.setdefault(current[e.thread], []).append(e)
    for (thread, index), trace in calls.items():
        _, sequence, timeout = trace[0].data
        floor = -1 if sequence is None else sequence
        results = [e.data[1] for e in trace if e.kind == "result"]
        assert len(results) == 1, f"{thread} call {index} did not return"
        result = results[0]
        loads = [e for e in trace if e.kind == "load"]
        assert loads, f"{thread} call {index} never read the publisher"
        last = loads[-1]
        if result is not None:
            assert result.meta.sequence > floor, (
                f"{thread} call {index}: sequence {result.meta.sequence} is not newer than {floor}")
            assert id(result) in stored, f"{thread} call {index}: returned a snapshot that was never published"
            assert last.data is result, f"{thread} call {index}: did not return what its last read got"
            continue
        assert timeout is not None, f"{thread} call {index}: returned None without a timeout"
        clocks = [e for e in trace if e.kind == "clock"]
        deadline = clocks[0].data + timeout
        assert any(e.kind == "clock" and e.data >= deadline and e.index < last.index for e in trace), (
            f"{thread} call {index}: returned None, but its last read was not after the deadline")
        assert last.data is None or last.data.meta.sequence <= floor, (
            f"{thread} call {index}: returned None although its last read found sequence {last.data.meta.sequence}")

    # The publishing thread never blocks on a lock a waiter holds.
    for e in events_:
        if e.kind == "blocked" and e.thread not in waiters:
            assert e.data not in waiters, f"{e.thread} blocked on a lock held by waiter {e.data}"

    # A publication releases only waiters registered before it started releasing.
    appended = {id(e.data): e.index for e in events_ if e.kind == "append"}
    drain_start: int | None = None
    for e in events_:
        if e.thread in waiters:
            continue
        if e.kind == "store":
            drain_start = None
        elif e.kind in ("append", "popleft") and drain_start is None:
            drain_start = e.index
        if e.kind == "popleft" and isinstance(e.data, ControlledWaiterLock):
            assert drain_start is not None
            assert appended[id(e.data)] < drain_start, (
                f"{e.thread} released {e.data.label}, registered after its publication began releasing")

    assert len(engine._waiters) == 0, f"registrations left behind: {engine._waiters.items}"


def scenario(
    wiring: _Wiring,
    threads: dict[str, list[Op]],
    interval_ms: float = 0.0,
    engine_type: Any = None,
) -> Callable[[Scheduler], tuple[dict[str, Callable[[], None]], Callable[[Scheduler], None]]]:
    waiters = {name for name, ops in threads.items() if all(op == "wait" for op, _ in ops)}
    lifecycle_threads = len(threads) - len(waiters)
    timed = [name for name, ops in threads.items() if any(op == "wait" and arg[1] is not None for op, arg in ops)]
    assert len(timed) <= 1, "clock reads are not yield points: at most one timed waiter (interleave.ControlledTime)"

    def build(scheduler: Scheduler) -> tuple[dict[str, Callable[[], None]], Callable[[Scheduler], None]]:
        wiring.scheduler = scheduler
        engine = (engine_type or impl.Engine)(SENSOR, "event_count", snapshot_interval_ms=interval_ms)
        if lifecycle_threads > 1:  # an uncontended lock adds schedules and no orderings
            engine._lifecycle = ControlledLock(scheduler)
        engine._publisher._slot = ControlledSlot(scheduler, engine._publisher._slot[0])
        engine._waiters = ControlledDeque(scheduler)
        if hasattr(engine, "_cond"):
            engine._cond = ControlledLock(scheduler)
        engine_module.time = ControlledTime(scheduler)  # restored by the fixture's monkeypatch
        bodies = {name: _body(scheduler, engine, ops) for name, ops in threads.items()}
        return bodies, lambda s: _check(s, engine, waiters)

    return build


SCENARIOS: dict[str, dict[str, Any]] = {
    "registration races one publication": dict(
        threads={"P": [("ingest", batch(1))], "W": [("wait", (None, None))]},
    ),
    "two waiters, one publication": dict(
        threads={"P": [("ingest", batch(1))], "W1": [("wait", (None, None))], "W2": [("wait", (None, None))]},
    ),
    "waiting past the first publication": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2))], "W": [("wait", (1, None))]},
    ),
    "repeated waits": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2))],
                 "W": [("wait", (None, None)), ("wait", (NEXT, TIMEOUT))]},
    ),
    "a timeout races a publication": dict(
        threads={"P": [("ingest", batch(1))], "W": [("wait", (None, TIMEOUT)), ("wait", (NEXT, TIMEOUT))]},
    ),
    "reset between publications": dict(
        threads={"P": [("ingest", batch(1)), ("reset", None), ("ingest", batch(2))], "W": [("wait", (1, None))]},
    ),
    "reset wakes nobody": dict(
        threads={"P": [("ingest", batch(1)), ("reset", None)], "W": [("wait", (1, TIMEOUT))]},
    ),
    "stop publishes the pending window": dict(
        threads={"P": [("ingest", batch(1)), ("ingest", batch(2)), ("stop", None)], "W": [("wait", (1, None))]},
        interval_ms=HOUR_MS,
    ),
    "stop with nothing pending wakes nobody": dict(
        threads={"P": [("ingest", batch(1)), ("stop", None)], "W": [("wait", (1, TIMEOUT))]},
        interval_ms=HOUR_MS,
    ),
}
"""``reset()`` and ``stop()`` run on the producer's thread here. Their ordering against
``ingest()`` from other threads is ``test_interleavings.py``'s; across threads with waiters,
``test_wait_for_newer.py``'s real-thread tests."""


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_interleaving_meets_the_wait_contract(wiring: _Wiring, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "time", engine_module.time)
    explored = explore(scenario(wiring, **SCENARIOS[name]), max_schedules=100_000, max_steps=500)
    assert explored > 1


# ------------------------------------------------------------------ mutants


def _newer(snapshot: Any, floor: int) -> bool:
    return snapshot is not None and snapshot.meta.sequence > floor


class _NoRereadAfterRegistering(impl.Engine):
    """Checks, then registers and blocks: the classic lost wakeup."""

    def wait_for_newer(self, sequence: Any, *, timeout: Any = None) -> Any:
        floor = -1 if sequence is None else sequence
        while True:
            snapshot = self._publisher.read()
            if _newer(snapshot, floor):
                return snapshot
            waiter = engine_module._allocate_lock()
            waiter.acquire()
            self._waiters.append(waiter)
            try:
                waiter.acquire()
            finally:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass


class _ReleasesBeforeStoring(impl.Engine):
    """Releases the registered waiters, then publishes."""

    def _publish(self, now_ns: int) -> None:
        self._sequence += 1
        buffer = self._publisher.begin_write()
        self._accumulator._read_into(buffer)
        marker = object()
        self._waiters.append(marker)
        while (waiter := self._waiters.popleft()) is not marker:
            waiter.release()
        self._publisher.end_write(impl.SnapshotMeta(watermark=self._accumulator.watermark, sequence=self._sequence))
        self._accumulator._close_window()
        self._snapshots_published += 1
        self._pending = False
        self._last_published_ns = now_ns


class _ConditionVariable(impl.Engine):
    """Condition-variable style: a waiter holds a registry lock while it checks and registers,
    and the publisher takes the same lock to release waiters."""

    _cond: Any = None

    def wait_for_newer(self, sequence: Any, *, timeout: Any = None) -> Any:
        floor = -1 if sequence is None else sequence
        while True:
            with self._cond:
                snapshot = self._publisher.read()
                if _newer(snapshot, floor):
                    return snapshot
                waiter = engine_module._allocate_lock()
                waiter.acquire()
                self._waiters.append(waiter)
            waiter.acquire()

    def _publish(self, now_ns: int) -> None:
        self._sequence += 1
        buffer = self._publisher.begin_write()
        self._accumulator._read_into(buffer)
        self._publisher.end_write(impl.SnapshotMeta(watermark=self._accumulator.watermark, sequence=self._sequence))
        with self._cond:
            while len(self._waiters):
                self._waiters.popleft().release()
        self._accumulator._close_window()
        self._snapshots_published += 1
        self._pending = False
        self._last_published_ns = now_ns


class _ReleasesUntilEmpty(impl.Engine):
    """Releases until the registry is empty instead of up to its own marker."""

    def _publish(self, now_ns: int) -> None:
        self._sequence += 1
        buffer = self._publisher.begin_write()
        self._accumulator._read_into(buffer)
        self._publisher.end_write(impl.SnapshotMeta(watermark=self._accumulator.watermark, sequence=self._sequence))
        while len(self._waiters):
            self._waiters.popleft().release()
        self._accumulator._close_window()
        self._snapshots_published += 1
        self._pending = False
        self._last_published_ns = now_ns


class _DeregistersTheOldest(impl.Engine):
    """Removes the oldest registration on the way out, which may be another waiter's."""

    def wait_for_newer(self, sequence: Any, *, timeout: Any = None) -> Any:
        floor = -1 if sequence is None else sequence
        while True:
            snapshot = self._publisher.read()
            if _newer(snapshot, floor):
                return snapshot
            waiter = engine_module._allocate_lock()
            waiter.acquire()
            self._waiters.append(waiter)
            try:
                snapshot = self._publisher.read()
                if _newer(snapshot, floor):
                    return snapshot
                waiter.acquire()
            finally:
                if len(self._waiters):
                    self._waiters.popleft()


class _ReturnsAfterAnyWake(impl.Engine):
    """Returns whatever it reads once woken, instead of waiting on when it isn't newer."""

    def wait_for_newer(self, sequence: Any, *, timeout: Any = None) -> Any:
        floor = -1 if sequence is None else sequence
        snapshot = self._publisher.read()
        if _newer(snapshot, floor):
            return snapshot
        waiter = engine_module._allocate_lock()
        waiter.acquire()
        self._waiters.append(waiter)
        try:
            snapshot = self._publisher.read()
            if _newer(snapshot, floor):
                return snapshot
            waiter.acquire()
        finally:
            try:
                self._waiters.remove(waiter)
            except ValueError:
                pass
        return self._publisher.read()


class _NewerMeansAtLeast(impl.Engine):
    """Treats a snapshot with the same sequence as newer."""

    def wait_for_newer(self, sequence: Any, *, timeout: Any = None) -> Any:
        return super().wait_for_newer(None if sequence is None or sequence == 0 else sequence - 1, timeout=timeout)


MUTANTS = {
    # mutant: (scenario, engine type, what must catch it)
    "no re-read after registering": (
        "registration races one publication", _NoRereadAfterRegistering, r"Deadlock"),
    "release before the store": (
        "registration races one publication", _ReleasesBeforeStoring, r"Deadlock"),
    "condition-variable wake": (
        "registration races one publication", _ConditionVariable, r"P blocked on a lock held by waiter W"),
    "release until empty": (
        "a timeout races a publication", _ReleasesUntilEmpty,
        r"registered after its publication began releasing|Unbounded"),
    "deregister the oldest": (
        "two waiters, one publication", _DeregistersTheOldest, r"Deadlock|release unlocked|pop from an empty deque"),
    "return after any wake": (
        "reset between publications", _ReturnsAfterAnyWake, r"is not newer than|returned None without a timeout"),
    "newer means at least": (
        "waiting past the first publication", _NewerMeansAtLeast, r"is not newer than"),
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_the_checks_catch_a_broken_wait(wiring: _Wiring, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "time", engine_module.time)
    base, engine_type, check = MUTANTS[name]
    with pytest.raises(Exception, match=check):  # a check's AssertionError, or the producer crashing
        explore(scenario(wiring, **{**SCENARIOS[base], "engine_type": engine_type}), max_schedules=100_000, max_steps=500)
