"""Deterministic interleaving of real code, for tests only.

Threads run one at a time. Each stops at a yield point (a controlled lock's acquire or
release, a controlled slot's load or store) and waits until the scheduler picks it.
``explore()`` re-runs a scenario from scratch under every schedule, depth first, so every
ordering of the yield points is covered, not a random sample.

Threads are started one after another and each runs alone up to its first yield point; the
code before it must not touch state shared with the other threads. Between two yield points
a thread runs uninterrupted: this enumerates orderings at the chosen boundaries, which is a
model of the orderings, not of a memory model.

Every yield point, and every entry added with ``Scheduler.log``, goes into
``Scheduler.events`` in execution order, so checks can refer to what happened before what.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class Deadlock(Exception):
    pass


class _Abandoned(BaseException):
    """Unwinds a managed thread when a run is abandoned."""


@dataclass(frozen=True)
class Event:
    index: int
    thread: str
    kind: str
    data: Any = None


@dataclass
class _Thread:
    name: str
    body: Callable[[], None]
    go: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    state: str = "new"  # new, parked, waiting-lock, running, done
    wants: Any = None
    error: BaseException | None = None


class Scheduler:
    """Runs managed threads one step at a time, following a list of choices."""

    def __init__(self, prefix: list[int]) -> None:
        self._prefix = prefix
        self.choices: list[int] = []
        self.enabled_counts: list[int] = []
        self.events: list[Event] = []
        self._threads: list[_Thread] = []
        self._by_ident: dict[int, _Thread] = {}
        self._cv = threading.Condition()
        self._abandon = False

    def current(self) -> str | None:
        me = self._by_ident.get(threading.get_ident())
        return None if me is None else me.name

    def log(self, kind: str, data: Any = None) -> None:
        name = self.current()
        if name is not None:
            self.events.append(Event(len(self.events), name, kind, data))

    # called from managed threads ----------------------------------------

    def point(self) -> None:
        """Park until the scheduler picks this thread. No-op outside managed threads."""
        me = self._by_ident.get(threading.get_ident())
        if me is None:
            return
        with self._cv:
            me.state = "parked"
            self._cv.notify_all()
        me.go.acquire()
        if self._abandon:
            raise _Abandoned

    def wait_for(self, lock: ControlledLock) -> None:
        me = self._by_ident[threading.get_ident()]
        with self._cv:
            me.state, me.wants = "waiting-lock", lock
            self._cv.notify_all()
        me.go.acquire()
        if self._abandon:
            raise _Abandoned

    # driver ---------------------------------------------------------------

    def run(self, bodies: dict[str, Callable[[], None]]) -> None:
        started = []
        try:
            for name, body in bodies.items():
                t = _Thread(name, body)
                self._threads.append(t)
                th = threading.Thread(target=self._wrap, args=(t,), daemon=True)
                th.start()
                started.append(th)
                with self._cv:
                    self._cv.wait_for(lambda: t.state != "new")
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: all(t.state in ("parked", "waiting-lock", "done") for t in self._threads))
                    enabled = [t for t in self._threads if self._enabled(t)]
                    if not enabled:
                        if all(t.state == "done" for t in self._threads):
                            break
                        raise Deadlock([t.name for t in self._threads if t.state != "done"])
                    step = len(self.choices)
                    pick = self._prefix[step] if step < len(self._prefix) else 0
                    self.choices.append(pick)
                    self.enabled_counts.append(len(enabled))
                    chosen = enabled[pick]
                    chosen.state = "running"
                chosen.go.release()
        finally:
            self._abandon = True
            for t in self._threads:
                if t.state != "done":
                    t.go.release()
            for th in started:
                th.join()
        for t in self._threads:
            if t.error is not None:
                raise t.error

    def _enabled(self, t: _Thread) -> bool:
        if t.state == "parked":
            return True
        if t.state == "waiting-lock":
            return t.wants.holder is None
        return False

    def _wrap(self, t: _Thread) -> None:
        self._by_ident[threading.get_ident()] = t
        try:
            t.body()
        except _Abandoned:
            pass
        except BaseException as error:  # noqa: BLE001 - re-raised by run()
            t.error = error
        finally:
            with self._cv:
                t.state = "done"
                self._cv.notify_all()


class ControlledLock:
    """A stand-in for ``threading.Lock``: acquire and release are yield points and are
    logged as ``acquire`` and ``release`` events."""

    def __init__(self, scheduler: Scheduler) -> None:
        self._s = scheduler
        self.holder: int | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self._s.point()
        me = threading.get_ident()
        while self.holder is not None:
            if not blocking:
                return False
            self._s.wait_for(self)
        self.holder = me
        self._s.log("acquire")
        return True

    def release(self) -> None:
        if self.holder != threading.get_ident():
            raise RuntimeError("release of a lock this thread doesn't hold")
        self.holder = None
        self._s.log("release")
        self._s.point()

    def locked(self) -> bool:
        return self.holder is not None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


class ControlledSlot(list):  # type: ignore[type-arg]
    """A one-element list whose item load and store are yield points, logged as ``load``
    and ``store`` events with the value loaded or stored."""

    def __init__(self, scheduler: Scheduler, value: Any = None) -> None:
        super().__init__([value])
        self._s = scheduler

    def __getitem__(self, index: Any) -> Any:
        self._s.point()
        value = super().__getitem__(index)
        self._s.log("load", value)
        return value

    def __setitem__(self, index: Any, value: Any) -> None:
        self._s.point()
        super().__setitem__(index, value)
        self._s.log("store", value)


def explore(
    scenario: Callable[[Scheduler], tuple[dict[str, Callable[[], None]], Callable[[Scheduler], None]]],
    max_schedules: int = 20_000,
) -> int:
    """Run *scenario* under every schedule. It returns the thread bodies and a check that
    runs after they finish. Returns the number of schedules explored; raises
    ``AssertionError`` on the first failing schedule, with its events."""
    prefix: list[int] = []
    explored = 0
    while True:
        scheduler = Scheduler(prefix)
        bodies, check = scenario(scheduler)
        try:
            scheduler.run(bodies)
            check(scheduler)
        except (AssertionError, Deadlock) as error:
            events = [(e.thread, e.kind) for e in scheduler.events]
            raise AssertionError(f"schedule {scheduler.choices} failed: {error!r}; events {events}") from error
        explored += 1
        if explored >= max_schedules:
            raise AssertionError(f"more than {max_schedules} schedules; bound the scenario")
        nxt = _next_prefix(scheduler.choices, scheduler.enabled_counts)
        if nxt is None:
            return explored
        prefix = nxt


def _next_prefix(choices: list[int], counts: list[int]) -> list[int] | None:
    for depth in range(len(choices) - 1, -1, -1):
        if choices[depth] + 1 < counts[depth]:
            return choices[:depth] + [choices[depth] + 1]
    return None
