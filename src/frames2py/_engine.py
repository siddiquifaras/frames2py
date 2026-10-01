"""The Engine: live accumulation with publication to any number of consumers."""

from __future__ import annotations

import collections
import math
import numbers
import sys
import threading
import time
from typing import Any, Final

from frames2py._accumulator import Accumulator
from frames2py._events import EventArray
from frames2py._types import EngineStats, Snapshot, SnapshotMeta
from frames2py.kernels import Kernel
from frames2py.publish import ImmutablePublisher, SnapshotPublisher

_VERIFIED_FREE_THREADED: Final = frozenset({(3, 14)})
"""Minor versions whose free-threaded build the publisher's handoff has been verified on."""

_allocate_lock = threading.Lock


def _sequence_floor(sequence: object) -> int:
    """The sequence a snapshot must exceed to be newer: *sequence*, or -1 for ``None``."""
    if sequence is None:
        return -1
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise TypeError(f"sequence must be an int or None, got {type(sequence).__name__}")
    if sequence < 0:
        raise ValueError(f"sequence must be >= 0, got {sequence}")
    return sequence


def _deadline(timeout: object) -> float | None:
    """The monotonic deadline for *timeout* seconds, or ``None`` for no limit."""
    if timeout is None:
        return None
    if not isinstance(timeout, numbers.Real):
        raise TypeError(f"timeout must be a real number or None, got {type(timeout).__name__}")
    if timeout != timeout:
        raise ValueError("timeout must not be NaN")
    if timeout < 0:
        raise ValueError(f"timeout must be >= 0, got {timeout}")
    try:
        seconds = float(timeout)
    except OverflowError:  # an int beyond float range: no deadline that could ever pass
        return None
    return None if seconds == math.inf else time.monotonic() + seconds


def _check_runtime() -> None:
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if not gil_enabled and tuple(sys.version_info[:2]) not in _VERIFIED_FREE_THREADED:
        verified = ", ".join(f"{major}.{minor}t" for major, minor in sorted(_VERIFIED_FREE_THREADED))
        raise RuntimeError(
            f"Engine is not supported on free-threaded Python {sys.version_info[0]}.{sys.version_info[1]} "
            f"with the GIL disabled; verified free-threaded versions: {verified}"
        )


class Engine:
    """Accumulates events from one producer and publishes snapshots for consumers.

    ``ingest()`` does its work on the caller's thread and never waits for a consumer.
    Consumers call ``snapshot()`` whenever they like, or ``wait_for_newer()`` to block until
    a newer snapshot is published; ``snapshot()`` and ``stats`` take no lock. Publication
    happens only inside ``ingest()`` and ``stop()``: there is no timer thread.

    The first ``ingest()`` that doesn't raise makes its thread the producer for the
    Engine's lifetime, ``reset()`` included; ``ingest()`` from any other thread raises
    ``RuntimeError``. ``start()``, ``stop()`` and ``reset()`` may be called from any thread:
    they and ``ingest()`` take one lock, so each runs whole, never interleaved with another.

    Raises ``RuntimeError`` on construction on a free-threaded build with the GIL disabled,
    unless that minor version has been verified (3.14).

    Args:
        sensor_size: ``(width, height)``. Frames are ``(height, width[, channels])``.
        kernel: A kernel instance, or ``"event_count"``, ``"polarity"`` or
            ``"time_surface"``.
        snapshot_interval_ms: ``0`` publishes on every ``ingest()``. A positive
            interval publishes at most once per interval, on the first ``ingest()``
            after it elapses. The first ``ingest()`` always publishes.
    """

    def __init__(
        self,
        sensor_size: tuple[int, int],
        kernel: str | Kernel = "event_count",
        *,
        snapshot_interval_ms: float = 16.0,
    ) -> None:
        _check_runtime()
        self._accumulator = Accumulator(sensor_size, kernel)
        shape, dtype = self._accumulator._output_spec
        self._publisher: SnapshotPublisher = ImmutablePublisher(shape, dtype)
        self._interval_ns = snapshot_interval_ms * 1e6
        self._created_ns = time.monotonic_ns()
        self._lifecycle = threading.Lock()
        self._producer: int | None = None
        self._running = True
        self._sequence = 0
        self._events_ingested = 0
        self._snapshots_published = 0
        self._pending = False
        self._last_published_ns: int | None = None
        self._waiters: collections.deque[Any] = collections.deque()

    def ingest(self, events: EventArray) -> None:
        """Accumulate one call's events, then publish if the interval allows.

        A no-op while stopped. Raises ``RuntimeError`` if called from a thread other than
        the producer's. Otherwise raises ``TypeError`` for a malformed array and
        ``ValueError`` if any event has ``t >= 2**63``, in both cases before any state or
        statistic changes.
        """
        caller = threading.get_ident()
        with self._lifecycle:
            if self._producer is not None and caller != self._producer:
                raise RuntimeError("ingest() called from a thread other than the producer's")
            if not self._running:
                self._producer = caller
                return
            inside = self._accumulator._accumulate(events)
            self._producer = caller
            self._events_ingested += len(events)
            if inside:
                self._pending = True
            now = time.monotonic_ns()
            if self._last_published_ns is None or now - self._last_published_ns >= self._interval_ns:
                self._publish(now)

    def snapshot(self) -> Snapshot | None:
        """The latest published snapshot, shared, not copied; or ``None`` before the first
        publication and after ``reset()`` until the next. Takes no lock."""
        return self._publisher.read()

    def wait_for_newer(self, sequence: int | None, *, timeout: float | None = None) -> Snapshot | None:
        """Block until a snapshot newer than *sequence* is published, then return the latest.

        A snapshot is newer when its ``meta.sequence`` is greater than *sequence*; with
        ``None``, any published snapshot is. If one is already published the call returns it
        at once. Otherwise it returns the snapshot published at the moment it reads it after a
        publication, which needn't be ``sequence + 1``: publications in between may be skipped.
        Passing back the returned ``meta.sequence`` gives strictly increasing snapshots.

        *timeout* is in seconds on the monotonic clock. ``None`` or ``math.inf`` waits without
        limit; ``0`` checks once without blocking; values above ``threading.TIMEOUT_MAX`` are
        honoured. On timeout the call returns ``None``, and only if no newer snapshot is
        published when it checks after the timeout has elapsed.

        ``stop()`` and ``reset()`` wake nobody: the publication ``stop()`` makes for a pending
        window wakes waiters like any other, and after ``reset()`` a waiter returns the first
        publication after it, whose sequence is larger than any before the reset. A reset that
        lands between a publication and the waiter's read leaves the waiter waiting for the
        next one. ``KeyboardInterrupt`` propagates out of a main-thread wait and leaves the
        Engine unchanged. CPython acts on a SIGINT that arrives in the instant before the wait
        blocks only when the wait next wakes: at a publication, at the timeout, or at another
        SIGINT.

        The call never takes the lock that ``ingest()`` holds. Each publication releases the
        waiters registered before it, one lock release each, on the publishing thread; it
        never waits for a waiter.

        Raises:
            TypeError: *sequence* is not an int or ``None`` (a bool is refused), or *timeout*
                is not a real number or ``None``.
            ValueError: *sequence* is negative, or *timeout* is negative or NaN.
            RuntimeError: called on the producer's thread, which can't publish while it waits.
        """
        floor = _sequence_floor(sequence)
        deadline = _deadline(timeout)
        if threading.get_ident() == self._producer:
            raise RuntimeError("wait_for_newer() called on the producer's thread")
        while True:
            expired = deadline is not None and time.monotonic() >= deadline
            snapshot = self._publisher.read()
            if snapshot is not None and snapshot.meta.sequence > floor:
                return snapshot
            if expired:
                return None
            waiter = _allocate_lock()
            waiter.acquire()
            self._waiters.append(waiter)
            try:
                # Registered before this read: a publication this read misses releases the lock.
                snapshot = self._publisher.read()
                if snapshot is not None and snapshot.meta.sequence > floor:
                    return snapshot
                if deadline is None:
                    waiter.acquire()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        waiter.acquire(timeout=min(remaining, threading.TIMEOUT_MAX))
            finally:
                try:
                    self._waiters.remove(waiter)
                except ValueError:  # a publication already took it
                    pass

    @property
    def stats(self) -> EngineStats:
        """Counters at the time of the call. Takes no lock."""
        return EngineStats(
            events_ingested=self._events_ingested,
            events_out_of_bounds=self._accumulator.events_out_of_bounds,
            snapshots_published=self._snapshots_published,
            uptime_ns=time.monotonic_ns() - self._created_ns,
        )

    def start(self) -> None:
        """Resume ingestion after ``stop()``."""
        with self._lifecycle:
            self._running = True

    def stop(self) -> None:
        """Publish events accumulated since the last publication, if any, then make
        ``ingest()`` a no-op. The latest snapshot stays readable."""
        with self._lifecycle:
            if self._running and self._pending:
                self._publish(time.monotonic_ns())
            self._running = False

    def reset(self) -> None:
        """Clear the kernel state, counters, watermark and published snapshot.

        The snapshot sequence, ``uptime_ns`` and the producer thread carry on. The next
        ``ingest()`` publishes, as the first one does.
        """
        with self._lifecycle:
            self._accumulator.reset()
            self._publisher.reset()
            self._events_ingested = 0
            self._snapshots_published = 0
            self._pending = False
            self._last_published_ns = None

    def _publish(self, now_ns: int) -> None:
        self._sequence += 1
        buffer = self._publisher.begin_write()
        self._accumulator._read_into(buffer)
        self._publisher.end_write(SnapshotMeta(watermark=self._accumulator.watermark, sequence=self._sequence))
        # Release the waiters registered before this point, and no later ones. The registry
        # operation (in the deque's critical section on free-threaded builds, under the GIL
        # otherwise) orders this against registration on every publication, with or without
        # waiters; an unlocked emptiness check could miss a waiter on free-threaded builds.
        marker = object()
        waiters = self._waiters
        waiters.append(marker)
        while (waiter := waiters.popleft()) is not marker:
            waiter.release()
        self._accumulator._close_window()
        self._snapshots_published += 1
        self._pending = False
        self._last_published_ns = now_ns
