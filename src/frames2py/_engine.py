"""The Engine: live accumulation with publication to any number of consumers."""

from __future__ import annotations

import time
from typing import Any

from numpy.typing import NDArray

from frames2py._accumulator import Accumulator
from frames2py._events import EventArray
from frames2py._types import EngineStats, SnapshotMeta
from frames2py.kernels import Kernel
from frames2py.publish import SeqlockPublisher, SnapshotPublisher


class Engine:
    """Accumulates events from one producer and publishes snapshots for consumers.

    ``ingest()`` does its work on the caller's thread and never waits for a consumer.
    Consumers call ``snapshot()`` whenever they like. Publication happens only inside
    ``ingest()`` and ``stop()``: there is no timer thread.

    Call ``ingest()``, ``start()``, ``stop()`` and ``reset()`` from one producer thread.

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
        self._accumulator = Accumulator(sensor_size, kernel)
        shape, dtype = self._accumulator._output_spec
        self._publisher: SnapshotPublisher = SeqlockPublisher(shape, dtype)
        self._interval_ns = snapshot_interval_ms * 1e6
        self._created_ns = time.monotonic_ns()
        self._running = True
        self._sequence = 0
        self._events_ingested = 0
        self._snapshots_published = 0
        self._pending = False
        self._last_published_ns: int | None = None

    def ingest(self, events: EventArray) -> None:
        """Accumulate one call's events, then publish if the interval allows.

        A no-op while stopped. Otherwise raises ``TypeError`` for a malformed array and
        ``ValueError`` if any event has ``t >= 2**63``, in both cases before any state
        or statistic changes.
        """
        if not self._running:
            return
        inside = self._accumulator._accumulate(events)
        self._events_ingested += len(events)
        if inside:
            self._pending = True
        now = time.monotonic_ns()
        if self._last_published_ns is None or now - self._last_published_ns >= self._interval_ns:
            self._publish(now)

    def snapshot(self) -> tuple[NDArray[Any], SnapshotMeta] | None:
        """A copy of the latest published frame and its metadata, or ``None`` before the
        first publication and after ``reset()`` until the next."""
        return self._publisher.read()

    @property
    def stats(self) -> EngineStats:
        """Counters at the time of the call."""
        return EngineStats(
            events_ingested=self._events_ingested,
            events_out_of_bounds=self._accumulator.events_out_of_bounds,
            snapshots_published=self._snapshots_published,
            uptime_ns=time.monotonic_ns() - self._created_ns,
        )

    def start(self) -> None:
        """Resume ingestion after ``stop()``."""
        self._running = True

    def stop(self) -> None:
        """Publish events accumulated since the last publication, if any, then make
        ``ingest()`` a no-op. The latest snapshot stays readable."""
        if self._running and self._pending:
            self._publish(time.monotonic_ns())
        self._running = False

    def reset(self) -> None:
        """Clear the kernel state, counters, watermark and published snapshot.

        The snapshot sequence and ``uptime_ns`` carry on. The next ``ingest()``
        publishes, as the first one does.
        """
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
        self._accumulator._close_window()
        self._snapshots_published += 1
        self._pending = False
        self._last_published_ns = now_ns
