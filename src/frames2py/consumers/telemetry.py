"""Telemetry consumer -- periodic engine health monitoring.

Polls :attr:`Engine.stats` and the engine's
:class:`~frames2py.core.engine.LatencyTracker` at a configurable
interval and stores a rolling history of
:class:`TelemetrySample` records.  This data drives dashboards,
log files, and the ``StatsOverlay``.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frames2py.consumers.viewer import Viewer
    from frames2py.core.engine import Engine


@dataclasses.dataclass(frozen=True, slots=True)
class TelemetrySample:
    """A single point-in-time snapshot of engine and consumer health.

    Attributes:
        wall_time_ns: ``time.monotonic_ns()`` when this sample was taken.
        events_ingested: Cumulative events written to the ring buffer.
        events_dropped: Cumulative events lost to overflow.
        chunks_dropped: Cumulative chunk evictions.
        buffer_fill_ratio: Ring-buffer occupancy in ``[0.0, 1.0]``.
        snapshots_published: Cumulative snapshots published.
        viewer_frames_shown: Viewer frame count, or ``None`` if no viewer.
        viewer_frames_dropped: Viewer skip count, or ``None`` if no viewer.
        accumulate_ms_p50: 50th percentile accumulate latency (ms).
        accumulate_ms_p95: 95th percentile accumulate latency (ms).
        accumulate_ms_p99: 99th percentile accumulate latency (ms).
    """

    wall_time_ns: int = 0
    events_ingested: int = 0
    events_dropped: int = 0
    chunks_dropped: int = 0
    buffer_fill_ratio: float = 0.0
    snapshots_published: int = 0
    viewer_frames_shown: int | None = None
    viewer_frames_dropped: int | None = None
    accumulate_ms_p50: float = 0.0
    accumulate_ms_p95: float = 0.0
    accumulate_ms_p99: float = 0.0


class Telemetry:
    """Periodic engine health monitor running in a daemon thread.

    Parameters:
        engine: The :class:`~frames2py.core.engine.Engine` to monitor.
        poll_interval_ms: Milliseconds between successive polls.
        viewer: Optional :class:`~frames2py.consumers.viewer.Viewer`
            to collect viewer-side metrics from.

    Example::

        tel = Telemetry(engine, poll_interval_ms=1000).start()
        # ... run pipeline ...
        for sample in tel.history:
            print(sample)
        tel.stop()
    """

    def __init__(
        self,
        engine: Engine,
        poll_interval_ms: float = 1000.0,
        viewer: Viewer | None = None,
    ) -> None:
        self._engine = engine
        self._interval_s = poll_interval_ms / 1000.0
        self._viewer = viewer
        self._history: list[TelemetrySample] = []
        self._lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> Telemetry:
        """Start the telemetry daemon thread.  Returns ``self`` for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="frames2py-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the telemetry thread to exit and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def history(self) -> list[TelemetrySample]:
        """Thread-safe copy of all collected samples."""
        with self._lock:
            return list(self._history)

    @property
    def latest(self) -> TelemetrySample | None:
        """Most recent sample, or ``None`` if none collected yet."""
        with self._lock:
            return self._history[-1] if self._history else None

    def _collect(self) -> TelemetrySample:
        """Take a single telemetry sample from the engine and viewer."""
        stats = self._engine.stats
        lt = self._engine.latency_tracker

        viewer_shown: int | None = None
        viewer_dropped: int | None = None
        if self._viewer is not None:
            viewer_shown = self._viewer.frames_shown
            viewer_dropped = self._viewer.frames_dropped

        return TelemetrySample(
            wall_time_ns=time.monotonic_ns(),
            events_ingested=stats.events_ingested,
            events_dropped=stats.events_dropped,
            chunks_dropped=stats.chunks_dropped,
            buffer_fill_ratio=stats.buffer_fill_ratio,
            snapshots_published=stats.snapshots_published,
            viewer_frames_shown=viewer_shown,
            viewer_frames_dropped=viewer_dropped,
            accumulate_ms_p50=lt.percentile(50),
            accumulate_ms_p95=lt.percentile(95),
            accumulate_ms_p99=lt.percentile(99),
        )

    # ------------------------------------------------------------------
    # Main telemetry loop (runs in daemon thread)
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set() and self._engine.running:
            sample = self._collect()
            with self._lock:
                self._history.append(sample)
            self._stop_event.wait(timeout=self._interval_s)
