"""The frames2py accumulation engine.

The engine is the central product of frames2py.  It receives event
batches via :meth:`ingest`, accumulates them through a pluggable kernel,
and publishes frame snapshots via a seqlock bridge that consumers
(viewer, recorder, telemetry) poll independently.

Architecture (three decoupled planes):

    Plane A -- Ingest + Accumulate (this module)
        ``ingest()`` → ring buffer write → kernel accumulate → seqlock publish.
        **Never blocks.**

    Plane B -- Snapshot Bridge (seqlock)
        Double-buffered, lock-free publication of (frame, metadata) pairs.

    Plane C -- Consumers (viewer, recorder, telemetry)
        Each polls ``latest_snapshot()`` at its own rate.  The engine
        has zero knowledge of consumers.

See ``docs/spec/engine.md`` for the formal semantics.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frames2py.core.policy import OverflowCounters
from frames2py.core.transport.ring_buffer import ChunkedRingBuffer
from frames2py.core.transport.seqlock import Seqlock
from frames2py.core.types import (
    BatchMeta,
    EngineStats,
    EventBatch,
    FrameView,
    OverflowPolicy,
    SnapshotMeta,
    validate_event_batch,
)
from frames2py.kernels.base import Kernel, get_kernel

# Ensure NumPy kernels are registered before any get_kernel() call.
import frames2py.kernels.numpy_kernels as _nk  # noqa: F401


class LatencyTracker:
    """Rolling-window tracker for accumulate-call durations.

    Stores the last *window_size* durations (in nanoseconds) and computes
    percentiles on demand.  Used by the :class:`Telemetry` consumer.
    """

    __slots__ = ("_window", "_idx", "_count", "_capacity")

    def __init__(self, window_size: int = 10_000) -> None:
        self._capacity = window_size
        self._window = np.zeros(window_size, dtype=np.int64)
        self._idx = 0
        self._count = 0

    def record(self, duration_ns: int) -> None:
        """Record a single duration measurement."""
        self._window[self._idx] = duration_ns
        self._idx = (self._idx + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    def percentile(self, q: float) -> float:
        """Return the *q*-th percentile (0-100) in **milliseconds**.

        Returns 0.0 if no measurements have been recorded.
        """
        if self._count == 0:
            return 0.0
        data = self._window[: self._count]
        return float(np.percentile(data, q)) / 1_000_000.0

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._window[:] = 0
        self._idx = 0
        self._count = 0

    @property
    def count(self) -> int:
        """Number of measurements recorded (up to window size)."""
        return self._count


class Engine:
    """Non-blocking event-to-frame accumulation engine.

    Parameters:
        sensor_size: ``(width, height)`` of the event sensor.
        kernel: Kernel name (``"event_count"``, ``"polarity"``,
            ``"time_surface"``, ``"exp_decay"``) or a pre-constructed
            :class:`Kernel` instance.
        buffer_capacity: Number of chunk slots in the ring buffer.
        chunk_size: Maximum events per chunk slot.
        overflow_policy: Strategy when the ring buffer is full.
        snapshot_interval_ms: Minimum milliseconds between automatic
            snapshots.  ``0`` means snapshot on every ``ingest`` call.
        frame_dtype: NumPy dtype for the output frame.

    Example::

        engine = Engine(sensor_size=(1280, 720), kernel="event_count")
        engine.ingest(events)
        result = engine.latest_snapshot()
        if result is not None:
            frame, meta = result
    """

    def __init__(
        self,
        sensor_size: tuple[int, int],
        kernel: str | Kernel = "event_count",
        buffer_capacity: int = 64,
        chunk_size: int = 65_536,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        snapshot_interval_ms: float = 0.0,
        frame_dtype: np.dtype = np.dtype(np.float32),
    ) -> None:
        self._sensor_size = sensor_size
        self._frame_dtype = frame_dtype
        self._snapshot_interval_ns = int(snapshot_interval_ms * 1_000_000)
        self._last_snapshot_ns: int = 0
        self._running = True
        self._start_ns = time.monotonic_ns()

        # Resolve kernel
        self._kernel: Kernel = get_kernel(kernel)
        self._state = self._kernel.init_state(sensor_size, frame_dtype)

        # Transport
        self._ring = ChunkedRingBuffer(
            capacity=buffer_capacity,
            chunk_size=chunk_size,
            overflow_policy=overflow_policy,
        )
        width, height = sensor_size
        channels = self._kernel.channels
        frame_shape: tuple[int, ...] = (
            (height, width, channels) if channels > 1 else (height, width)
        )
        # TimeSurface uses float64 internally
        actual_dtype = frame_dtype
        if self._kernel.name == "time_surface":
            actual_dtype = np.dtype(np.float64)
        self._seqlock = Seqlock(frame_shape, actual_dtype)

        # Counters
        self._counters = OverflowCounters()
        self._latency = LatencyTracker()
        self._snapshots_published = 0

    # ------------------------------------------------------------------
    # Ingest -- the hot path.  Must never block.
    # ------------------------------------------------------------------
    def ingest(
        self,
        events: EventBatch,
        meta: BatchMeta | None = None,
    ) -> None:
        """Ingest an event batch: buffer → accumulate → publish.

        This is the only entry point for event data.  It:

        1. Validates and writes *events* to the ring buffer.
        2. Drains the ring buffer through the accumulation kernel.
        3. Publishes a new snapshot via seqlock (if interval elapsed).
        4. Updates telemetry counters.

        **Never blocks.**  If the ring buffer is full, the oldest chunk(s)
        are evicted and the drop counters are incremented.

        Parameters:
            events: 1-D structured array with fields ``t``, ``x``, ``y``,
                ``p``.  The caller's array is never modified.
            meta: Optional adapter metadata.
        """
        if not self._running:
            return

        events = validate_event_batch(events)
        if len(events) == 0:
            return

        # 1. Write to ring buffer
        events_dropped = self._ring.write(events)
        self._counters.record_ingest(len(events))
        if events_dropped > 0:
            self._counters.record_drop(events_dropped)

        # 2. Accumulate directly from ring buffer chunks (zero-copy).
        t0 = time.perf_counter_ns()
        for chunk in self._ring.iter_chunks():
            self._kernel.accumulate(chunk, self._state)
        t1 = time.perf_counter_ns()
        self._latency.record(t1 - t0)

        # 3. Publish snapshot if interval elapsed
        now_ns = time.monotonic_ns()
        if self._snapshot_interval_ns == 0 or (
            now_ns - self._last_snapshot_ns >= self._snapshot_interval_ns
        ):
            self._publish_snapshot()
            self._last_snapshot_ns = now_ns

    def _publish_snapshot(self) -> None:
        """Copy kernel state into the seqlock frame buffer."""
        write_buf = self._seqlock.begin_write()
        snap_meta = self._kernel.snapshot(self._state, write_buf)
        # Override seq with engine-level counter for monotonicity.
        self._snapshots_published += 1
        snap_meta = SnapshotMeta(
            timestamp=snap_meta.timestamp,
            seq=self._snapshots_published,
            events_accumulated=snap_meta.events_accumulated,
            events_dropped=self._counters.events_dropped,
            wall_time_ns=time.monotonic_ns(),
        )
        self._seqlock.end_write(snap_meta)

    # ------------------------------------------------------------------
    # Consumer interface
    # ------------------------------------------------------------------
    def latest_snapshot(self) -> tuple[FrameView, SnapshotMeta] | None:
        """Read the most recently published snapshot.

        Returns:
            ``(frame_copy, meta)`` if available, or ``None`` if no
            snapshot has been published yet or a write was in progress.

        Thread-safe: may be called from any number of consumer threads
        concurrently.
        """
        return self._seqlock.try_read()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    @property
    def stats(self) -> EngineStats:
        """Live engine statistics.  Always available, zero-cost read."""
        return EngineStats(
            events_ingested=self._counters.events_ingested,
            events_dropped=self._counters.events_dropped,
            chunks_dropped=self._counters.chunks_dropped,
            buffer_fill_ratio=self._ring.fill_ratio,
            snapshots_published=self._snapshots_published,
            uptime_ns=time.monotonic_ns() - self._start_ns,
        )

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Access to per-call accumulate latency measurements."""
        return self._latency

    @property
    def running(self) -> bool:
        """``True`` while the engine is accepting events."""
        return self._running

    @property
    def sensor_size(self) -> tuple[int, int]:
        """``(width, height)`` of the sensor."""
        return self._sensor_size

    @property
    def kernel(self) -> Kernel:
        """The active accumulation kernel."""
        return self._kernel

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all state: ring buffer, kernel, counters.

        The engine remains running and ready to accept new events.
        """
        self._ring.clear()
        self._kernel.reset(self._state)
        self._seqlock.reset()
        self._counters.reset()
        self._latency.reset()
        self._snapshots_published = 0
        self._last_snapshot_ns = 0
        self._start_ns = time.monotonic_ns()

    def stop(self) -> None:
        """Signal all consumers to stop polling.

        After this call, :meth:`ingest` becomes a no-op and consumers
        checking :attr:`running` will exit their loops.
        """
        self._running = False

    def start(self) -> None:
        """Re-enable the engine after a :meth:`stop`."""
        self._running = True
        self._start_ns = time.monotonic_ns()
