"""Core data types for frames2py.

Defines the canonical event layout, metadata containers, engine statistics,
and overflow policy. These types form the shared vocabulary across all modules:
engine, kernels, transport, consumers, and adapters.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EVENT_DTYPE",
    "EventBatch",
    "FrameView",
    "KernelState",
    "OverflowPolicy",
    "BatchMeta",
    "SnapshotMeta",
    "EngineStats",
    "validate_event_batch",
]

# ---------------------------------------------------------------------------
# Canonical event dtype -- 13 bytes per event.
# Fields: t (uint64 us), x (uint16), y (uint16), p (uint8 polarity 0/1).
# ---------------------------------------------------------------------------
EVENT_DTYPE: Final = np.dtype(
    [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")],
    align=False,
)

EventBatch = NDArray[np.void]
"""1-D structured array with at least fields ``t``, ``x``, ``y``, ``p``."""

FrameView = NDArray[np.floating[Any]]
"""2-D or 3-D float array written to by kernels and read by consumers."""

KernelState = Any
"""Opaque accumulator state owned by each kernel implementation."""


# ---------------------------------------------------------------------------
# Overflow policy
# ---------------------------------------------------------------------------
class OverflowPolicy(enum.Enum):
    """Ring-buffer overflow strategy.

    Only drop policies are supported.  There is **no** ``BLOCK`` option --
    blocking would re-introduce the exact latency coupling that frames2py
    exists to eliminate (see invariants.md #4).
    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


# ---------------------------------------------------------------------------
# Frozen metadata containers
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, slots=True)
class BatchMeta:
    """Optional metadata attached to an event batch by adapters.

    Attributes:
        monotonic: ``True`` if the adapter guarantees that timestamps in the
            batch are non-decreasing.
        reordered: ``True`` if the adapter sorted or reordered events to
            achieve monotonicity.
        source: Human-readable identifier for the data source,
            e.g. ``"prophesee"``, ``"h5"``, ``"udp"``.
        sensor_size: ``(width, height)`` if the adapter knows the sensor
            resolution, else ``None``.
    """

    monotonic: bool = False
    reordered: bool = False
    source: str = ""
    sensor_size: tuple[int, int] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotMeta:
    """Metadata published alongside each snapshot frame.

    Returned by :meth:`Engine.latest_snapshot` and carried through to
    consumers (viewer, recorder, telemetry).

    Attributes:
        timestamp: Latest event timestamp (microseconds) in this snapshot.
        seq: Monotonically increasing publication sequence number.
        events_accumulated: Total events that contributed to this snapshot
            since the last reset.
        events_dropped: Events dropped since the *previous* snapshot.
        wall_time_ns: Wall-clock time (``time.monotonic_ns``) when the
            snapshot was published.
    """

    timestamp: int = 0
    seq: int = 0
    events_accumulated: int = 0
    events_dropped: int = 0
    wall_time_ns: int = dataclasses.field(default_factory=time.monotonic_ns)


@dataclasses.dataclass(slots=True)
class EngineStats:
    """Live telemetry counters exposed by :attr:`Engine.stats`.

    All counters are monotonically non-decreasing and are maintained
    unconditionally -- even when no :class:`Telemetry` consumer is attached
    (see invariants.md #11).

    Attributes:
        events_ingested: Total events written to the ring buffer.
        events_dropped: Total events lost to overflow.
        chunks_dropped: Total chunks evicted from the ring buffer.
        buffer_fill_ratio: Current ring-buffer occupancy in ``[0.0, 1.0]``.
        snapshots_published: Total snapshots published via seqlock.
        uptime_ns: Nanoseconds since engine construction.
    """

    events_ingested: int = 0
    events_dropped: int = 0
    chunks_dropped: int = 0
    buffer_fill_ratio: float = 0.0
    snapshots_published: int = 0
    uptime_ns: int = 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_event_batch(events: NDArray[Any]) -> EventBatch:
    """Validate that *events* is a conforming ``EventBatch``.

    Checks:
    - 1-D structured array
    - C-contiguous
    - Has required fields ``t``, ``x``, ``y``, ``p``

    Returns:
        The same array (no copy) if valid.

    Raises:
        TypeError: If *events* is not a structured NumPy array.
        ValueError: If required fields are missing, array is not 1-D, or
            array is not C-contiguous.
    """
    if not isinstance(events, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(events).__name__}"
        )
    if events.dtype.names is None:
        raise TypeError(
            "Expected a structured array with named fields, "
            f"got dtype {events.dtype}"
        )
    missing = {"t", "x", "y", "p"} - set(events.dtype.names)
    if missing:
        raise ValueError(
            f"Event array missing required fields: {sorted(missing)}"
        )
    if events.ndim != 1:
        raise ValueError(
            f"Event array must be 1-D, got {events.ndim}-D with "
            f"shape {events.shape}"
        )
    if not events.flags["C_CONTIGUOUS"]:
        raise ValueError("Event array must be C-contiguous")
    return events  # type: ignore[return-value]
