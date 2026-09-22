"""Vectorised NumPy accumulation kernels.

Performance characteristics:
    - EventCount, Polarity, and ExpDecay use ``np.bincount`` for O(n)
      scatter-accumulation with sequential memory access -- typically
      2-5x faster than ``np.add.at`` which does unbuffered random writes.
    - TimeSurface uses flat-indexed ``np.maximum.at`` for 1-D scatter
      (better cache locality than 2-D ``(y, x)`` scatter).
    - Out-of-bounds events are **discarded** (not clamped to edges) to
      prevent edge-pixel corruption in production pipelines.
    - One heap allocation per ``accumulate`` call: the ``bincount``
      result array of size H*W (or H*W*2 for polarity).

All kernels satisfy the :class:`~frames2py.kernels.base.Kernel` protocol.
See ``docs/spec/kernel_contract.md`` for the formal contract.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import EventBatch, FrameView, SnapshotMeta
from frames2py.kernels.base import register_kernel

__all__ = [
    "EventCountKernel",
    "PolarityKernel",
    "TimeSurfaceKernel",
    "ExpDecayKernel",
]


# ------------------------------------------------------------------
# Shared state container
# ------------------------------------------------------------------
@dataclasses.dataclass(slots=True)
class _AccumState:
    """Internal accumulator state shared by all NumPy kernels."""

    buf: NDArray[np.floating[Any]]
    width: int
    height: int
    events_accumulated: int = 0
    latest_t: int = 0
    snapshot_seq: int = 0


def _valid_flat(
    events: EventBatch,
    width: int,
    height: int,
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Flat row-major pixel indices and validity mask for in-bounds events.

    Out-of-bounds events (``x >= width`` or ``y >= height``) are
    excluded from the returned index array.  Since ``x`` and ``y`` are
    unsigned, negative values are impossible.

    Returns:
        ``(flat_indices, valid_mask)`` where ``flat_indices`` has length
        ``valid_mask.sum()`` and contains row-major pixel offsets.

    Complexity: O(n) time, O(n) space.
    """
    x = events["x"]
    y = events["y"]
    valid: NDArray[np.bool_] = (x < width) & (y < height)
    xv = x[valid].astype(np.intp)
    yv = y[valid].astype(np.intp)
    return yv * width + xv, valid


# ===================================================================
# EventCountKernel
# ===================================================================
class EventCountKernel:
    """Counts events per pixel.  Resets on snapshot.

    Uses ``np.bincount`` for O(n + H*W) accumulation with contiguous
    memory access instead of unbuffered ``np.add.at`` scatter writes.

    Output shape: ``(height, width)``, dtype ``float32``.
    """

    @property
    def name(self) -> str:
        return "event_count"

    @property
    def channels(self) -> int:
        return 1

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype[Any] = np.dtype(np.float32),
    ) -> _AccumState:
        w, h = sensor_size
        return _AccumState(buf=np.zeros((h, w), dtype=frame_dtype), width=w, height=h)

    def accumulate(self, events: EventBatch, state: _AccumState) -> None:
        n = len(events)
        if n == 0:
            return
        flat, _ = _valid_flat(events, state.width, state.height)
        if len(flat) > 0:
            hw = state.width * state.height
            counts = np.bincount(flat, minlength=hw)
            state.buf.ravel()[:] += counts.astype(state.buf.dtype)
        state.events_accumulated += n
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: _AccumState, out: FrameView) -> SnapshotMeta:
        np.copyto(out, state.buf)
        state.buf[:] = 0
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: _AccumState) -> None:
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ===================================================================
# PolarityKernel
# ===================================================================
class PolarityKernel:
    """Two-channel accumulation: channel 0 = OFF (p=0), channel 1 = ON (p=1).

    Encodes ``(y, x, p)`` into a single flat index
    ``(y * W + x) * 2 + p`` so the entire accumulation is one
    ``np.bincount`` call.

    Accumulate complexity: O(n + H*W*2).
    Output shape: ``(height, width, 2)``, dtype ``float32``.
    Resets on snapshot.
    """

    @property
    def name(self) -> str:
        return "polarity"

    @property
    def channels(self) -> int:
        return 2

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype[Any] = np.dtype(np.float32),
    ) -> _AccumState:
        w, h = sensor_size
        return _AccumState(
            buf=np.zeros((h, w, 2), dtype=frame_dtype), width=w, height=h
        )

    def accumulate(self, events: EventBatch, state: _AccumState) -> None:
        n = len(events)
        if n == 0:
            return
        x = events["x"]
        y = events["y"]
        p = events["p"]
        w, h = state.width, state.height
        valid: NDArray[np.bool_] = (x < w) & (y < h)
        xv = x[valid].astype(np.intp)
        yv = y[valid].astype(np.intp)
        pv = p[valid].astype(np.intp)
        if len(xv) > 0:
            combined = (yv * w + xv) * 2 + pv
            hw2 = h * w * 2
            counts = np.bincount(combined, minlength=hw2)
            state.buf.ravel()[:] += counts[:hw2].astype(state.buf.dtype)
        state.events_accumulated += n
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: _AccumState, out: FrameView) -> SnapshotMeta:
        np.copyto(out, state.buf)
        state.buf[:] = 0
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: _AccumState) -> None:
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ===================================================================
# TimeSurfaceKernel
# ===================================================================
class TimeSurfaceKernel:
    """Stores the most recent timestamp per pixel.  Does **not** reset.

    Uses flat-indexed ``np.maximum.at`` for cache-friendlier 1-D
    scatter (avoids the 2-D ``(y, x)`` addressing overhead).

    Accumulate complexity: O(n).
    Output shape: ``(height, width)``, dtype ``float64``.
    """

    @property
    def name(self) -> str:
        return "time_surface"

    @property
    def channels(self) -> int:
        return 1

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype[Any] = np.dtype(np.float64),
    ) -> _AccumState:
        w, h = sensor_size
        return _AccumState(buf=np.zeros((h, w), dtype=frame_dtype), width=w, height=h)

    def accumulate(self, events: EventBatch, state: _AccumState) -> None:
        n = len(events)
        if n == 0:
            return
        flat, valid = _valid_flat(events, state.width, state.height)
        if len(flat) > 0:
            t = events["t"][valid].astype(state.buf.dtype)
            np.maximum.at(state.buf.ravel(), flat, t)
        state.events_accumulated += n
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: _AccumState, out: FrameView) -> SnapshotMeta:
        np.copyto(out, state.buf)
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: _AccumState) -> None:
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ===================================================================
# ExpDecayKernel
# ===================================================================
class ExpDecayKernel:
    """Exponential-decay surface.  Does **not** reset on snapshot.

    On each ``accumulate``, all pixels are decayed by a multiplicative
    factor, then event locations are incremented via ``np.bincount``.

    Accumulate complexity: O(H*W + n).
    Output shape: ``(height, width)``, dtype ``float32``.

    Parameters:
        decay: Multiplicative decay factor applied to the entire surface
            before adding new events.  Values close to 1.0 produce
            longer trails.
    """

    def __init__(self, decay: float = 0.95) -> None:
        self._decay: float = decay

    @property
    def name(self) -> str:
        return "exp_decay"

    @property
    def channels(self) -> int:
        return 1

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype[Any] = np.dtype(np.float32),
    ) -> _AccumState:
        w, h = sensor_size
        return _AccumState(buf=np.zeros((h, w), dtype=frame_dtype), width=w, height=h)

    def accumulate(self, events: EventBatch, state: _AccumState) -> None:
        state.buf *= self._decay
        n = len(events)
        if n == 0:
            return
        flat, _ = _valid_flat(events, state.width, state.height)
        if len(flat) > 0:
            hw = state.width * state.height
            counts = np.bincount(flat, minlength=hw)
            state.buf.ravel()[:] += counts.astype(state.buf.dtype)
        state.events_accumulated += n
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: _AccumState, out: FrameView) -> SnapshotMeta:
        np.copyto(out, state.buf)
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: _AccumState) -> None:
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
register_kernel("event_count", EventCountKernel)
register_kernel("polarity", PolarityKernel)
register_kernel("time_surface", TimeSurfaceKernel)
register_kernel("exp_decay", ExpDecayKernel)
