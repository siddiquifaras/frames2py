"""Python wrappers for C++ (pybind11) accumulation kernels.

Each wrapper implements the :class:`~frames2py.kernels.base.Kernel`
protocol and delegates to ``_frames2py_native`` for the hot path.
If the native module is not available (not compiled), the wrapper
**falls back** to the corresponding NumPy kernel with a one-time
warning.  The engine is agnostic to which backend is running.

Build the native module::

    cd native && cmake -B build && cmake --build build
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import EVENT_DTYPE, EventBatch, FrameView, SnapshotMeta
from frames2py.kernels.base import register_kernel

try:
    import _frames2py_native as _native  # type: ignore[import-not-found]

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


@dataclasses.dataclass(slots=True)
class _NativeState:
    """State container for native kernels."""

    buf: NDArray[np.floating[Any]]
    width: int
    height: int
    events_accumulated: int = 0
    latest_t: int = 0
    snapshot_seq: int = 0


def _warn_fallback(kernel_name: str) -> None:
    warnings.warn(
        f"Native module not available; {kernel_name} falling back to "
        f"NumPy backend.  Build the C++ module for full performance: "
        f"cd native && cmake -B build && cmake --build build",
        RuntimeWarning,
        stacklevel=3,
    )


# ===================================================================
# NativeEventCountKernel
# ===================================================================
class NativeEventCountKernel:
    """C++ event-count kernel with NumPy fallback.

    Identical semantics to
    :class:`~frames2py.kernels.numpy_kernels.EventCountKernel` but
    delegates ``accumulate`` to the pybind11 module with GIL released.
    """

    def __init__(self) -> None:
        self._fallback: Any | None = None
        if not _HAS_NATIVE:
            _warn_fallback("NativeEventCountKernel")
            from frames2py.kernels.numpy_kernels import EventCountKernel

            self._fallback = EventCountKernel()

    @property
    def name(self) -> str:
        return "native_event_count"

    @property
    def channels(self) -> int:
        return 1

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype = np.dtype(np.float32),
    ) -> Any:
        if self._fallback is not None:
            return self._fallback.init_state(sensor_size, frame_dtype)
        w, h = sensor_size
        return _NativeState(buf=np.zeros((h, w), dtype=np.float32), width=w, height=h)

    def accumulate(self, events: EventBatch, state: Any) -> None:
        if self._fallback is not None:
            return self._fallback.accumulate(events, state)
        if len(events) == 0:
            return
        _native.accumulate_event_count(events, state.buf, state.width, state.height)
        state.events_accumulated += len(events)
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: Any, out: FrameView) -> SnapshotMeta:
        if self._fallback is not None:
            return self._fallback.snapshot(state, out)
        np.copyto(out, state.buf)
        state.buf[:] = 0
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: Any) -> None:
        if self._fallback is not None:
            return self._fallback.reset(state)
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ===================================================================
# NativePolarityKernel
# ===================================================================
class NativePolarityKernel:
    """C++ polarity kernel with NumPy fallback."""

    def __init__(self) -> None:
        self._fallback: Any | None = None
        if not _HAS_NATIVE:
            _warn_fallback("NativePolarityKernel")
            from frames2py.kernels.numpy_kernels import PolarityKernel

            self._fallback = PolarityKernel()

    @property
    def name(self) -> str:
        return "native_polarity"

    @property
    def channels(self) -> int:
        return 2

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype = np.dtype(np.float32),
    ) -> Any:
        if self._fallback is not None:
            return self._fallback.init_state(sensor_size, frame_dtype)
        w, h = sensor_size
        return _NativeState(buf=np.zeros((h, w, 2), dtype=np.float32), width=w, height=h)

    def accumulate(self, events: EventBatch, state: Any) -> None:
        if self._fallback is not None:
            return self._fallback.accumulate(events, state)
        if len(events) == 0:
            return
        _native.accumulate_polarity(events, state.buf, state.width, state.height)
        state.events_accumulated += len(events)
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: Any, out: FrameView) -> SnapshotMeta:
        if self._fallback is not None:
            return self._fallback.snapshot(state, out)
        np.copyto(out, state.buf)
        state.buf[:] = 0
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: Any) -> None:
        if self._fallback is not None:
            return self._fallback.reset(state)
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ===================================================================
# NativeTimeSurfaceKernel
# ===================================================================
class NativeTimeSurfaceKernel:
    """C++ time-surface kernel with NumPy fallback."""

    def __init__(self) -> None:
        self._fallback: Any | None = None
        if not _HAS_NATIVE:
            _warn_fallback("NativeTimeSurfaceKernel")
            from frames2py.kernels.numpy_kernels import TimeSurfaceKernel

            self._fallback = TimeSurfaceKernel()

    @property
    def name(self) -> str:
        return "native_time_surface"

    @property
    def channels(self) -> int:
        return 1

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype = np.dtype(np.float64),
    ) -> Any:
        if self._fallback is not None:
            return self._fallback.init_state(sensor_size, frame_dtype)
        w, h = sensor_size
        return _NativeState(buf=np.zeros((h, w), dtype=np.float64), width=w, height=h)

    def accumulate(self, events: EventBatch, state: Any) -> None:
        if self._fallback is not None:
            return self._fallback.accumulate(events, state)
        if len(events) == 0:
            return
        _native.accumulate_time_surface(events, state.buf, state.width, state.height)
        state.events_accumulated += len(events)
        state.latest_t = max(state.latest_t, int(events["t"][-1]))

    def snapshot(self, state: Any, out: FrameView) -> SnapshotMeta:
        if self._fallback is not None:
            return self._fallback.snapshot(state, out)
        np.copyto(out, state.buf)
        state.snapshot_seq += 1
        return SnapshotMeta(
            timestamp=state.latest_t,
            seq=state.snapshot_seq,
            events_accumulated=state.events_accumulated,
            wall_time_ns=time.monotonic_ns(),
        )

    def reset(self, state: Any) -> None:
        if self._fallback is not None:
            return self._fallback.reset(state)
        state.buf[:] = 0
        state.events_accumulated = 0
        state.latest_t = 0


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
register_kernel("native_event_count", NativeEventCountKernel)
register_kernel("native_polarity", NativePolarityKernel)
register_kernel("native_time_surface", NativeTimeSurfaceKernel)
