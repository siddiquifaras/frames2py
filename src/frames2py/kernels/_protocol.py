"""The Kernel protocol: how a representation plugs into the Accumulator."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

KernelState = Any  # opaque: whatever init_state returns, passed back unchanged


@runtime_checkable
class Kernel(Protocol):
    """A representation kernel.

    The Accumulator owns validation, the timestamp-range check, the bounds check, the
    watermark and the out-of-bounds count; a kernel only turns in-bounds events into
    its representation. For each accepted call the Accumulator calls ``begin_call``
    once, then passes the call's in-bounds events to ``accumulate``. Publication reads
    the representation, then closes the window.
    """

    name: str

    def output_spec(self, sensor_size: tuple[int, int]) -> tuple[tuple[int, ...], np.dtype[Any]]:
        """Shape and dtype of the representation for a ``(width, height)`` sensor."""
        ...

    def init_state(self, sensor_size: tuple[int, int]) -> KernelState:
        """Allocate the kernel's state for a ``(width, height)`` sensor."""
        ...

    def begin_call(self, state: KernelState) -> None:
        """Once per accepted accumulate/ingest call, before any accumulation."""
        ...

    def accumulate(self, events: NDArray[np.void], state: KernelState, watermark: int | None) -> None:
        """Merge the call's in-bounds events. ``watermark`` includes this call's events."""
        ...

    def read(self, state: KernelState, out: NDArray[Any], watermark: int | None) -> None:
        """Write the representation, evaluated at ``watermark``, into ``out``.

        ``watermark`` is the latest accumulated watermark or a later time; ``None``
        only before the first in-bounds event. The Accumulator and Engine pass the
        accumulated watermark. Must not change the state, whatever ``watermark`` is.
        Kernels without time dependence ignore it.

        Since 1.1, ``read`` may be given a time later than the accumulated watermark.
        A kernel written for 1.0, which assumed it always received exactly the
        accumulated watermark, may need adapting. ``frames2py.replay.windows()`` passes
        each frame's boundary, later than every event accumulated, and requires a kernel
        that evaluates its representation at the time it is given.
        """
        ...

    def close_window(self, state: KernelState) -> None:
        """Called after each publication. Windowed kernels start a new window here;
        running kernels do nothing."""
        ...

    def reset(self, state: KernelState) -> None:
        """Return the state to what ``init_state`` produced."""
        ...
