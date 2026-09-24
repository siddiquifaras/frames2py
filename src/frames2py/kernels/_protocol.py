"""The Kernel protocol: how a representation plugs into the Accumulator."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

KernelState = Any
"""Whatever a kernel keeps between calls. The Accumulator stores it and passes it back."""


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

        Must not change the state. ``watermark`` is ``None`` before the first
        in-bounds event; kernels without time dependence ignore it.
        """
        ...

    def close_window(self, state: KernelState) -> None:
        """Called after each publication. Windowed kernels start a new window here;
        running kernels do nothing."""
        ...

    def reset(self, state: KernelState) -> None:
        """Return the state to what ``init_state`` produced."""
        ...
