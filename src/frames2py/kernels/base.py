"""Kernel protocol and factory for accumulation backends.

A kernel transforms a stream of events into a 2-D (or 3-D) frame by
implementing two operations:

- **accumulate**: merge new events into an internal state buffer.
- **snapshot**: copy the current state into an output frame, optionally
  resetting the accumulator.

The protocol is deliberately minimal so that implementations can be
written in pure Python (NumPy), C++ (pybind11), or any other backend
without the engine knowing the difference.

See ``docs/spec/kernel_contract.md`` for the language-agnostic contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import EventBatch, FrameView, KernelState, SnapshotMeta


@runtime_checkable
class Kernel(Protocol):
    """Protocol that every accumulation kernel must satisfy.

    Implementations must be stateless with respect to the engine -- all
    mutable state lives in the ``KernelState`` object returned by
    :meth:`init_state`.
    """

    @property
    def name(self) -> str:
        """Short human-readable name (e.g. ``"event_count"``)."""
        ...

    @property
    def channels(self) -> int:
        """Number of output channels (1 for scalar, 2 for polarity, etc.)."""
        ...

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype = np.dtype(np.float32),
    ) -> KernelState:
        """Allocate and return the initial accumulator state.

        This is called once at engine construction time.  The returned
        object is passed to every subsequent ``accumulate`` / ``snapshot``
        / ``reset`` call.

        Parameters:
            sensor_size: ``(width, height)`` of the sensor.
            frame_dtype: Desired dtype for the output frame.
        """
        ...

    def accumulate(self, events: EventBatch, state: KernelState) -> None:
        """Merge *events* into *state* in-place.

        **Must not allocate memory.**  Writes directly into the
        pre-allocated arrays inside *state*.

        Parameters:
            events: 1-D structured array with fields ``t``, ``x``, ``y``,
                ``p``.
            state: The mutable accumulator returned by :meth:`init_state`.
        """
        ...

    def snapshot(
        self,
        state: KernelState,
        out: FrameView,
    ) -> SnapshotMeta:
        """Copy current accumulator into *out* and return metadata.

        Whether the accumulator is reset after snapshot is
        kernel-dependent (e.g. event_count resets, time_surface does not).

        Parameters:
            state: The accumulator state.
            out: Pre-allocated output buffer to write into.

        Returns:
            :class:`SnapshotMeta` describing the published frame.
        """
        ...

    def reset(self, state: KernelState) -> None:
        """Zero the accumulator without publishing a snapshot."""
        ...


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
_KERNEL_REGISTRY: dict[str, type] = {}


def register_kernel(name: str, cls: type) -> None:
    """Register a kernel class under the given short name."""
    _KERNEL_REGISTRY[name] = cls


def get_kernel(name: str | Kernel) -> Kernel:
    """Resolve a kernel by short name or pass through an existing instance.

    Parameters:
        name: One of the registered kernel names (``"event_count"``,
            ``"polarity"``, ``"time_surface"``, ``"exp_decay"``) or
            a pre-constructed :class:`Kernel` instance.

    Returns:
        A :class:`Kernel` instance ready for use.

    Raises:
        KeyError: If *name* is a string not found in the registry.
    """
    if isinstance(name, str):
        if name not in _KERNEL_REGISTRY:
            available = ", ".join(sorted(_KERNEL_REGISTRY)) or "(none)"
            raise KeyError(
                f"Unknown kernel {name!r}. Available: {available}"
            )
        return _KERNEL_REGISTRY[name]()  # type: ignore[return-value]
    return name
