"""The Accumulator: synchronous accumulation of events through one kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from frames2py._events import TIMESTAMP_LIMIT, EventArray, validate
from frames2py.kernels import EventCount, Kernel, Polarity, TimeSurface

_BY_NAME: Final[dict[str, Callable[[], Kernel]]] = {
    "event_count": EventCount,
    "polarity": Polarity,
    "time_surface": TimeSurface,
}


def resolve_kernel(kernel: str | Kernel) -> Kernel:
    """A configured kernel as given, or a new built-in kernel for a name."""
    if isinstance(kernel, str):
        try:
            return _BY_NAME[kernel]()
        except KeyError:
            raise ValueError(
                f"no kernel named {kernel!r}; names exist for {sorted(_BY_NAME)}, and "
                "kernels with parameters are passed configured, e.g. ExpDecay(0.9)"
            ) from None
    return kernel


class Accumulator:
    """Accumulates events through one kernel.

    Synchronous and single-threaded: each call does its work on the caller's thread
    and returns. There is no publication; ``read()`` returns the current
    representation.

    Args:
        sensor_size: ``(width, height)``. Frames are ``(height, width[, channels])``.
        kernel: A kernel instance, or ``"event_count"``, ``"polarity"`` or
            ``"time_surface"``.
    """

    def __init__(self, sensor_size: tuple[int, int], kernel: str | Kernel) -> None:
        width, height = sensor_size
        self._sensor_size = (int(width), int(height))
        self._kernel = resolve_kernel(kernel)
        self._shape, self._dtype = self._kernel.output_spec(self._sensor_size)
        self._state = self._kernel.init_state(self._sensor_size)
        self._watermark: int | None = None
        self._events_out_of_bounds = 0

    @property
    def watermark(self) -> int | None:
        """The largest timestamp among accumulated in-bounds events, or ``None``."""
        return self._watermark

    @property
    def events_out_of_bounds(self) -> int:
        """Events skipped by the bounds check since construction or ``reset()``."""
        return self._events_out_of_bounds

    def accumulate(self, events: EventArray) -> None:
        """Accumulate one call's events.

        Raises ``TypeError`` for a malformed array and ``ValueError`` if any event has
        ``t >= 2**63``, in both cases before anything changes. Out-of-bounds events are
        counted and otherwise ignored.
        """
        self._accumulate(events)

    def _accumulate(self, events: object) -> int:
        """``accumulate``, returning the number of in-bounds events."""
        checked = validate(events)
        if len(checked) and int(checked["t"].max()) >= TIMESTAMP_LIMIT:
            raise ValueError("an event has t >= 2**63; the whole call is rejected")
        inside = self._in_bounds(checked)
        self._kernel.begin_call(self._state)
        self._events_out_of_bounds += len(checked) - len(inside)
        if len(inside):
            latest = int(inside["t"].max())
            if self._watermark is None or latest > self._watermark:
                self._watermark = latest
            self._kernel.accumulate(inside, self._state, self._watermark)
        return len(inside)

    def _in_bounds(self, events: EventArray) -> EventArray:
        if not len(events):
            return events
        width, height = self._sensor_size
        x, y = events["x"], events["y"]
        if int(x.max()) < width and int(y.max()) < height:
            return events
        inside: EventArray = events[(x < width) & (y < height)]
        return inside

    def read(self) -> NDArray[Any]:
        """A copy of the current representation. Doesn't change any state."""
        out = np.empty(self._shape, dtype=self._dtype)
        self._read_into(out)
        return out

    def reset(self) -> None:
        """Clear the kernel state, the watermark and the out-of-bounds count."""
        self._kernel.reset(self._state)
        self._watermark = None
        self._events_out_of_bounds = 0

    def _read_into(self, out: NDArray[Any]) -> None:
        self._kernel.read(self._state, out, self._watermark)

    def _close_window(self) -> None:
        self._kernel.close_window(self._state)

    @property
    def _output_spec(self) -> tuple[tuple[int, ...], np.dtype[Any]]:
        return self._shape, self._dtype
