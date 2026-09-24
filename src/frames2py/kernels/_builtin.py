"""The five built-in kernels."""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

EventArray = NDArray[np.void]
Spec = tuple[tuple[int, ...], np.dtype[Any]]

_FOLD_BELOW: Final = 2.0**-959
"""exp_decay folds its scale into storage before the scale falls below this. Every
increment is then at most 2**959, so storage stays below 2**1023 for fewer than 2**64
events per pixel."""

_REBASE_ABOVE: Final = 665.0
"""timestamp_decay moves its reference time to the watermark once the watermark is more
than this many time constants past it: floor(ln(DBL_MAX) - 64 ln 2), so storage can't
overflow for fewer than 2**64 events per pixel."""


def _flat_index(events: EventArray, width: int) -> NDArray[np.intp]:
    """Row-major pixel index of each event. Computed in intp: uint16 * int would wrap."""
    index: NDArray[np.intp] = np.multiply(events["y"], width, dtype=np.intp)
    np.add(index, events["x"], out=index)
    return index


def _hw(sensor_size: tuple[int, int]) -> tuple[int, int]:
    width, height = sensor_size
    return height, width


def _real(value: object) -> float | None:
    return float(value) if isinstance(value, numbers.Real) else None


def _positive_real(value: object, name: str) -> float:
    number = _real(value)
    if number is None or not math.isfinite(number) or not number > 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return number


@dataclasses.dataclass(slots=True)
class _Counts:
    counts: NDArray[np.uint32]
    width: int


class EventCount:
    """Events per pixel in the current window. ``(H, W)`` uint32, windowed.

    Counts wrap modulo 2**32; they never saturate.
    """

    name = "event_count"

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return _hw(sensor_size), np.dtype(np.uint32)

    def init_state(self, sensor_size: tuple[int, int]) -> _Counts:
        height, width = _hw(sensor_size)
        return _Counts(np.zeros(height * width, dtype=np.uint32), width)

    def begin_call(self, state: _Counts) -> None:
        pass

    def accumulate(self, events: EventArray, state: _Counts, watermark: int | None) -> None:
        counts = np.bincount(_flat_index(events, state.width), minlength=state.counts.size)
        np.add(state.counts, counts, out=state.counts, casting="unsafe")  # wraps modulo 2**32

    def read(self, state: _Counts, out: NDArray[Any], watermark: int | None) -> None:
        out[...] = state.counts.reshape(out.shape)

    def close_window(self, state: _Counts) -> None:
        state.counts[...] = 0

    def reset(self, state: _Counts) -> None:
        state.counts[...] = 0


class Polarity:
    """Events per pixel and polarity in the current window. ``(H, W, 2)`` uint32,
    windowed. Channel 0 counts ``p == 0`` (OFF), channel 1 every other ``p`` (ON).

    Counts wrap modulo 2**32; they never saturate.
    """

    name = "polarity"

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return (*_hw(sensor_size), 2), np.dtype(np.uint32)

    def init_state(self, sensor_size: tuple[int, int]) -> _Counts:
        height, width = _hw(sensor_size)
        return _Counts(np.zeros(height * width * 2, dtype=np.uint32), width)

    def begin_call(self, state: _Counts) -> None:
        pass

    def accumulate(self, events: EventArray, state: _Counts, watermark: int | None) -> None:
        index = _flat_index(events, state.width)
        np.multiply(index, 2, out=index)
        np.add(index, events["p"] != 0, out=index)
        counts = np.bincount(index, minlength=state.counts.size)
        np.add(state.counts, counts, out=state.counts, casting="unsafe")  # wraps modulo 2**32

    def read(self, state: _Counts, out: NDArray[Any], watermark: int | None) -> None:
        out[...] = state.counts.reshape(out.shape)

    def close_window(self, state: _Counts) -> None:
        state.counts[...] = 0

    def reset(self, state: _Counts) -> None:
        state.counts[...] = 0


@dataclasses.dataclass(slots=True)
class _Surface:
    latest: NDArray[np.uint64]
    width: int


class TimeSurface:
    """Largest timestamp per pixel. ``(H, W)`` uint64, running.

    0 means no event, so an event at ``t = 0`` is indistinguishable from none.
    """

    name = "time_surface"

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return _hw(sensor_size), np.dtype(np.uint64)

    def init_state(self, sensor_size: tuple[int, int]) -> _Surface:
        height, width = _hw(sensor_size)
        return _Surface(np.zeros(height * width, dtype=np.uint64), width)

    def begin_call(self, state: _Surface) -> None:
        pass

    def accumulate(self, events: EventArray, state: _Surface, watermark: int | None) -> None:
        np.maximum.at(state.latest, _flat_index(events, state.width), np.ascontiguousarray(events["t"]))

    def read(self, state: _Surface, out: NDArray[Any], watermark: int | None) -> None:
        out[...] = state.latest.reshape(out.shape)

    def close_window(self, state: _Surface) -> None:
        pass

    def reset(self, state: _Surface) -> None:
        state.latest[...] = 0


@dataclasses.dataclass(slots=True)
class _Decay:
    stored: NDArray[np.float64]
    width: int
    scale: float = 1.0


class ExpDecay:
    """Exponentially decaying event count. ``(H, W)`` float32, running.

    Once per accepted ``accumulate()`` or ``ingest()`` call the surface is multiplied
    by ``decay``, then each in-bounds event adds 1. The decay is per call, not per unit
    of time, so the result depends on how events are batched into calls; use
    ``TimestampDecay`` for decay in event time.

    Args:
        decay: Finite, with ``0 < decay < 1``. Anything else raises ``ValueError``.
    """

    name = "exp_decay"

    def __init__(self, decay: float) -> None:
        number = _real(decay)
        if number is None or not math.isfinite(number) or not 0 < number < 1:
            raise ValueError(f"decay must be a finite number with 0 < decay < 1, got {decay!r}")
        self._decay: float = number

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return _hw(sensor_size), np.dtype(np.float32)

    def init_state(self, sensor_size: tuple[int, int]) -> _Decay:
        height, width = _hw(sensor_size)
        return _Decay(np.zeros(height * width, dtype=np.float64), width)

    def begin_call(self, state: _Decay) -> None:
        # Stored values are value / scale. Decaying multiplies the scale; before it gets
        # small enough for 1 / scale to threaten float64's range, fold it into storage.
        scale = state.scale * self._decay
        if scale >= _FOLD_BELOW:
            state.scale = scale
            return
        np.multiply(state.stored, state.scale, out=state.stored)
        np.multiply(state.stored, self._decay, out=state.stored)
        state.scale = 1.0

    def accumulate(self, events: EventArray, state: _Decay, watermark: int | None) -> None:
        np.add.at(state.stored, _flat_index(events, state.width), 1.0 / state.scale)

    def read(self, state: _Decay, out: NDArray[Any], watermark: int | None) -> None:
        np.multiply(state.stored.reshape(out.shape), state.scale, out=out, casting="same_kind")

    def close_window(self, state: _Decay) -> None:
        pass

    def reset(self, state: _Decay) -> None:
        state.stored[...] = 0.0
        state.scale = 1.0


@dataclasses.dataclass(slots=True)
class _TimestampDecay:
    stored: NDArray[np.float64]
    width: int
    reference: int | None = None


class TimestampDecay:
    """Event-time exponential decay. ``(H, W)`` float32, running.

    Each pixel reads ``sum(exp(-(T - t_i) / tau_us))`` over its accumulated in-bounds
    events, where ``T`` is the watermark. Every event weighs 1, whatever its polarity.
    Without new in-bounds events ``T`` doesn't move and the surface doesn't change; a
    consumer can extrapolate a snapshot to a later time ``T'`` by multiplying it by
    ``exp(-(T' - T) / tau_us)``.

    In exact arithmetic the result doesn't depend on event order or on how events are
    split into calls. An accepted event with a far-future timestamp moves the watermark
    so far that earlier contributions underflow to zero.

    Args:
        tau_us: Time constant in µs. Finite and > 0; anything else raises ``ValueError``.
    """

    name = "timestamp_decay"

    def __init__(self, tau_us: float) -> None:
        self._tau_us = _positive_real(tau_us, "tau_us")

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return _hw(sensor_size), np.dtype(np.float32)

    def init_state(self, sensor_size: tuple[int, int]) -> _TimestampDecay:
        height, width = _hw(sensor_size)
        return _TimestampDecay(np.zeros(height * width, dtype=np.float64), width)

    def begin_call(self, state: _TimestampDecay) -> None:
        pass

    def accumulate(self, events: EventArray, state: _TimestampDecay, watermark: int | None) -> None:
        # Stored values are sum(exp((t_i - reference) / tau)). Differences are taken in
        # int64 (timestamps are below 2**63), then converted to float64.
        if watermark is None:
            return
        if state.reference is None:
            state.reference = watermark
        elif (watermark - state.reference) / self._tau_us > _REBASE_ABOVE:
            np.multiply(state.stored, math.exp(-(watermark - state.reference) / self._tau_us), out=state.stored)
            state.reference = watermark
        since = events["t"].astype(np.int64)
        np.subtract(since, np.int64(state.reference), out=since)
        weights = np.exp(since / self._tau_us)
        np.add.at(state.stored, _flat_index(events, state.width), weights)

    def read(self, state: _TimestampDecay, out: NDArray[Any], watermark: int | None) -> None:
        if watermark is None or state.reference is None:
            out[...] = 0.0
            return
        factor = math.exp(-(watermark - state.reference) / self._tau_us)
        np.multiply(state.stored.reshape(out.shape), factor, out=out, casting="same_kind")

    def close_window(self, state: _TimestampDecay) -> None:
        pass

    def reset(self, state: _TimestampDecay) -> None:
        state.stored[...] = 0.0
        state.reference = None
