"""The two temporal kernels: ``StackedHistogram`` and ``VoxelGrid``.

Both are running kernels on an absolute event-time grid: bin ``k`` covers
``[k * bin_us, (k + 1) * bin_us)``. A frame read at time ``T`` shows only bins completed
by ``T``, the bins before ``k_T = T // bin_us``.

State is a set of planes, one per live bin: the bins a read at the accumulated watermark
or any later time can still show. Bin ``k`` lives in plane ``k % ring``. When the
watermark enters a new bin, the planes of the bins it entered are cleared
(``_clear_planes``); the bins they held are too old to be shown again.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from frames2py.kernels._builtin import EventArray, Spec, _flat_index, _hw

BIN_US_LIMIT: Final = 2**28
"""``bin_us`` must be below this."""

_MAGIC: Final = 6755399441055744.0  # 1.5 * 2**52
_MAGIC_BITS: Final = np.uint64(0x4338000000000000)  # its float64 bits
_EXACT_BELOW: Final = 2**51  # |n| below this: bits(_MAGIC) + n is the float64 _MAGIC + n exactly


def _parameter(value: Any, name: str, minimum: int, limit: int | None = None) -> int:
    """*value* as an int through ``operator.index``; anything else, or out of range, raises ``ValueError``."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an int, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an int, got {value!r}") from None
    if number < minimum or (limit is not None and number >= limit):
        bound = f">= {minimum}" if limit is None else f"in [{minimum}, {limit})"
        raise ValueError(f"{name} must be {bound}, got {number}")
    return number


@dataclasses.dataclass(slots=True)
class _Planes:
    planes: NDArray[Any]  # (parts, ring, H * W)
    width: int
    newest: int | None = None  # the bin of the watermark at the last accumulate()
    total: int = 0  # events accumulated since reset
    cleared_at: list[int] | None = None  # per plane: total when the plane was last cleared
    scratch: NDArray[Any] | None = None  # VoxelGrid.read's working plane, kept so it isn't faulted in again


def _clear_planes(state: _Planes, newest: int) -> None:
    """Move the newest live bin to *newest*, zeroing the planes of the bins it enters."""
    previous = state.newest
    state.newest = newest if previous is None else max(previous, newest)
    if previous is None or newest <= previous:
        return
    ring = state.planes.shape[1]
    if newest - previous >= ring:
        state.planes[...] = 0
        if state.cleared_at is not None:
            state.cleared_at[:] = [state.total] * ring
        return
    for k in range(previous + 1, newest + 1):
        state.planes[:, k % ring] = 0
        if state.cleared_at is not None:
            state.cleared_at[k % ring] = state.total


def _recent(events: EventArray, oldest: int, bin_us: int) -> EventArray:
    """The events in bin *oldest* or later; earlier ones can't be shown at any later read."""
    if oldest <= 0:
        return events
    start = oldest * bin_us
    t = events["t"]
    if int(t.min()) >= start:
        return events
    recent: EventArray = events[t >= np.uint64(start)]
    return recent


class _Temporal:
    """Parameters and the lifecycle shared by the two kernels."""

    _MIN_BINS: int

    def __init__(self, *, bins: int, bin_us: int) -> None:
        self._bins = _parameter(bins, "bins", self._MIN_BINS)
        self._bin_us = _parameter(bin_us, "bin_us", 1, BIN_US_LIMIT)

    def begin_call(self, state: _Planes) -> None:
        pass

    def close_window(self, state: _Planes) -> None:
        pass

    def reset(self, state: _Planes) -> None:
        state.planes[...] = 0
        state.newest = None
        state.total = 0
        if state.cleared_at is not None:
            state.cleared_at[:] = [0] * len(state.cleared_at)

    def _live(self, state: _Planes, k: int) -> bool:
        """Whether bin *k* is held in a plane. Bins after the newest hold no event yet."""
        assert state.newest is not None
        ring: int = state.planes.shape[1]
        return state.newest - ring < k <= state.newest


class StackedHistogram(_Temporal):
    """Events per polarity in each of the last *bins* completed time bins.
    ``(2, bins, H, W)`` uint32, running.

    Bin ``k`` covers ``[k * bin_us, (k + 1) * bin_us)`` of event time. Read at the watermark
    ``T``, the frame shows the *bins* bins before the one ``T`` falls in, oldest first: index
    ``j`` is bin ``T // bin_us - bins + j``, so the frame covers ``bins * bin_us`` µs and never
    shows the bin still being filled. Channel 0 counts ``p == 0`` (OFF), channel 1 every
    other ``p`` (ON). Events older than the frame count for nothing. Counts wrap modulo
    2**32; they are never clipped or normalised.

    The result doesn't depend on event order or on how events are split into calls.

    Args:
        bins: Time bins shown, an int ``>= 1``.
        bin_us: Bin width in µs, an int with ``1 <= bin_us < 2**28``.

    Anything else, a bool or a float included, raises ``ValueError``.
    """

    name = "stacked_histogram"
    _MIN_BINS = 1

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return (2, self._bins, *_hw(sensor_size)), np.dtype(np.uint32)

    def init_state(self, sensor_size: tuple[int, int]) -> _Planes:
        height, width = _hw(sensor_size)
        # bins + 1 planes: the bins a read at the watermark shows, and the bin in progress.
        return _Planes(np.zeros((2, self._bins + 1, height * width), dtype=np.uint32), width)

    def accumulate(self, events: EventArray, state: _Planes, watermark: int | None) -> None:
        if watermark is None:
            return
        newest = watermark // self._bin_us
        _clear_planes(state, newest)
        ring = state.planes.shape[1]
        events = _recent(events, newest - ring + 1, self._bin_us)
        plane = np.remainder(events["t"] // self._bin_us, ring)
        index = np.multiply(plane, state.planes.shape[2], dtype=np.intp)
        np.add(index, _flat_index(events, state.width), out=index)
        np.add(index, np.multiply(events["p"] != 0, ring * state.planes.shape[2], dtype=np.intp), out=index)
        np.add.at(state.planes.reshape(-1), index, np.uint32(1))  # uint32 arithmetic: wraps modulo 2**32

    def read(self, state: _Planes, out: NDArray[Any], watermark: int | None) -> None:
        if watermark is None or state.newest is None:
            out[...] = 0
            return
        ring = state.planes.shape[1]
        first = watermark // self._bin_us - self._bins
        for j in range(self._bins):
            k = first + j
            if self._live(state, k):
                out[:, j] = state.planes[:, k % ring].reshape(out.shape[0], *out.shape[2:])
            else:
                out[:, j] = 0


class VoxelGrid(_Temporal):
    """Signed events spread linearly in time over *bins* knots. ``(bins, H, W)`` float32,
    running.

    Bin ``k`` covers ``[k * bin_us, (k + 1) * bin_us)`` of event time. Read at the watermark
    ``T``, knot ``j`` sits at ``(T // bin_us - bins + 1 + j) * bin_us``, oldest first, so the
    last knot is at the start of the bin ``T`` falls in. The frame covers the ``bins - 1``
    completed bins between the first and last knots, ``(bins - 1) * bin_us`` µs; the bin
    still being filled is never shown. An event in that span, at ``t = q * bin_us + r``, adds
    ``s * (bin_us - r) / bin_us`` to the knot at ``q * bin_us`` and ``s * r / bin_us`` to the
    next, with ``s = +1`` for ``p != 0`` and ``-1`` for ``p == 0``. Events outside the span
    count for nothing. No normalisation.

    Each knot's numerator, the sum of ``s * (bin_us - r)`` and ``s * r`` terms, is kept
    exactly as an integer modulo 2**64. It is read as a two's-complement int64 ``n``, so it
    is exact while the true sum lies in ``[-2**63, 2**63 - 1]`` and wraps outside it. The
    output is ``float32(float64(n) / float64(bin_us))``, correctly rounded while
    ``|n| <= 2**53``. The result doesn't depend on event order or on how events are split
    into calls.

    Args:
        bins: Knots, an int ``>= 2``.
        bin_us: Bin width in µs, an int with ``1 <= bin_us < 2**28``.

    Anything else, a bool or a float included, raises ``ValueError``.
    """

    name = "voxel_grid"
    _MIN_BINS = 2

    def output_spec(self, sensor_size: tuple[int, int]) -> Spec:
        return (self._bins, *_hw(sensor_size)), np.dtype(np.float32)

    def init_state(self, sensor_size: tuple[int, int]) -> _Planes:
        height, width = _hw(sensor_size)
        # Per live bin, the numerator parts its events give the knot at its start (plane 0)
        # and the knot at its end (plane 1): bins planes each, the span at the watermark
        # and the bin in progress.
        planes = np.zeros((2, self._bins, height * width), dtype=np.uint64)
        return _Planes(planes, width, cleared_at=[0] * self._bins)

    def accumulate(self, events: EventArray, state: _Planes, watermark: int | None) -> None:
        if watermark is None:
            return
        newest = watermark // self._bin_us
        _clear_planes(state, newest)
        ring = state.planes.shape[1]
        pixels = state.planes.shape[2]
        events = _recent(events, newest - ring + 1, self._bin_us)
        state.total += len(events)
        q, r = np.divmod(events["t"], np.uint64(self._bin_us))
        index = np.multiply(np.remainder(q, ring), pixels, dtype=np.intp)
        np.add(index, _flat_index(events, state.width), out=index)
        off = events["p"] == 0
        # uint64 arithmetic: negation and sums are modulo 2**64.
        start = np.subtract(np.uint64(self._bin_us), r)
        start = np.where(off, np.negative(start), start)
        r = np.where(off, np.negative(r), r)
        flat = state.planes.reshape(-1)
        np.add.at(flat, index, start)
        np.add(index, ring * pixels, out=index)
        np.add.at(flat, index, r)

    def read(self, state: _Planes, out: NDArray[Any], watermark: int | None) -> None:
        if watermark is None or state.newest is None:
            out[...] = 0
            return
        ring = state.planes.shape[1]
        first = watermark // self._bin_us - self._bins + 1  # the first knot's bin
        last = first + self._bins - 1  # the last knot's bin: the one in progress at the watermark
        if state.scratch is None:
            state.scratch = np.empty(state.planes.shape[2], dtype=np.uint64)
        scratch = state.scratch
        for j in range(self._bins):
            k = first + j
            # Knot k gets the start parts of bin k and the end parts of bin k - 1, each only
            # from a bin in the span, first ... last - 1.
            start = state.planes[0, k % ring] if k < last and self._live(state, k) else None
            end = state.planes[1, (k - 1) % ring] if k > first and self._live(state, k - 1) else None
            if start is None:
                if end is None:
                    out[j] = 0
                    continue
                numerator = end
            else:
                numerator = start if end is None else np.add(start, end, out=scratch)
            # A plane holds at most the events accumulated since it was cleared, each moving a
            # numerator by at most bin_us. Below 2**51 the conversion to float64 can add a
            # constant to the bits and subtract it as a float instead, exactly and faster.
            assert state.cleared_at is not None
            events_in = 2 * state.total - state.cleared_at[k % ring] - state.cleared_at[(k - 1) % ring]
            if events_in * self._bin_us < _EXACT_BELOW:
                exact = scratch.view(np.float64)
                np.add(numerator, _MAGIC_BITS, out=scratch)
                np.subtract(exact, _MAGIC, out=exact)
                np.divide(exact.reshape(out.shape[1:]), float(self._bin_us), out=out[j], casting="same_kind")
            else:
                np.divide(numerator.view(np.int64).reshape(out.shape[1:]), float(self._bin_us), out=out[j],
                          dtype=np.float64, casting="same_kind")
