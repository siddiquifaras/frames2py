"""Replay recorded event batches: at the pace their timestamps give, or as frames in event time.

::

    from frames2py.adapters import evt
    from frames2py.replay import paced

    with evt.open("recording.raw") as reader:
        for events in paced(reader, speed=1.0):
            engine.ingest(events)

``paced()`` runs on the caller's thread and waits by sleeping; it starts no thread, keeps no
queue, never drops, reorders or copies a batch, and never catches up after falling behind.

``windows()`` reads a representation every N µs of event time, with no clock at all::

    from frames2py import VoxelGrid
    from frames2py.replay import windows

    with evt.open("recording.raw") as reader:
        for t_us, frame in windows(reader, (1280, 720), VoxelGrid(bins=5, bin_us=12_500), every_us=50_000):
            ...
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frames2py import Accumulator
from frames2py._events import TIMESTAMP_LIMIT, validate
from frames2py.kernels import ExpDecay, Kernel

EventArray = NDArray[np.void]

__all__ = ["paced", "windows"]


def paced(
    batches: Iterable[EventArray],
    *,
    speed: float = 1.0,
    clock: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], Any] = time.sleep,
) -> Iterator[EventArray]:
    """Yield each batch of *batches*, unchanged, once its timestamps say it is due.

    The first nonempty batch sets the start: its smallest timestamp ``t0`` and the clock's
    reading when it arrives. ``M`` is the largest timestamp seen so far, the batch about to
    be yielded included. A batch is yielded once the clock has advanced
    ``(M - t0) / speed`` µs (rounded up to a whole ns) past the start. Timestamps are never repaired and no reset is
    inferred; a discontinuity is the caller's to handle. A batch whose timestamps go back
    leaves ``M`` where it was and is yielded without waiting; a jump forward makes the
    batch, and the ones after it, wait for it. Empty batches are yielded at once. A
    consumer slower than the recording gets every batch late; nothing is skipped to catch
    up.

    Args:
        batches: ``EVENT_DTYPE`` arrays, such as a file adapter's reader.
        speed: Replay rate relative to the recording; 2 is twice as fast.
        clock: Nanoseconds, monotonic. Injectable for tests.
        sleep: Waits the given seconds. Injectable for tests.

    Raises:
        ValueError: *speed* is not a finite number above 0 (on the call).
        TypeError: a batch is not an ``EVENT_DTYPE``-compatible array (when it is reached).
    """
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not (math.isfinite(speed) and speed > 0):
        raise ValueError(f"speed must be a finite number above 0, got {speed!r}")
    return _paced(iter(batches), float(speed), clock, sleep)


def _paced(
    batches: Iterator[EventArray], speed: float, clock: Callable[[], int], sleep: Callable[[float], Any]
) -> Iterator[EventArray]:
    t0: int | None = None
    top = 0
    start = 0
    for batch in batches:
        events = validate(batch)
        if len(events) == 0:
            yield batch
            continue
        if t0 is None:
            t0, top, start = int(events["t"].min()), int(events["t"].max()), clock()
        else:
            top = max(top, int(events["t"].max()))
        due_ns = math.ceil((top - t0) * 1000 / speed)  # whole ns, never early
        while (remaining := due_ns - (clock() - start)) > 0:
            sleep(remaining / 1e9)
        yield batch


def windows(
    batches: Iterable[EventArray],
    sensor_size: tuple[int, int],
    kernel: str | Kernel,
    *,
    every_us: int,
) -> Iterator[tuple[int, NDArray[Any]]]:
    """Yield ``(t_us, frame)``: the representation at every multiple of *every_us* µs of event time.

    The events of *batches* are accumulated in arrival order through an ``Accumulator``. The
    boundaries are ``m * every_us``, absolute event times. The frame at boundary ``t_us`` is
    read just before the first in-bounds event that brings the watermark to ``t_us`` or past
    it is accumulated, so for in-order input it holds exactly the events before ``t_us``. The
    first frame is at the first boundary after the first in-bounds event; out-of-bounds
    events count for nothing and cross no boundary. A windowed kernel starts a new window
    after each frame. Every boundary the watermark passes gets a frame, gaps included, so a
    forward timestamp spike yields a frame for each boundary up to it: handle discontinuities
    before calling (see the event contract on timestamp discontinuities). The incomplete
    window after the last boundary reached is not yielded. Each frame is a new writable
    array. The frames don't depend on how the events are split into batches.

    Each frame is read at its boundary ``t_us``, not at the accumulated watermark: the
    kernel's ``read`` is given ``t_us``, which is later than any event accumulated. The
    temporal kernels then show the bins completed by ``t_us``, and ``TimestampDecay`` decays
    to ``t_us``. A custom kernel must evaluate at the time ``read`` is given (``Kernel.read``).

    ``ExpDecay`` is refused: its decay is per call, and this function splits batches at the
    boundaries. Use ``TimestampDecay`` for decay in event time. A custom kernel whose result
    depends on how events are split into calls isn't detected; its frames depend on these
    split points.

    Args:
        batches: ``EVENT_DTYPE`` arrays, such as a file adapter's reader.
        sensor_size: ``(width, height)``, as for ``Accumulator``.
        kernel: A kernel instance, or a name, as for ``Accumulator``.
        every_us: The interval between boundaries in µs, an int ``>= 1``.

    Raises:
        TypeError: *every_us* is not an int (a bool included), or *kernel* is an ``ExpDecay``
            (on the call); a batch is not an ``EVENT_DTYPE``-compatible array (when it is
            reached, before any of its events is accumulated).
        ValueError: *every_us* is below 1, or *kernel* is an unknown name (on the call); a batch
            has an event with ``t >= 2**63`` (when it is reached, before any of its events is
            accumulated or any frame it would complete is yielded).
    """
    if isinstance(every_us, (bool, np.bool_)):
        raise TypeError("every_us must be an int, got a bool")
    try:
        every = operator.index(every_us)
    except TypeError:
        raise TypeError(f"every_us must be an int, got {type(every_us).__name__}") from None
    if every < 1:
        raise ValueError(f"every_us must be >= 1, got {every}")
    if isinstance(kernel, ExpDecay):
        raise TypeError(
            "windows() doesn't accept ExpDecay: its decay is per call, so its frames would depend on where "
            "the batches are split; use TimestampDecay for decay in event time"
        )
    width, height = sensor_size
    return _windows(iter(batches), Accumulator(sensor_size, kernel), (int(width), int(height)), every)


def _windows(
    batches: Iterator[EventArray], accumulator: Accumulator, sensor_size: tuple[int, int], every: int
) -> Iterator[tuple[int, NDArray[Any]]]:
    shape, dtype = accumulator._output_spec
    for batch in batches:
        events = validate(batch)
        if len(events) and int(events["t"].max()) >= TIMESTAMP_LIMIT:
            raise ValueError("an event has t >= 2**63; the whole batch is rejected")
        start = 0
        for position, passed, reached in zip(*_crossings(events, accumulator.watermark, sensor_size, every)):
            if position > start:
                accumulator._accumulate(events[start:position])
                start = position
            for m in range(passed + 1, reached + 1):
                frame = np.empty(shape, dtype=dtype)
                accumulator._read_into(frame, at=m * every)
                accumulator._close_window()
                yield m * every, frame
        if start < len(events):
            accumulator._accumulate(events[start:])


def _crossings(
    events: EventArray, watermark: int | None, sensor_size: tuple[int, int], every: int
) -> tuple[list[int], list[int], list[int]]:
    """The in-bounds events that bring the watermark past a boundary, in arrival order.

    For each: its position in *events*, the window ``watermark // every`` before it, and the
    window it brings the watermark to. The first in-bounds event ever crosses nothing.
    """
    if not len(events):
        return [], [], []
    width, height = sensor_size
    x, y, t = events["x"], events["y"], events["t"]
    position: NDArray[np.intp] | None = None
    if int(x.max()) >= width or int(y.max()) >= height:
        position = np.flatnonzero((x < width) & (y < height))
        t = t[position]
        if not len(t):
            return [], [], []
    running = np.maximum.accumulate(t)
    if watermark is not None:
        np.maximum(running, np.uint64(watermark), out=running)
    # Timestamps are below 2**63, so dividing by min(every, 2**63) gives the same windows.
    window = running // np.uint64(min(every, TIMESTAMP_LIMIT))
    previous = np.empty_like(window)
    previous[0] = (int(t[0]) if watermark is None else watermark) // every
    previous[1:] = window[:-1]
    crossing = np.flatnonzero(window > previous)
    at = crossing if position is None else position[crossing]
    return at.tolist(), previous[crossing].tolist(), window[crossing].tolist()
