"""Replay recorded event batches at the pace their timestamps give.

::

    from frames2py.adapters import evt
    from frames2py.replay import paced

    with evt.open("recording.raw") as reader:
        for events in paced(reader, speed=1.0):
            engine.ingest(events)

``paced()`` runs on the caller's thread and waits by sleeping; it starts no thread, keeps no
queue, never drops, reorders or copies a batch, and never catches up after falling behind.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from frames2py._events import validate

EventArray = NDArray[np.void]

__all__ = ["paced"]


def paced(
    batches: Iterable[EventArray],
    speed: float = 1,
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
