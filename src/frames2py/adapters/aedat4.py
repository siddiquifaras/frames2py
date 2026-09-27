"""Read iniVation AEDAT 4.0 files through dv-processing.

::

    from frames2py.adapters import aedat4

    with aedat4.open("recording.aedat4") as reader:
        engine = frames2py.Engine(reader.sensor_size, "polarity")
        for events in reader:
            engine.ingest(events)

dv-processing does the decoding; this module maps its events to ``EVENT_DTYPE``. Needs the
``frames2py[aedat4]`` extra.

Which stream: dv-processing reads the first camera named in the file's description; of its
event streams, the one named ``events`` is read, or the only one there is. A camera with
several event streams and none named ``events`` raises ``ValueError``: dv-processing's Python
API doesn't expose their order in the description.

Timestamps are the file's int64 microseconds, unchanged and in file order (the format page
describes them as Unix time; recordings from other sources may use another epoch). A negative
timestamp raises ``ValueError`` when iteration reaches it; it is never clamped. Polarity maps
to ``p = 1`` for ON and ``0`` for OFF. Geometry is the stream's resolution; an explicit
``sensor_size`` must match it, and a stream without one needs ``sensor_size``.

Known limitation: dv-processing has been seen to give no result for minutes on a file with
corrupted bytes in the middle of a packet. That happens inside dv-processing, before any
events reach this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from frames2py._events import EVENT_DTYPE, EventArray
from frames2py.adapters._reader import (
    PathArg,
    Reader,
    check_batch_size,
    check_readable,
    check_sensor_size,
    merge_sensor_size,
    require,
)

__all__ = ["open"]


def open(
    path: PathArg,
    *,
    sensor_size: tuple[int, int] | None = None,
    batch_size: int | None = None,
) -> Reader:
    """Open an AEDAT 4.0 file for one pass over its first event stream.

    Args:
        path: The ``.aedat4`` file.
        sensor_size: ``(width, height)``; required if the stream has no resolution.
        batch_size: Events per yielded array (the last may have fewer), or ``None`` for
            the file's own packets.

    Raises:
        ImportError: dv-processing is not installed (``frames2py[aedat4]``).
        FileNotFoundError: *path* doesn't exist.
        OSError: *path* is a directory or can't be read.
        ValueError: dv-processing can't read the file, it has no event stream, the stream
            has no resolution and no ``sensor_size`` was given, or ``sensor_size``
            conflicts with it.
    """
    explicit = check_sensor_size(sensor_size)
    size = check_batch_size(batch_size)
    dv = require("dv_processing", "aedat4")
    name = check_readable(path)
    try:
        recording = dv.io.MonoCameraRecording(name)
        stream = _event_stream(recording)
        resolution = recording.getEventResolution(stream)
    except RuntimeError as exc:
        raise ValueError(f"dv-processing could not read {name!r}: {_first_line(exc)}") from exc
    source = None if resolution is None else (int(resolution[0]), int(resolution[1]))
    geometry = merge_sensor_size(explicit, source, "event stream")
    if geometry is None:
        raise ValueError("the event stream has no resolution; pass sensor_size=(width, height)")
    holder = [recording]
    return Reader(lambda: _batches(holder, stream), holder.clear, geometry, size)


def _event_stream(recording: Any) -> str:
    streams = [name for name in recording.getStreamNames() if recording.isStreamOfEventType(name)]
    if "events" in streams:
        return "events"
    if len(streams) == 1:
        return str(streams[0])
    if not streams:
        raise ValueError(f"the file's first camera ({recording.getCameraName()!r}) has no event stream")
    raise ValueError(f"the file's first camera has several event streams and none named 'events': {streams}")


def _batches(holder: list[Any], stream: str) -> Iterator[EventArray]:
    while holder:
        try:
            store = holder[0].getNextEventBatch(stream)
        except RuntimeError as exc:
            raise ValueError(f"dv-processing failed reading the event stream: {_first_line(exc)}") from exc
        if store is None:
            return
        yield _to_events(store.numpy())


def _to_events(packet: Any) -> EventArray:
    """dv-processing's ``timestamp <i8, x <i2, y <i2, polarity i1`` records as ``EVENT_DTYPE``."""
    t, x, y = packet["timestamp"], packet["x"], packet["y"]
    if len(packet):
        if int(t.min()) < 0:
            raise ValueError(f"the file has a negative timestamp ({int(t.min())} us); it is not clamped")
        if int(x.min()) < 0 or int(y.min()) < 0:
            raise ValueError("the file has an event with a negative coordinate")
    events = np.empty(len(packet), dtype=EVENT_DTYPE)
    events["t"] = t
    events["x"] = x
    events["y"] = y
    events["p"] = packet["polarity"] != 0
    return events


def _first_line(exc: BaseException) -> str:
    """dv-processing appends a native stack trace to its messages; the cause keeps it."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
