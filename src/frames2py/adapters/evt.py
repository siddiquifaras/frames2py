"""Read Prophesee EVT 2.0 and EVT 3.0 RAW files.

::

    from frames2py.adapters import evt

    with evt.open("recording.raw") as reader:
        engine = frames2py.Engine(reader.sensor_size, "event_count")
        for events in reader:
            engine.ingest(events)

The header's version picks the decoder: ``% evt 2.0`` or ``% format EVT2``, ``% evt 3.0`` or
``% format EVT3``. Any other version raises ``ValueError``. Geometry comes from the header
(``% geometry WxH`` or the ``format`` line's ``width=`` and ``height=``); a header without it
needs ``sensor_size``, and an explicit ``sensor_size`` must match a header that has it.

Only CD events are decoded; triggers and monitoring words are skipped. Timestamps are the
sensor clock's microseconds, reconstructed across the counter's wraps and otherwise left as
the file has them: never sorted, clamped, shifted to start at 0, or reset. The decoding
rules, including what counts as a wrap, are in ``frames2py.adapters._evt_decode``.

The decoder has no geometry filter: an event with y at or beyond the sensor height is
yielded, and the Accumulator or Engine counts it as out of bounds.

No dependency beyond NumPy; the ``frames2py[evt]`` extra exists so the install command
stays the same if that changes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import BinaryIO, Final

from frames2py._events import EventArray
from frames2py.adapters._evt_decode import Evt2Decoder, Evt3Decoder
from frames2py.adapters._reader import (
    PathArg,
    Reader,
    Size,
    check_batch_size,
    check_sensor_size,
    merge_sensor_size,
    open_binary,
)

__all__ = ["open"]

READ_SIZE: Final = 1 << 20
"""Bytes read from the file per decoder feed."""

_VERSIONS: Final = {"2.0": "2.0", "3.0": "3.0", "EVT2": "2.0", "EVT3": "3.0"}


def open(
    path: PathArg,
    *,
    sensor_size: tuple[int, int] | None = None,
    batch_size: int | None = None,
) -> Reader:
    """Open an EVT 2.0 or 3.0 RAW file for one pass over its CD events.

    Args:
        path: The ``.raw`` file.
        sensor_size: ``(width, height)``; required if the header has no geometry.
        batch_size: Events per yielded array (the last may have fewer), or ``None`` for
            the decoder's own boundaries.

    Raises:
        FileNotFoundError: *path* doesn't exist.
        OSError: *path* is a directory or can't be read.
        ValueError: the header is malformed, its version is not EVT 2.0 or 3.0, it has no
            geometry and no ``sensor_size`` was given, or ``sensor_size`` conflicts with it.
    """
    explicit = check_sensor_size(sensor_size)
    size = check_batch_size(batch_size)
    handle = open_binary(path)
    try:
        header = _read_header(handle)
        version = _version(header)
        geometry = merge_sensor_size(explicit, _geometry(header), "header")
        if geometry is None:
            raise ValueError("the EVT header has no geometry; pass sensor_size=(width, height)")
    except BaseException:
        handle.close()
        raise
    decoder = Evt2Decoder() if version == "2.0" else Evt3Decoder()
    return Reader(lambda: _batches(handle, decoder), handle.close, geometry, size)


def _batches(handle: BinaryIO, decoder: Evt2Decoder | Evt3Decoder) -> Iterator[EventArray]:
    while chunk := handle.read(READ_SIZE):
        yield from decoder.feed(chunk)


def _read_header(handle: BinaryIO) -> dict[str, list[str]]:
    """Header lines start with ``%``. The header ends at the first line that doesn't, or
    after ``% end``. Returns each keyword's values; the file is left at the body."""
    fields: dict[str, list[str]] = {}
    while True:
        at = handle.tell()
        if handle.read(1) != b"%":
            handle.seek(at)
            return fields
        line = handle.readline().decode("latin-1").strip()
        if line == "end":
            return fields
        key, _, value = line.partition(" ")
        if key:
            fields.setdefault(key, []).append(value.strip())


def _one(header: dict[str, list[str]], key: str) -> str | None:
    values = set(header.get(key, ()))
    if len(values) > 1:
        raise ValueError(f"the EVT header has conflicting {key!r} lines: {sorted(values)}")
    return values.pop() if values else None


def _version(header: dict[str, list[str]]) -> str:
    stated = {}
    if (evt := _one(header, "evt")) is not None:
        stated["evt"] = evt
    if (fmt := _one(header, "format")) is not None:
        stated["format"] = fmt.split(";")[0].strip()
    if not stated:
        raise ValueError("the file has no EVT version in its header (no '% evt' or '% format' line)")
    versions = set()
    for key, value in stated.items():
        if value not in _VERSIONS:
            raise ValueError(f"unsupported EVT version {value!r} ({key!r} header line); EVT 2.0 and 3.0 are read")
        versions.add(_VERSIONS[value])
    if len(versions) > 1:
        raise ValueError(f"the EVT header states two versions: {stated}")
    return versions.pop()


def _geometry(header: dict[str, list[str]]) -> Size | None:
    found = set()
    if (geometry := _one(header, "geometry")) is not None:
        match = re.fullmatch(r"(\d+)x(\d+)", geometry)
        if match is None:
            raise ValueError(f"malformed EVT header geometry {geometry!r}")
        found.add((int(match[1]), int(match[2])))
    if (fmt := _one(header, "format")) is not None:
        options = dict(part.strip().partition("=")[::2] for part in fmt.split(";")[1:] if part.strip())
        if "width" in options or "height" in options:
            try:
                found.add((int(options["width"]), int(options["height"])))
            except (KeyError, ValueError):
                raise ValueError(f"malformed EVT header format geometry {fmt!r}") from None
    if len(found) > 1:
        raise ValueError(f"the EVT header states two geometries: {sorted(found)}")
    size = found.pop() if found else None
    if size is not None and (size[0] < 1 or size[1] < 1):
        raise ValueError(f"the EVT header states an empty geometry {size}")
    return size
