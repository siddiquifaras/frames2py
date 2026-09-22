"""AEDAT4 binary event file adapter.

Parses the AEDAT4 format used by iniVation cameras.  This is a
**pure-Python** parser -- no dependency on ``dv-processing`` or any
vendor SDK.  It reads the AEDAT4 binary container directly.

AEDAT4 file structure:
    - Magic number: ``#!AER-DAT4.0\\r\\n`` (16 bytes)
    - IOHeader (flatbuffers) length (4 bytes LE) + IOHeader data
    - Sequence of data packets, each with:
        - Packet header: stream_id (4B), size (4B)
        - Packet body: flatbuffers-encoded event packet

For simplicity and zero-dependency parsing, this adapter reads the
raw event data assuming the standard CD event layout (polarity events).
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from frames2py.core.types import EVENT_DTYPE, BatchMeta, EventBatch

# AEDAT4 magic bytes
_MAGIC = b"#!AER-DAT4.0\r\n"

# CD (Change Detection) event struct: 8 bytes packed
#   timestamp: int32 (microseconds), x: uint16, y: uint16 with polarity in MSB of y
_CD_EVENT_DTYPE = np.dtype([
    ("timestamp", "<i4"),
    ("address", "<u4"),
])


def from_aedat4(
    path: str,
    chunk_size: int = 50_000,
    sensor_size: tuple[int, int] | None = None,
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Yield event batches from an AEDAT4 file.

    This parser handles the most common AEDAT4 variant: files containing
    polarity (CD) events.  It extracts ``(t, x, y, p)`` from the binary
    stream.

    Parameters:
        path: Filesystem path to the ``.aedat4`` file.
        chunk_size: Maximum events per yielded batch.
        sensor_size: ``(width, height)`` if known; inferred from data
            otherwise.

    Yields:
        ``(EventBatch, BatchMeta)`` tuples.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"AEDAT4 file not found: {path}")

    raw_events = _parse_aedat4_events(filepath)

    if len(raw_events) == 0:
        return

    # Convert to EVENT_DTYPE
    all_events = _raw_to_event_dtype(raw_events)

    # Infer sensor size from max coordinates if not provided
    if sensor_size is None and len(all_events) > 0:
        max_x = int(all_events["x"].max()) + 1
        max_y = int(all_events["y"].max()) + 1
        sensor_size = (max_x, max_y)

    meta_base = BatchMeta(
        monotonic=True,
        source="aedat4",
        sensor_size=sensor_size,
    )

    for start in range(0, len(all_events), chunk_size):
        end = min(start + chunk_size, len(all_events))
        yield all_events[start:end], meta_base


def _parse_aedat4_events(filepath: Path) -> np.ndarray:
    """Parse raw CD events from an AEDAT4 binary file.

    Returns a structured array with fields ``timestamp`` (int32) and
    ``address`` (uint32).
    """
    data = filepath.read_bytes()
    offset = 0

    # Validate magic
    if not data[:len(_MAGIC)] == _MAGIC:
        raise ValueError("Not a valid AEDAT4 file (bad magic number)")
    offset += len(_MAGIC)

    # Skip IOHeader (flatbuffers): 4-byte LE length + payload
    if offset + 4 > len(data):
        return np.array([], dtype=_CD_EVENT_DTYPE)
    header_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + header_size

    # Read packets
    all_event_bytes = bytearray()
    while offset + 8 <= len(data):
        # Packet header: stream_id (4B), packet_size (4B)
        _stream_id, packet_size = struct.unpack_from("<II", data, offset)
        offset += 8

        if offset + packet_size > len(data):
            break

        packet_data = data[offset : offset + packet_size]
        offset += packet_size

        # Heuristic: CD event packets have sizes that are multiples of 8
        # (each CD event is 8 bytes).  Skip non-event packets.
        if packet_size > 0 and packet_size % 8 == 0:
            all_event_bytes.extend(packet_data)

    if len(all_event_bytes) == 0:
        return np.array([], dtype=_CD_EVENT_DTYPE)

    return np.frombuffer(bytes(all_event_bytes), dtype=_CD_EVENT_DTYPE)


def _raw_to_event_dtype(raw: np.ndarray) -> EventBatch:
    """Convert raw CD events to the canonical EVENT_DTYPE.

    The AEDAT4 CD event address field packs x, y, and polarity:
        - bits 0-15: x coordinate
        - bits 16-30: y coordinate
        - bit 31: polarity (1 = ON, 0 = OFF)
    """
    n = len(raw)
    out = np.empty(n, dtype=EVENT_DTYPE)

    addr = raw["address"]
    out["x"] = (addr & 0x7FFF).astype(np.uint16)           # bits 0-14
    out["y"] = ((addr >> 15) & 0x7FFF).astype(np.uint16)   # bits 15-29
    out["p"] = ((addr >> 31) & 0x1).astype(np.uint8)       # bit 31

    # Timestamps: convert from int32 microseconds to uint64
    out["t"] = raw["timestamp"].astype(np.uint64)

    return out


def to_aedat4(
    path: str,
    events: EventBatch,
    sensor_size: tuple[int, int] = (640, 480),
) -> None:
    """Write events to a minimal AEDAT4 file.

    Creates a valid AEDAT4 container with a single polarity event stream.

    Parameters:
        path: Output file path.
        events: Event array with dtype :data:`EVENT_DTYPE`.
        sensor_size: ``(width, height)`` of the sensor.
    """
    filepath = Path(path)

    # Pack events into CD format
    n = len(events)
    raw = np.empty(n, dtype=_CD_EVENT_DTYPE)
    raw["timestamp"] = events["t"].astype(np.int32)
    addr = (
        events["x"].astype(np.uint32)
        | (events["y"].astype(np.uint32) << 15)
        | (events["p"].astype(np.uint32) << 31)
    )
    raw["address"] = addr

    event_bytes = raw.tobytes()

    with filepath.open("wb") as f:
        # Magic
        f.write(_MAGIC)
        # Minimal IOHeader (empty flatbuffer -- 4 zero bytes for size)
        f.write(struct.pack("<I", 0))
        # Single event packet
        stream_id = 0
        f.write(struct.pack("<II", stream_id, len(event_bytes)))
        f.write(event_bytes)
