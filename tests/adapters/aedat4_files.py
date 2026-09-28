"""AEDAT 4.0 files for tests, written by dv-processing.

dv-processing's writer refuses negative and backward timestamps, so those files are written
uncompressed with placeholder timestamps that are then replaced byte for byte: every
occurrence of a placeholder's int64 (the packet and the file's index) becomes the wanted
value. The same trick removes a stream's resolution attributes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PLACEHOLDER = 7_000_000_000_000


def write(path: Path, packets: dict[str, list[list[tuple[int, int, int, int]]]], *,
          resolution: tuple[int, int] = (640, 480), camera: str = "cam", compressed: bool = True) -> Path:
    """Write event packets per stream name; each event is ``(t, x, y, p)``."""
    import dv_processing as dv

    compression = dv.CompressionType.LZ4 if compressed else dv.CompressionType.NONE
    config = dv.io.MonoCameraWriter.Config(camera, compression)
    for name in packets:
        config.addEventStream(resolution, name)
    writer = dv.io.MonoCameraWriter(str(path), config)
    for name, stream in packets.items():
        for packet in stream:
            store = dv.EventStore()
            for t, x, y, p in packet:
                store.push_back(t, x, y, bool(p))
            writer.writeEvents(store, name)
    del writer
    return path


def write_with_timestamps(path: Path, packets: list[list[int]]) -> Path:
    """One event stream whose packets hold exactly these timestamps (any order, any sign)."""
    values = [t for packet in packets for t in packet]
    placeholders = iter(range(PLACEHOLDER, PLACEHOLDER + len(values)))
    events = [[(next(placeholders), i, 7, i % 2) for i, _ in enumerate(packet)] for packet in packets]
    write(path, {"events": events}, compressed=False)
    data = path.read_bytes()
    for k, t in enumerate(values):
        old = np.int64(PLACEHOLDER + k).tobytes()
        assert data.count(old) >= 1
        data = data.replace(old, np.int64(t).tobytes())
    path.write_bytes(data)
    return path


def without_resolution(path: Path) -> Path:
    data = path.read_bytes()
    assert b'key="sizeX"' in data and b'key="sizeY"' in data
    path.write_bytes(data.replace(b'key="sizeX"', b'key="sizeQ"').replace(b'key="sizeY"', b'key="sizeR"'))
    return path
