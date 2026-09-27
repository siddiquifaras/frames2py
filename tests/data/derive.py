"""Rebuild the committed fixtures in this directory from their source recordings.

    uv run python -m tests.data.derive [--check]

Each fixture is derived from a CC0 Prophesee recording fetched and verified by
``tests.recordings``; ``README.md`` gives the provenance. With ``--check`` nothing is
written: the rebuilt bytes are compared with the committed files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from tests import recordings

HERE: Final = Path(__file__).resolve().parent
EVT2_EVENTS: Final = 100_000
EVT3_BODY_BYTES: Final = 1 << 18


def split_header(data: bytes) -> int:
    """Offset of the body: header lines start with '%'; '% end' ends the header."""
    at = 0
    while data[at : at + 1] == b"%":
        end = data.index(b"\n", at) + 1
        line, at = data[at:end], end
        if line.strip() == b"% end":
            break
    return at


def evt2_excerpt() -> bytes:
    """``sparklers.raw``: its header and every word up to its 100,000th CD event."""
    data = recordings.path("sparklers.raw").read_bytes()
    body = split_header(data)
    words = np.frombuffer(data, dtype="<u4", offset=body, count=(len(data) - body) // 4)
    kind = words >> 28
    first_time_high = int(np.flatnonzero(kind == 8)[0])
    cd = np.flatnonzero(kind <= 1)
    last = int(cd[cd > first_time_high][EVT2_EVENTS - 1])
    return data[: body + 4 * (last + 1)]


def evt3_excerpt() -> bytes:
    """``active_marker.raw``: its header and the first 256 KiB of its body."""
    data = recordings.path("active_marker.raw").read_bytes()
    return data[: split_header(data) + EVT3_BODY_BYTES]


def sparklers_events() -> np.ndarray:
    """The events of ``sparklers_100k.evt2.raw``, decoded by the EVT adapter."""
    from frames2py.adapters import evt

    with evt.open(HERE / "sparklers_100k.evt2.raw", sensor_size=(640, 480)) as reader:
        return np.concatenate(list(reader))


def aedat4_rewrite() -> bytes:
    """The sparklers excerpt's events written by dv-processing's ``MonoCameraWriter``
    (event-only configuration, camera ``sparklers_cc0_derived``, 640x480, default settings)."""
    import dv_processing as dv

    events = sparklers_events()
    store = dv.EventStore()
    for t, x, y, p in zip(events["t"].tolist(), events["x"].tolist(), events["y"].tolist(), events["p"].tolist()):
        store.push_back(t, x, y, bool(p))
    target = HERE / "aedat4_rewrite.tmp.aedat4"
    writer = dv.io.MonoCameraWriter(str(target), dv.io.MonoCameraWriter.EventOnlyConfig("sparklers_cc0_derived", (640, 480)))
    writer.writeEvents(store)
    del writer
    try:
        return target.read_bytes()
    finally:
        target.unlink()


FIXTURES: Final[dict[str, Callable[[], bytes]]] = {
    "sparklers_100k.evt2.raw": evt2_excerpt,
    "active_marker_head.evt3.raw": evt3_excerpt,
    "sparklers_100k.aedat4": aedat4_rewrite,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.data.derive")
    parser.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    args = parser.parse_args(argv)
    status = 0
    for name, build in FIXTURES.items():
        data = build()
        digest = hashlib.sha256(data).hexdigest()
        target = HERE / name
        if args.check:
            same = target.is_file() and target.read_bytes() == data
            status |= not same
            print(f"{name}: {'identical' if same else 'DIFFERENT'} ({digest})")
        else:
            target.write_bytes(data)
            print(f"{name}: {len(data)} bytes, SHA-256 {digest}")
    return status


if __name__ == "__main__":
    sys.exit(main())
