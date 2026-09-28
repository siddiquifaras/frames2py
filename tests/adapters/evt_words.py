"""EVT 2.0 and 3.0 words and files built from the format pages, and a per-word reference decoder.

The reference decoder is a plain loop over words, one state machine step per word, written
from the decoding rules rather than from the vectorised decoder, so tests can compare the two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])

EVT2_WRAP_MIN_BACKSTEP = ((1 << 28) - 1) * 64 - 10_000


# EVT 2.0: 32-bit words, type in bits 31..28
def evt2_time_high(value: int) -> int:
    """EVT_TIME_HIGH carrying time bits 33..6."""
    return 0x8 << 28 | (value & 0x0FFFFFFF)


def evt2_cd(x: int, y: int, *, on: bool, low: int = 0) -> int:
    """CD_ON / CD_OFF at (x, y) with time bits 5..0."""
    return (0x1 if on else 0x0) << 28 | (low & 0x3F) << 22 | (x & 0x7FF) << 11 | (y & 0x7FF)


def evt2_other(kind: int, payload: int = 0) -> int:
    return (kind & 0xF) << 28 | (payload & 0x0FFFFFFF)


# EVT 3.0: 16-bit words, type in bits 15..12
def evt3_y(y: int) -> int:
    return 0x0 << 12 | (y & 0x7FF)


def evt3_reserved_row(y: int = 0) -> int:
    """A type 0x1 row word."""
    return 0x1 << 12 | (y & 0x7FF)


def evt3_x(x: int, *, on: bool) -> int:
    return 0x2 << 12 | int(on) << 11 | (x & 0x7FF)


def evt3_base(x: int, *, on: bool) -> int:
    return 0x3 << 12 | int(on) << 11 | (x & 0x7FF)


def evt3_vect12(mask: int) -> int:
    return 0x4 << 12 | (mask & 0xFFF)


def evt3_vect8(mask: int) -> int:
    return 0x5 << 12 | (mask & 0xFF)


def evt3_time_low(value: int) -> int:
    return 0x6 << 12 | (value & 0xFFF)


def evt3_time_high(value: int) -> int:
    return 0x8 << 12 | (value & 0xFFF)


def evt3_other(kind: int, payload: int = 0) -> int:
    return (kind & 0xF) << 12 | (payload & 0xFFF)


def body(words: list[int], version: str) -> bytes:
    return np.asarray(words, dtype="<u4" if version == "2.0" else "<u2").tobytes()


def header(version: str, geometry: tuple[int, int] | None = (640, 480), *, extra: tuple[str, ...] = ()) -> bytes:
    fmt = "EVT2" if version == "2.0" else "EVT3"
    lines = [f"% evt {version}", f"% format {fmt}"]
    if geometry is not None:
        lines.append(f"% geometry {geometry[0]}x{geometry[1]}")
    lines += list(extra)
    return ("\n".join(lines) + "\n").encode()


def write_raw(path: Path, words: list[int], version: str, geometry: tuple[int, int] | None = (640, 480)) -> Path:
    path.write_bytes(header(version, geometry) + body(words, version))
    return path


def events(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.empty(len(rows), dtype=EVENT_DTYPE)
    for i, (t, x, y, p) in enumerate(rows):
        out[i] = (t, x, y, p)
    return out


def reference_decode(data: bytes, version: str) -> np.ndarray:
    """Decode a whole EVT body (no header) one word at a time."""
    rows: list[tuple[int, int, int, int]] = []
    size = 4 if version == "2.0" else 2
    words = np.frombuffer(data[: len(data) // size * size], dtype="<u4" if version == "2.0" else "<u2").tolist()
    if version == "2.0":
        started, high, wraps = False, 0, 0
        for word in words:
            kind = word >> 28
            if kind == 0x8:
                value = (word & 0x0FFFFFFF) << 6
                if started and high - value >= EVT2_WRAP_MIN_BACKSTEP:
                    wraps += 1
                started, high = True, value
            elif kind in (0x0, 0x1) and started:
                t = high + wraps * (1 << 34) + (word >> 22 & 0x3F)
                rows.append((t, word >> 11 & 0x7FF, word & 0x7FF, kind))
        return events(rows)

    started = False
    high = low = wraps = 0
    in_cd_row, y = False, 0
    has_base, base, polarity = False, 0, 0
    for word in words:
        kind, value = word >> 12, word & 0xFFF
        if not started:
            if kind != 0x8:
                continue
            started, high = True, max(value - 1, 0)
        t = (wraps << 24) | (high << 12) | low
        if kind == 0x8:
            if high - value > 3840:
                wraps += 1
            if value != high:
                low = 0
            high = value
        elif kind == 0x6:
            low = value
        elif kind == 0x0:
            in_cd_row, y = True, word & 0x7FF
        elif kind == 0x1:
            in_cd_row = False
        elif kind == 0x3:
            has_base, base, polarity = True, word & 0x7FF, word >> 11 & 1
        elif kind == 0x2 and in_cd_row:
            rows.append((t, word & 0x7FF, y, word >> 11 & 1))
        elif kind in (0x4, 0x5) and in_cd_row:
            width = 12 if kind == 0x4 else 8
            if has_base:
                for i in range(width):
                    if word >> i & 1:
                        if base + i > 0xFFFF:
                            raise ValueError("vector event x beyond 65535")
                        rows.append((t, base + i, y, polarity))
            base += width
    return events(rows)
