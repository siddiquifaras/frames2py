"""Streaming NumPy decoders for EVT 2.0 and EVT 3.0 CD events.

``feed(chunk)`` takes the next bytes of the event stream (the file body after the
header), in reads of any size, and returns the CD events those bytes complete, as
``EVENT_DTYPE`` arrays in stream order. Decoder state, including a partial word, carries
from one ``feed()`` to the next, so the events never depend on how the stream is split.
A partial word left over at the end of the stream is never decoded.

Both decoders reconstruct timestamps and nothing more: no sorting, no clamping, no epoch
inference, no reset. A backward step in the source's time stays a backward step in the
output, except for the counter wraps defined below.

EVT 2.0 (32-bit little-endian words, type in bits 31..28):
- CD_OFF 0x0 / CD_ON 0x1: time bits 5..0 in bits 27..22, x in 21..11, y in 10..0.
- EVT_TIME_HIGH 0x8: time bits 33..6 in bits 27..0.
- An event's time is TIME_HIGH's value (bits 33..6) with the event's 6 low bits.
- A TIME_HIGH lower than the previous one by at least ``(2**28 - 1) * 64 - 10000`` µs is a
  counter wrap and adds 2**34 µs from then on. Any other backward step is kept.
- CD events before the first TIME_HIGH are dropped; their time is unknown. Other word
  types are ignored.

EVT 3.0 (16-bit little-endian words, type in bits 15..12):
- EVT_ADDR_Y 0x0 starts a CD row (y in bits 10..0). Type 0x1 is reserved: it starts a row
  in which nothing is emitted and vector words don't move the vector base.
- EVT_ADDR_X 0x2: one event at x (bits 10..0), polarity bit 11.
- VECT_BASE_X 0x3: vector base x (bits 10..0) and polarity (bit 11).
- VECT_12 0x4 / VECT_8 0x5: an event at base + i for each set bit i of the low 12 / 8
  bits, then the base moves on by 12 / 8.
- EVT_TIME_LOW 0x6 (bits 11..0) and EVT_TIME_HIGH 0x8 (time bits 23..12). An event's time
  is TIME_HIGH || TIME_LOW plus 2**24 µs per counter wrap.
- A TIME_HIGH lower than the previous one by more than 3840 (which includes 4095 -> 0) is
  a wrap. Every other backward step is kept as a backward step.
- A TIME_HIGH that changes the value sets TIME_LOW to 0 until the next TIME_LOW. A
  TIME_LOW lower than the one before it is kept.
- Decoding starts at the first TIME_HIGH; earlier words are skipped. After it, events in
  a row not started by an EVT_ADDR_Y, and vector words before the first VECT_BASE_X, emit
  nothing.
- Rows at any y are decoded; the sensor height is not applied here. Other word types
  are ignored.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from frames2py._events import EVENT_DTYPE, EventArray

EVT2_TIME_HIGH_MAX: Final = ((1 << 28) - 1) << 6
EVT2_WRAP_MIN_BACKSTEP: Final = EVT2_TIME_HIGH_MAX - 10_000
EVT2_WRAP: Final = 1 << 34

EVT3_WRAP_MIN_BACKSTEP: Final = 3841
EVT3_WRAP: Final = 1 << 24
EVT3_SLICE_WORDS: Final = 32_768
"""EVT 3.0 bodies are decoded in slices of this many words, one output array per slice."""

_NO_ROW: Final = 0x1000
"""Row state before the first EVT_ADDR_Y: a reserved-type row, in which nothing is emitted."""

_POPCOUNT12: Final = np.array([bin(v).count("1") for v in range(4096)], dtype=np.intp)
_BIT_START: Final = np.concatenate(([0], np.cumsum(_POPCOUNT12)[:-1])).astype(np.intp)
_BIT_OFFSETS: Final = np.array([i for v in range(4096) for i in range(12) if v >> i & 1], dtype=np.uint64)
"""For each 12-bit mask ``v``: its set bits, in order, at ``_BIT_START[v]``."""


def _empty() -> EventArray:
    return np.empty(0, dtype=EVENT_DTYPE)


class _Words:
    """Splits a byte stream into whole little-endian words, carrying a partial word."""

    def __init__(self, dtype: str) -> None:
        self._dtype = np.dtype(dtype)
        self._tail = b""

    def take(self, chunk: bytes | bytearray | memoryview) -> NDArray[np.unsignedinteger] | None:
        data = self._tail + bytes(chunk) if self._tail else chunk
        size = self._dtype.itemsize
        n = len(data) // size
        self._tail = bytes(memoryview(data)[n * size :])
        if n == 0:
            return None
        words: NDArray[np.unsignedinteger] = np.frombuffer(data, dtype=self._dtype, count=n)
        return words

    @property
    def pending_bytes(self) -> int:
        return len(self._tail)


class Evt2Decoder:
    """EVT 2.0 CD events; one array per ``feed()``."""

    def __init__(self) -> None:
        self._words = _Words("<u4")
        self._started = False
        self._time_high = 0  # the last TIME_HIGH's value << 6, without wraps
        self._wraps = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[EventArray]:
        words = self._words.take(chunk)
        if words is None:
            return []
        kind = words >> 28
        th_pos = np.flatnonzero(kind == 8)
        cd_pos = np.flatnonzero(kind <= 1)
        if len(th_pos):
            th = (words[th_pos] & 0x0FFFFFFF).astype(np.int64) << 6
            prev = np.empty_like(th)
            prev[0] = self._time_high if self._started else th[0]
            prev[1:] = th[:-1]
            wraps = self._wraps + np.cumsum(prev - th >= EVT2_WRAP_MIN_BACKSTEP)
            base = np.empty(len(th) + 1, dtype=np.uint64)
            base[0] = self._time_high + self._wraps * EVT2_WRAP
            base[1:] = th.astype(np.uint64) + wraps.astype(np.uint64) * np.uint64(EVT2_WRAP)
            per_segment = np.diff(np.searchsorted(cd_pos, th_pos), prepend=0, append=len(cd_pos))
            if not self._started:
                cd_pos = cd_pos[per_segment[0] :]
                per_segment[0] = 0
            self._started = True
            self._time_high, self._wraps = int(th[-1]), int(wraps[-1])
            t = np.repeat(base, per_segment)
        elif self._started:
            t = np.full(len(cd_pos), self._time_high + self._wraps * EVT2_WRAP, dtype=np.uint64)
        else:
            return []
        cd = words[cd_pos]
        out = np.empty(len(cd), dtype=EVENT_DTYPE)
        out["t"] = t + ((cd >> 22) & 0x3F)
        out["x"] = (cd >> 11) & 0x7FF
        out["y"] = cd & 0x7FF
        out["p"] = kind[cd_pos] & 1
        return [out] if len(out) else []

    @property
    def pending_bytes(self) -> int:
        """Bytes of an incomplete word waiting for the next ``feed()``."""
        return self._words.pending_bytes


class Evt3Decoder:
    """EVT 3.0 CD events; one array per slice of ``EVT3_SLICE_WORDS`` words."""

    def __init__(self) -> None:
        self._words = _Words("<u2")
        self._started = False
        self._time_high = 0
        self._time_low = 0
        self._wraps = 0
        self._row = _NO_ROW  # the word that started the current row
        self._has_base = False
        self._base = 0  # vector base x, moved on by the vector words since VECT_BASE_X
        self._base_polarity = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[EventArray]:
        words = self._words.take(chunk)
        if words is None:
            return []
        out = []
        for start in range(0, len(words), EVT3_SLICE_WORDS):
            events = self._decode(words[start : start + EVT3_SLICE_WORDS])
            if len(events):
                out.append(events)
        return out

    @property
    def pending_bytes(self) -> int:
        """Bytes of an incomplete word waiting for the next ``feed()``."""
        return self._words.pending_bytes

    def _decode(self, w: NDArray[np.unsignedinteger]) -> EventArray:
        kind = w >> 12
        if not self._started:
            first = np.flatnonzero(kind == 8)
            if not len(first):
                return _empty()
            start = int(first[0])
            # As if the TIME_HIGH before it were one lower: never a wrap, and TIME_LOW starts at 0.
            self._started, self._time_high = True, max(int(w[start] & 0xFFF) - 1, 0)
            w, kind = w[start:], kind[start:]

        is_time = (kind == 6) | (kind == 8)
        is_row = kind <= 1
        ev_pos = np.flatnonzero((kind == 2) | (kind == 4) | (kind == 5))
        # Time words (low 16 bits) and row words (high 16 bits) before each event word.
        # Exact: a slice has at most 2**15 words.
        packed = np.cumsum(is_time.astype(np.uint32) | (is_row.astype(np.uint32) << 16), dtype=np.uint32)[ev_pos]
        times_before = packed & 0xFFFF
        rows_before = packed >> 16

        t_after = self._times(w[is_time])
        t_ev = t_after[times_before]

        rows = np.empty(int(np.count_nonzero(is_row)) + 1, dtype=np.uint16)
        rows[0] = self._row
        rows[1:] = w[is_row]
        self._row = int(rows[-1])
        row_ev = rows[rows_before]
        y_ev = row_ev & 0x7FF
        cd_ev = row_ev < 0x1000

        ev_kind = kind[ev_pos]
        is_vector = ev_kind != 2
        if not is_vector.any():
            events = w[ev_pos]
            last_base = np.flatnonzero(kind == 3)
            if len(last_base):
                word = int(w[last_base[-1]])
                self._has_base, self._base, self._base_polarity = True, word & 0x7FF, word >> 11 & 1
            if not cd_ev.all():
                events, t_ev, y_ev = events[cd_ev], t_ev[cd_ev], y_ev[cd_ev]
            out = np.empty(len(events), dtype=EVENT_DTYPE)
            out["t"] = t_ev
            out["x"] = events & 0x7FF
            out["y"] = y_ev
            out["p"] = events >> 11 & 1
            return out

        base, polarity, has_base = self._vector_bases(w, kind, cd_ev[is_vector])
        vector_words = w[ev_pos[is_vector]]
        valid = vector_words & np.where(ev_kind[is_vector] == 4, 0xFFF, 0xFF).astype(np.uint16)
        per_vector = _POPCOUNT12[valid]
        per_vector[~(cd_ev[is_vector] & has_base)] = 0

        counts = cd_ev.astype(np.intp)
        counts[is_vector] = per_vector
        total = int(counts.sum())
        out = np.empty(total, dtype=EVENT_DTYPE)
        if not total:
            return out
        out["t"] = np.repeat(t_ev, counts)
        out["y"] = np.repeat(y_ev, counts)
        single = np.repeat(~is_vector, counts)
        x = np.empty(total, dtype=np.uint64)
        p = np.empty(total, dtype=np.uint8)
        singles = w[ev_pos[~is_vector & cd_ev]]
        x[single] = singles & 0x7FF
        p[single] = singles >> 11 & 1
        emitting = per_vector > 0
        masks, n_bits = valid[emitting], per_vector[emitting]
        owner = np.repeat(np.arange(len(masks)), n_bits)
        rank = np.arange(len(owner)) - np.repeat(np.cumsum(n_bits) - n_bits, n_bits)
        offsets = _BIT_OFFSETS[np.repeat(_BIT_START[masks], n_bits) + rank]
        vector_x = base[emitting][owner] + offsets
        if len(vector_x) and int(vector_x.max()) > 0xFFFF:
            raise ValueError("malformed EVT 3.0 stream: a vector event's x exceeds 65535")
        x[~single] = vector_x
        p[~single] = polarity[emitting][owner]
        out["x"] = x
        out["p"] = p
        return out

    def _times(self, tw: NDArray[np.unsignedinteger]) -> NDArray[np.uint64]:
        """Event time after each time word; index 0 is the time carried into the slice."""
        t_after = np.empty(len(tw) + 1, dtype=np.uint64)
        t_after[0] = (self._wraps << 24) | (self._time_high << 12) | self._time_low
        if not len(tw):
            return t_after
        is_high = tw >= 0x8000
        value = (tw & 0xFFF).astype(np.int64)
        high_idx = np.flatnonzero(is_high)
        high = np.empty(len(high_idx) + 1, dtype=np.int64)
        high[0] = self._time_high
        high[1:] = value[high_idx]
        wraps = np.empty_like(high)
        wraps[0] = self._wraps
        np.cumsum(high[:-1] - high[1:] >= EVT3_WRAP_MIN_BACKSTEP, out=wraps[1:])
        wraps[1:] += self._wraps
        # The TIME_HIGH in force at each time word, and the wraps counted up to it.
        n_high = np.cumsum(is_high)
        high_at, wraps_at = high[n_high], wraps[n_high]
        # TIME_LOW in force: the last TIME_LOW, unless a value-changing TIME_HIGH came after it.
        changes = np.zeros(len(tw), dtype=bool)
        changes[high_idx[high[1:] != high[:-1]]] = True
        last = np.maximum.accumulate(np.where(~is_high | changes, np.arange(len(tw)), -1))
        low_at = np.where(last >= 0, np.where(is_high[last], 0, value[last]), self._time_low)
        t_after[1:] = (wraps_at.astype(np.uint64) << 24) | (high_at.astype(np.uint64) << 12) | low_at.astype(np.uint64)
        self._time_high, self._wraps, self._time_low = int(high_at[-1]), int(wraps_at[-1]), int(low_at[-1])
        return t_after

    def _vector_bases(
        self, w: NDArray[np.unsignedinteger], kind: NDArray[np.integer], cd_vector: NDArray[np.bool_]
    ) -> tuple[NDArray[np.uint64], NDArray[np.uint8], NDArray[np.bool_]]:
        """Base x, polarity and whether a base exists, for each vector word in the slice.

        Computed along the subsequence of VECT_BASE_X, VECT_12 and VECT_8 words: a vector
        word's base is the last VECT_BASE_X plus the steps of the vector words between them.
        Vector words outside a CD row take no step.
        """
        sub = w[(kind >= 3) & (kind <= 5)]
        sub_kind = sub >> 12
        is_base = sub_kind == 3
        step = np.zeros(len(sub), dtype=np.int64)
        step[~is_base] = np.where(sub_kind[~is_base] == 4, 12, 8) * cd_vector
        steps_through = np.cumsum(step)
        steps_before = steps_through - step
        n_base = np.cumsum(is_base)  # VECT_BASE_X words up to each, 0 meaning the carried base
        base_x = np.empty(int(n_base[-1]) + 1, dtype=np.int64)
        base_steps = np.empty_like(base_x)
        base_pol = np.empty(len(base_x), dtype=np.uint8)
        base_x[0], base_steps[0], base_pol[0] = self._base, 0, self._base_polarity
        base_words = sub[is_base]
        base_x[1:] = base_words & 0x7FF
        base_steps[1:] = steps_before[is_base]
        base_pol[1:] = base_words >> 11 & 1
        owner = n_base[~is_base]
        base = base_x[owner] + steps_before[~is_base] - base_steps[owner]
        has_base = owner > 0 if not self._has_base else np.ones(len(owner), dtype=bool)
        last = int(n_base[-1])
        self._base = int(base_x[last] + steps_through[-1] - base_steps[last])
        self._base_polarity = int(base_pol[last])
        self._has_base = self._has_base or last > 0
        return base.astype(np.uint64), base_pol[owner], has_base
