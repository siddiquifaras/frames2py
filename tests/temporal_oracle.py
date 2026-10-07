"""Reference model of the temporal kernels and of ``frames2py.replay.windows``, for tests.

This is an oracle, not an implementation. It keeps every accumulated in-bounds event as
plain Python values and computes each frame from that log, straight from the definitions:

- Grid bin ``k`` is ``[k * bin_us, (k + 1) * bin_us)``; ``k_T = T // bin_us`` for the
  watermark ``T``. Only completed bins, ``k < k_T``, are ever shown.
- ``stacked_histogram``: ``(2, bins, H, W)``, index ``j`` is grid bin ``k_T - bins + j``;
  channel 0 counts ``p == 0``, channel 1 every other ``p``; counts modulo 2^32.
- ``voxel_grid``: ``(bins, H, W)``, knot ``j`` at ``(k_T - bins + 1 + j) * bin_us``. The span is
  grid bins ``k_T - bins + 1 ... k_T - 1``. An event in the span with ``q, r = divmod(t, bin_us)``
  and sign ``s`` (+1 for ``p != 0``, -1 for ``p == 0``) adds ``s * (bin_us - r)`` to knot ``q``
  and ``s * r`` to knot ``q + 1``. The numerator ``N`` is that unbounded integer sum; the
  output is ``float32(float64(n_signed) / float64(bin_us))``, where ``n_signed`` is
  ``N mod 2^64`` read as two's-complement int64.
- Every value is 0 while ``T`` is ``None``.

Accumulator-level rules are those of ``tests/oracle.py``: a call with any ``t >= 2**63``
is rejected whole; out-of-bounds events are counted and otherwise ignored; the watermark
is the largest in-bounds ``t``.

``windows`` is the offline helper's definition over an event stream in arrival order.

The oracle is slow on purpose. Use it on small inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

TEMPORAL_KERNELS: Final = ("stacked_histogram", "voxel_grid")
TIMESTAMP_LIMIT: Final = 2**63
COUNT_MODULUS: Final = 2**32
NUMERATOR_MODULUS: Final = 2**64
BIN_US_LIMIT: Final = 2**28


def signed_reading(numerator: int) -> int:
    """``numerator mod 2**64``, read as a two's-complement 64-bit integer."""
    stored = numerator % NUMERATOR_MODULUS
    return stored - NUMERATOR_MODULUS if stored >= 2**63 else stored


def voxel_value(numerator: int, bin_us: int) -> np.float32:
    """The observable float32 for an exact numerator: ``float32(float64(n_signed) / float64(bin_us))``."""
    return np.float32(float(signed_reading(numerator)) / float(bin_us))


def correctly_rounded_float32(value: Fraction) -> np.float32:
    """The float32 nearest to ``value``, ties to even. Normal range only."""
    if value == 0:
        return np.float32(0.0)
    sign = -1 if value < 0 else 1
    a, b = abs(value.numerator), value.denominator
    # s such that 2**23 <= a * 2**s / b < 2**24
    s = 23 - (a.bit_length() - b.bit_length())

    def scaled(shift: int) -> tuple[int, int]:
        return (a << shift, b) if shift >= 0 else (a, b << -shift)

    while scaled(s)[0] < scaled(s)[1] << 23:
        s += 1
    while scaled(s)[0] >= scaled(s)[1] << 24:
        s -= 1
    num, den = scaled(s)
    m, r = divmod(num, den)
    if 2 * r > den or (2 * r == den and m % 2 == 1):
        m += 1
    exponent = (m.bit_length() - 1) - s  # value is m * 2**-s, with 2**23 <= m <= 2**24
    if not -126 <= exponent <= 127:
        raise ValueError(f"{value} is outside float32's normal range")
    return np.float32(sign * m * 2.0**-s)  # exact: m has at most 25 significant bits


class Reference(Protocol):
    def accumulate(self, events: NDArray[Any]) -> None: ...
    def close_window(self) -> None: ...
    def read(self, at: int | None = None) -> NDArray[Any]: ...


class TemporalReference:
    """Accumulates events and reads one temporal kernel's representation."""

    def __init__(self, kernel: str, sensor_size: tuple[int, int], *, bins: int, bin_us: int) -> None:
        if kernel not in TEMPORAL_KERNELS:
            raise ValueError(f"unknown temporal kernel {kernel!r}")
        if bins < (2 if kernel == "voxel_grid" else 1) or not 1 <= bin_us < BIN_US_LIMIT:
            raise ValueError(f"invalid parameters bins={bins}, bin_us={bin_us}")
        self.kernel = kernel
        self.width, self.height = sensor_size
        self.bins = bins
        self.bin_us = bin_us
        self.reset()

    def reset(self) -> None:
        self._log: list[tuple[int, int, int, bool]] = []  # (t, x, y, on), in arrival order
        self.watermark: int | None = None
        self.out_of_bounds = 0

    def accumulate(self, events: NDArray[Any]) -> None:
        rows = list(zip(*(np.asarray(events[name]).tolist() for name in ("t", "x", "y", "p"))))
        if any(int(t) >= TIMESTAMP_LIMIT for t, _, _, _ in rows):
            raise ValueError("event timestamp >= 2**63; the whole call is rejected")
        for t, x, y, p in rows:
            t, x, y, p = int(t), int(x), int(y), int(p)
            if not (0 <= x < self.width and 0 <= y < self.height):
                self.out_of_bounds += 1
                continue
            self._log.append((t, x, y, p != 0))
            self.watermark = t if self.watermark is None else max(self.watermark, t)

    def close_window(self) -> None:
        pass  # both kernels are running

    def counts(self, at: int | None = None) -> NDArray[Any]:
        """``stacked_histogram`` only: exact counts, as Python ints, before the 2^32 wrap."""
        assert self.kernel == "stacked_histogram"
        counts = np.zeros((2, self.bins, self.height, self.width), dtype=object)
        T = self.watermark if at is None else at
        if T is None:
            return counts
        oldest = T // self.bin_us - self.bins
        for t, x, y, on in self._log:
            j = t // self.bin_us - oldest
            if 0 <= j < self.bins:
                counts[1 if on else 0, j, y, x] += 1
        return counts

    def numerators(self, at: int | None = None) -> NDArray[Any]:
        """``voxel_grid`` only: the exact numerators N, as unbounded Python ints."""
        assert self.kernel == "voxel_grid"
        numerators = np.zeros((self.bins, self.height, self.width), dtype=object)
        T = self.watermark if at is None else at
        if T is None:
            return numerators
        first_knot = T // self.bin_us - self.bins + 1
        for t, x, y, on in self._log:
            q, r = divmod(t, self.bin_us)
            if first_knot <= q < first_knot + self.bins - 1:  # bins k_T - bins + 1 ... k_T - 1
                s = 1 if on else -1
                numerators[q - first_knot, y, x] += s * (self.bin_us - r)
                numerators[q + 1 - first_knot, y, x] += s * r
        return numerators

    def read(self, at: int | None = None) -> NDArray[Any]:
        """The observable output, evaluated at ``at`` instead of the watermark when given."""
        if self.kernel == "stacked_histogram":
            exact = self.counts(at)
            wrapped = [int(c) % COUNT_MODULUS for c in exact.flat]
            return np.array(wrapped, dtype=np.uint32).reshape(exact.shape)
        exact = self.numerators(at)
        values = [voxel_value(int(n), self.bin_us) for n in exact.flat]
        return np.array(values, dtype=np.float32).reshape(exact.shape)


def windows(
    batches: Iterable[NDArray[Any]],
    sensor_size: tuple[int, int],
    make_reference: Callable[[], Reference],
    every_us: int,
) -> list[tuple[int, NDArray[Any]]]:
    """The offline helper's frames: ``(boundary, frame)`` for every boundary the watermark passes.

    Boundaries are the multiples of ``every_us``. The frame at a boundary is read at that
    boundary, just before the first in-bounds event that brings the watermark to it or past
    it is accumulated; the first is the first boundary after the first in-bounds event.
    Windows close after each frame. Boundaries the watermark never reaches have no frame.

    ``exp_decay`` is rejected with ``TypeError``: its decay is per call, so its frames would
    depend on where the helper splits batches.
    """
    width, height = sensor_size
    reference = make_reference()
    if getattr(reference, "kernel", None) == "exp_decay":
        raise TypeError("windows() doesn't accept exp_decay; use timestamp_decay")
    watermark: int | None = None
    boundary = 0
    out: list[tuple[int, NDArray[Any]]] = []
    for batch in batches:
        if len(batch) and int(np.asarray(batch["t"]).max()) >= TIMESTAMP_LIMIT:
            raise ValueError("event timestamp >= 2**63; the whole batch is rejected")
        events = np.asarray(batch)
        for i in range(len(events)):
            event = events[i : i + 1]
            t, x, y = int(event["t"][0]), int(event["x"][0]), int(event["y"][0])
            if 0 <= x < width and 0 <= y < height:
                if watermark is None:
                    watermark, boundary = t, (t // every_us + 1) * every_us
                else:
                    watermark = max(watermark, t)
                    while watermark >= boundary:
                        out.append((boundary, reference.read(at=boundary)))
                        reference.close_window()
                        boundary += every_us
            reference.accumulate(event)
    return out
