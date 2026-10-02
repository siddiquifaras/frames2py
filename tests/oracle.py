"""Reference model of the v1 accumulation semantics, for tests.

This is an oracle, not an implementation. It keeps a log of every accumulated
event as plain Python values and computes each representation from that log,
straight from its definition:

- ``event_count``: events per pixel since the window opened, modulo 2^32.
- ``polarity``: the same, split into channel 0 (``p == 0``) and channel 1
  (``p != 0``).
- ``time_surface``: the largest ``t`` per pixel since reset; 0 if none.
- ``exp_decay``: every accepted call applies one decay step before its events
  are added, so an event accumulated in call ``c`` has weight
  ``decay ** (C - c)`` after ``C`` accepted calls.
- ``timestamp_decay``: ``Σ exp(-(T - t_i) / tau_us)`` at the watermark ``T``,
  with ``T - t_i`` exact integer arithmetic, the ratio rounded once, and each
  pixel summed with ``math.fsum``.

Accumulator-level rules, applied the same way for every kernel:

- A call with any ``t >= 2**63`` raises ``ValueError`` before anything changes.
  The check covers out-of-bounds events too.
- An event is in bounds iff ``0 <= x < width`` and ``0 <= y < height``.
  Out-of-bounds events are counted and otherwise ignored: they don't advance
  the watermark or touch any representation.
- The watermark is the largest ``t`` among accumulated in-bounds events since
  reset, or ``None`` before the first one. Closing a window doesn't change it.
- ``close_window()`` is what publication does to kernel state: windowed kernels
  start a new window, running kernels are unchanged.

Out of scope: structural validation, field dtype checking,
Engine cadence, and statistics other than the accumulated and out-of-bounds
counts. Fields are read by name, so extra fields are ignored.

The oracle is slow on purpose. Use it on small inputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

KERNELS: Final = ("event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay")
WINDOWED: Final = frozenset({"event_count", "polarity"})
TIMESTAMP_LIMIT: Final = 2**63
COUNT_MODULUS: Final = 2**32


def counts_to_uint32(counts: NDArray[Any]) -> NDArray[np.uint32]:
    """Exact integer counts as observable ``uint32`` output: wrap, never saturate."""
    exact = np.asarray(counts, dtype=object)
    wrapped = [int(c) % COUNT_MODULUS for c in exact.flat]
    return np.array(wrapped, dtype=np.uint32).reshape(exact.shape)


class ReferenceAccumulator:
    """Accumulates events through one kernel and reads its representation."""

    def __init__(
        self,
        kernel: str,
        sensor_size: tuple[int, int],
        *,
        decay: float | None = None,
        tau_us: float | None = None,
    ) -> None:
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        width, height = sensor_size
        if width < 1 or height < 1:
            raise ValueError(f"sensor_size must be positive, got {sensor_size}")
        if kernel == "exp_decay":
            if decay is None or not math.isfinite(decay):
                raise ValueError(f"exp_decay needs a finite decay, got {decay!r}")
        elif decay is not None:
            raise ValueError(f"{kernel} takes no decay")
        if kernel == "timestamp_decay":
            if tau_us is None or not math.isfinite(tau_us) or not tau_us > 0:
                raise ValueError(f"timestamp_decay needs a finite tau_us > 0, got {tau_us!r}")
        elif tau_us is not None:
            raise ValueError(f"{kernel} takes no tau_us")
        self.kernel = kernel
        self.width = width
        self.height = height
        self.decay = decay
        self.tau_us = tau_us
        self.reset()

    def reset(self) -> None:
        # (t, x, y, on, call) per accumulated in-bounds event, in arrival order.
        self._log: list[tuple[int, int, int, bool, int]] = []
        self._window_start = 0
        self._calls = 0
        self.watermark: int | None = None
        self.accumulated = 0
        self.out_of_bounds = 0

    def accumulate(self, events: NDArray[Any]) -> None:
        columns = [np.asarray(events[name]).tolist() for name in ("t", "x", "y", "p")]
        rows = list(zip(*columns))
        if any(int(t) >= TIMESTAMP_LIMIT for t, _, _, _ in rows):
            raise ValueError("event timestamp >= 2**63; the whole call is rejected")
        self._calls += 1
        for t, x, y, p in rows:
            t, x, y, p = int(t), int(x), int(y), int(p)
            if not (0 <= x < self.width and 0 <= y < self.height):
                self.out_of_bounds += 1
                continue
            self._log.append((t, x, y, p != 0, self._calls))
            self.accumulated += 1
            self.watermark = t if self.watermark is None else max(self.watermark, t)

    def close_window(self) -> None:
        if self.kernel in WINDOWED:
            self._window_start = len(self._log)

    def read(self, at: int | None = None) -> NDArray[Any]:
        """The observable output: its shape and dtype are part of the contract.

        ``at`` evaluates the representation at that time instead of the watermark; only
        ``timestamp_decay`` depends on it.
        """
        if self.kernel == "event_count":
            return self._read_event_count()
        if self.kernel == "polarity":
            return self._read_polarity()
        if self.kernel == "time_surface":
            return self._read_time_surface()
        return self.read_exact(at).astype(np.float32)

    def read_exact(self, at: int | None = None) -> NDArray[np.float64]:
        """Decay kernels only: the float64 value of each pixel before float32 rounding."""
        if self.kernel == "exp_decay":
            return self._exp_decay_exact()
        if self.kernel == "timestamp_decay":
            return self._timestamp_decay_exact(self.watermark if at is None else at)
        raise ValueError(f"{self.kernel} has no inexact output")

    def _read_event_count(self) -> NDArray[np.uint32]:
        counts = np.zeros((self.height, self.width), dtype=object)
        for _, x, y, _, _ in self._log[self._window_start :]:
            counts[y, x] += 1
        return counts_to_uint32(counts)

    def _read_polarity(self) -> NDArray[np.uint32]:
        counts = np.zeros((self.height, self.width, 2), dtype=object)
        for _, x, y, on, _ in self._log[self._window_start :]:
            counts[y, x, 1 if on else 0] += 1
        return counts_to_uint32(counts)

    def _read_time_surface(self) -> NDArray[np.uint64]:
        surface = np.zeros((self.height, self.width), dtype=np.uint64)
        for t, x, y, _, _ in self._log:
            if t > int(surface[y, x]):
                surface[y, x] = t
        return surface

    def _exp_decay_exact(self) -> NDArray[np.float64]:
        assert self.decay is not None
        terms: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        for _, x, y, _, call in self._log:
            terms[(y, x)].append(self.decay ** (self._calls - call))
        return self._sum(terms)

    def _timestamp_decay_exact(self, at: int | None) -> NDArray[np.float64]:
        assert self.tau_us is not None
        terms: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        if at is not None:
            tau = Fraction(self.tau_us)
            for t, x, y, _, _ in self._log:
                terms[(y, x)].append(math.exp(-float(Fraction(at - t) / tau)))
        return self._sum(terms)

    def _sum(self, terms: dict[tuple[int, int], list[float]]) -> NDArray[np.float64]:
        surface = np.zeros((self.height, self.width), dtype=np.float64)
        for (y, x), values in terms.items():
            surface[y, x] = math.fsum(values)
        return surface


def float32_ulp_distance(actual: NDArray[Any], expected: NDArray[Any]) -> NDArray[np.int64]:
    """Elementwise distance in float32 units in the last place.

    Both arguments are rounded to float32 first. Adjacent float32 values are 1
    apart; +0.0 and -0.0 are 0 apart. NaN is not supported.
    """
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(expected, dtype=np.float32)
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN has no ULP distance")

    def ordered(v: NDArray[np.float32]) -> NDArray[np.int64]:
        bits = v.view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    distance: NDArray[np.int64] = np.abs(ordered(a) - ordered(b))
    return distance
