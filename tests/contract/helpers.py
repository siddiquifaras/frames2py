"""Event builders, kernel configurations and oracle comparisons for the v1 suite."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tests.contract.api import EVENT_DTYPE, impl
from tests.oracle import ReferenceAccumulator, float32_ulp_distance

SENSOR = (6, 4)  # width 6, height 4: frames are (4, 6)

GIL_ENABLED: bool = getattr(sys, "_is_gil_enabled", lambda: True)()

VERIFIED_FREE_THREADED = {(3, 14)}
"""Free-threaded minor versions the publisher's handoff is verified on."""

SUPPORTED_RUNTIME = GIL_ENABLED or tuple(sys.version_info[:2]) in VERIFIED_FREE_THREADED
"""Whether the Engine accepts this interpreter: any build with the GIL, or a verified
free-threaded minor version with it disabled."""

TIMESTAMP_DECAY_MAX_ULP = 1
"""Tolerance on this suite's workloads, not an API guarantee."""

EXP_DECAY_MAX_ULP = 1
"""Tolerance on this suite's workloads, not an API guarantee."""


def events(*rows: tuple[int, int, int, int]) -> NDArray[Any]:
    return np.array(list(rows), dtype=EVENT_DTYPE)


BACKWARD_JUMP = (
    events((10**12, 1, 1, 0), (10**12 - 3, 4, 2, 1)),
    events((5, 1, 1, 1), (7, 3, 2, 0)),
)
"""A long-running source clock, then the same source after its clock restarted near 0."""

FORWARD_SPIKE = (
    events((1_000, 1, 1, 0), (1_002, 3, 2, 1)),
    events((2**62, 5, 3, 0)),
    events((1_010, 1, 1, 1)),
)
"""Ordinary timestamps, one far-future timestamp, then ordinary timestamps again."""


def random_events(
    seed: int,
    n: int,
    sensor: tuple[int, int] = SENSOR,
    t_max: int = 1_000,
    out_of_bounds: bool = True,
) -> NDArray[Any]:
    """Unordered timestamps, every polarity value, some events out of bounds."""
    rng = np.random.default_rng(seed)
    extra = 2 if out_of_bounds else 0
    out = np.empty(n, dtype=EVENT_DTYPE)
    out["t"] = rng.integers(0, t_max, size=n)
    out["x"] = rng.integers(0, sensor[0] + extra, size=n)
    out["y"] = rng.integers(0, sensor[1] + extra, size=n)
    out["p"] = rng.integers(0, 256, size=n)
    return out


def partitions(batch: NDArray[Any], seed: int, parts: int = 5) -> list[NDArray[Any]]:
    """*batch* shuffled and cut into *parts* contiguous calls."""
    rng = np.random.default_rng(10_000 + seed)
    shuffled = batch[rng.permutation(len(batch))]
    cuts = np.sort(rng.choice(np.arange(1, len(batch)), size=parts - 1, replace=False))
    return [np.ascontiguousarray(part) for part in np.split(shuffled, cuts)]


@dataclass(frozen=True)
class KernelCase:
    """One v1 kernel: how to configure it and the matching oracle."""

    name: str
    make: Callable[[], Any]
    oracle_params: dict[str, float]
    shape: Callable[[tuple[int, int]], tuple[int, ...]]
    dtype: type
    windowed: bool

    def oracle(self, sensor: tuple[int, int] = SENSOR) -> ReferenceAccumulator:
        return ReferenceAccumulator(self.name, sensor, **self.oracle_params)

    def accumulator(self, sensor: tuple[int, int] = SENSOR) -> Any:
        return impl.Accumulator(sensor, self.make())

    def engine(self, sensor: tuple[int, int] = SENSOR, interval_ms: float = 0.0) -> Any:
        return impl.Engine(sensor, self.make(), snapshot_interval_ms=interval_ms)


def _hw(sensor: tuple[int, int]) -> tuple[int, ...]:
    return (sensor[1], sensor[0])


def _hw2(sensor: tuple[int, int]) -> tuple[int, ...]:
    return (sensor[1], sensor[0], 2)


KERNELS = {
    case.name: case
    for case in (
        KernelCase("event_count", lambda: impl.EventCount(), {}, _hw, np.uint32, True),
        KernelCase("polarity", lambda: impl.Polarity(), {}, _hw2, np.uint32, True),
        KernelCase("time_surface", lambda: impl.TimeSurface(), {}, _hw, np.uint64, False),
        KernelCase("exp_decay", lambda: impl.ExpDecay(0.5), {"decay": 0.5}, _hw, np.float32, False),
        KernelCase(
            "timestamp_decay", lambda: impl.TimestampDecay(10.0), {"tau_us": 10.0}, _hw, np.float32, False
        ),
    )
}
ALL = list(KERNELS)
EXACT = ["event_count", "polarity", "time_surface"]
ORDER_INVARIANT = ["event_count", "polarity", "time_surface", "timestamp_decay"]


def assert_matches(frame: NDArray[Any], oracle: ReferenceAccumulator) -> None:
    """*frame* equals the oracle: exactly for integer kernels, within the stated ULP
    tolerance of the correctly rounded value for the decay kernels."""
    expected = oracle.read()
    assert frame.shape == expected.shape
    assert frame.dtype == expected.dtype
    if oracle.kernel in EXACT:
        np.testing.assert_array_equal(frame, expected)
        return
    limit = TIMESTAMP_DECAY_MAX_ULP if oracle.kernel == "timestamp_decay" else EXP_DECAY_MAX_ULP
    distance = float32_ulp_distance(frame, oracle.read_exact())
    worst = int(distance.max()) if distance.size else 0
    assert worst <= limit, f"{oracle.kernel}: {worst} ULP from the oracle (limit {limit})"
