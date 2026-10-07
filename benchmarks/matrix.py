"""Benchmark cells: the hard performance gate, and characterization suites.

A cell is one (kernel, resolution, batch size, publication interval,
distribution) condition plus the seed of its event stream. Whether a cell is a
hard-gate cell is decided by membership in the gate definition below, never by
which suite it came from, so characterization cells that happen to share a gate
condition are recognised as such and nothing outside the gate is.

Gate definition: the five v1 kernels, three resolutions, five
batch/interval conditions and two distributions, 150 cells, each needing
>= 20M events/s at both kernel level and ``Engine.ingest()`` level on the
reference machine. 10k events at 0 ms is not a gate condition.

The temporal-kernel gate (``benchmarks/temporal_gate_preregistration.md``) has the same
grid over five temporal kernel parameter sets instead of the v1 kernels, 150 more cells. Its
streams run at 20M events/s of event time.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Final

from benchmarks.workloads import DEFAULT_EVENT_RATE_HZ, DISTRIBUTIONS, Workload

V1_KERNELS: Final = ("event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay")
PROTOTYPE_KERNELS: Final = ("event_count", "polarity", "time_surface", "exp_decay")

GATE_RESOLUTIONS: Final = ((346, 260), (640, 480), (1280, 720))
GATE_BATCH_INTERVALS: Final = (
    (10_000, 16.0),
    (100_000, 0.0),
    (100_000, 16.0),
    (1_000_000, 0.0),
    (1_000_000, 16.0),
)
GATE_DISTRIBUTIONS: Final = DISTRIBUTIONS
GATE_THRESHOLD_EVENTS_PER_S: Final = 20_000_000
REFERENCE_MACHINE: Final = {"chip": "Apple M4", "memory_bytes": 16 * 2**30}

TEMPORAL_KERNEL_CONFIGS: Final = {
    "stacked_histogram_5x10000": ("stacked_histogram", 5, 10_000),
    "stacked_histogram_15x3333": ("stacked_histogram", 15, 3_333),
    "stacked_histogram_10x5000": ("stacked_histogram", 10, 5_000),
    "voxel_grid_5x12500": ("voxel_grid", 5, 12_500),
    "voxel_grid_15x3571": ("voxel_grid", 15, 3_571),
}
"""The temporal gate's kernel parameter sets: name -> (kernel, ``bins``, ``bin_us``)."""
TEMPORAL_EVENT_RATE_HZ: Final = 20_000_000
"""Event-time rate of the temporal gate's streams: 20 events per µs."""

_SEED_OFFSET: Final = {"uniform": 0, "clustered": 1}


def default_seed(sensor_size: tuple[int, int], batch_size: int, distribution: str) -> int:
    """Seed for a cell's event stream.

    The rule is fixed: changing it changes every workload, and results from before
    and after the change stop being comparable. The seed doesn't depend on the
    kernel or the interval: every kernel, and both intervals, see identical input
    for the same resolution, batch size and distribution.
    """
    return 7 * sensor_size[0] + batch_size + _SEED_OFFSET[distribution]


@dataclasses.dataclass(frozen=True, slots=True)
class Cell:
    """One benchmark condition."""

    kernel: str
    sensor_size: tuple[int, int]
    batch_size: int
    interval_ms: float
    distribution: str
    seed: int

    @property
    def condition(self) -> tuple[str, tuple[int, int], int, float, str]:
        """The condition without the seed: what the gate is defined over."""
        return (self.kernel, self.sensor_size, self.batch_size, self.interval_ms, self.distribution)

    @property
    def in_gate(self) -> bool:
        return self.condition in _GATE_CONDITIONS

    @property
    def label(self) -> str:
        w, h = self.sensor_size
        return (
            f"{self.kernel} {w}x{h} {_count(self.batch_size)} @ {self.interval_ms:g} ms "
            f"{self.distribution}"
        )

    def workload(self) -> Workload:
        temporal = self.kernel in TEMPORAL_KERNEL_CONFIGS
        return Workload(
            distribution=self.distribution,
            sensor_size=self.sensor_size,
            batch_size=self.batch_size,
            seed=self.seed,
            event_rate_hz=TEMPORAL_EVENT_RATE_HZ if temporal else DEFAULT_EVENT_RATE_HZ,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "sensor_size": list(self.sensor_size),
            "batch_size": self.batch_size,
            "interval_ms": self.interval_ms,
            "distribution": self.distribution,
            "seed": self.seed,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Cell:
        width, height = record["sensor_size"]
        return cls(
            kernel=record["kernel"],
            sensor_size=(int(width), int(height)),
            batch_size=int(record["batch_size"]),
            interval_ms=float(record["interval_ms"]),
            distribution=record["distribution"],
            seed=int(record["seed"]),
        )


def _count(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def _cells(
    kernels: tuple[str, ...],
    resolutions: tuple[tuple[int, int], ...],
    batch_intervals: tuple[tuple[int, float], ...],
    distributions: tuple[str, ...],
) -> tuple[Cell, ...]:
    return tuple(
        Cell(
            kernel=kernel,
            sensor_size=resolution,
            batch_size=batch_size,
            interval_ms=interval_ms,
            distribution=distribution,
            seed=default_seed(resolution, batch_size, distribution),
        )
        for kernel, resolution, (batch_size, interval_ms), distribution in itertools.product(
            kernels, resolutions, batch_intervals, distributions
        )
    )


def gate_cells() -> tuple[Cell, ...]:
    """The 150 hard-gate cells."""
    return _cells(V1_KERNELS, GATE_RESOLUTIONS, GATE_BATCH_INTERVALS, GATE_DISTRIBUTIONS)


def prototype_baseline_cells() -> tuple[Cell, ...]:
    """The cells of the recorded prototype baseline, for comparison with it.

    The prototype's four kernels, uniform events, 10k/100k/1M events at 0 and 16 ms.
    This is characterization: it includes 10k @ 0 ms, which is outside the gate.
    """
    return _cells(
        PROTOTYPE_KERNELS,
        GATE_RESOLUTIONS,
        tuple((n, iv) for n in (10_000, 100_000, 1_000_000) for iv in (0.0, 16.0)),
        ("uniform",),
    )


def temporal_gate_cells() -> tuple[Cell, ...]:
    """The 150 cells of the temporal-kernel gate, in gate order."""
    return _cells(tuple(TEMPORAL_KERNEL_CONFIGS), GATE_RESOLUTIONS, GATE_BATCH_INTERVALS, GATE_DISTRIBUTIONS)


_GATE_CONDITIONS: Final = frozenset(cell.condition for cell in (*gate_cells(), *temporal_gate_cells()))

SUITES: Final = {
    "gate": gate_cells,
    "prototype-baseline": prototype_baseline_cells,
    "temporal-gate": temporal_gate_cells,
}
