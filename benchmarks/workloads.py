"""Deterministic synthetic event workloads.

Every workload produces in-bounds events only, with ``p`` uniform over {0, 1} and
timestamps that increase across the batches of one stream at a fixed event-time
rate. The same parameters always produce the same batches.

Two distributions:

- ``uniform``: ``x`` and ``y`` uniform over the sensor. Scattered writes give the
  worst cache locality, which is why it is the primary pessimistic workload.
- ``clustered``: events drawn around a fixed set of cluster centres, with a
  Gaussian spread. The centres are fixed for the whole stream.

The uniform draw order (``x``, then ``y``, then ``p``, per batch, from one
generator seeded with the cell's seed) is part of the stream definition: changing
it changes every uniform workload.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from frames2py import EVENT_DTYPE

EventArray = NDArray[np.void]

DISTRIBUTIONS: Final = ("uniform", "clustered")

DEFAULT_EVENT_RATE_HZ: Final = 1_000_000
"""One event per microsecond of event time."""

DEFAULT_CLUSTERS: Final = 8
DEFAULT_CLUSTER_SIGMA_FRACTION: Final = 0.02
"""Cluster spread as a fraction of ``min(width, height)``."""


@dataclasses.dataclass(frozen=True, slots=True)
class Workload:
    """A deterministic event stream for one sensor size and batch size.

    Attributes:
        distribution: ``"uniform"`` or ``"clustered"``.
        sensor_size: ``(width, height)``.
        batch_size: Events per batch.
        seed: Seed for every random draw in the stream.
        event_rate_hz: Event-time rate. Event ``i`` of the stream has
            ``t = (i * 1_000_000) // event_rate_hz`` microseconds.
        clusters: Number of cluster centres (clustered only).
        cluster_sigma_fraction: Standard deviation of the spread around a centre,
            as a fraction of ``min(width, height)`` (clustered only).
    """

    distribution: str
    sensor_size: tuple[int, int]
    batch_size: int
    seed: int
    event_rate_hz: int = DEFAULT_EVENT_RATE_HZ
    clusters: int = DEFAULT_CLUSTERS
    cluster_sigma_fraction: float = DEFAULT_CLUSTER_SIGMA_FRACTION

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"unknown distribution {self.distribution!r}; expected one of {DISTRIBUTIONS}"
            )
        width, height = self.sensor_size
        if width < 1 or height < 1 or width > 65_536 or height > 65_536:
            raise ValueError(f"sensor_size must be within 1..65536 per axis, got {self.sensor_size}")
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")
        if self.event_rate_hz < 1:
            raise ValueError(f"event_rate_hz must be >= 1, got {self.event_rate_hz}")
        if self.distribution == "clustered":
            if self.clusters < 1:
                raise ValueError(f"clusters must be >= 1, got {self.clusters}")
            if not self.cluster_sigma_fraction > 0:
                raise ValueError(
                    f"cluster_sigma_fraction must be > 0, got {self.cluster_sigma_fraction}"
                )

    def batches(self, count: int) -> list[EventArray]:
        """The first *count* batches of the stream, each a fresh C-contiguous array."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        rng = np.random.default_rng(self.seed)
        centres = self._centres() if self.distribution == "clustered" else None
        out: list[EventArray] = []
        for index in range(count):
            batch = np.empty(self.batch_size, dtype=EVENT_DTYPE)
            batch["t"] = self._timestamps(index * self.batch_size)
            if centres is None:
                self._fill_uniform(rng, batch)
            else:
                self._fill_clustered(rng, batch, centres)
            batch["p"] = rng.integers(0, 2, size=self.batch_size, dtype=np.uint8)
            out.append(batch)
        return out

    def describe(self) -> dict[str, Any]:
        """The parameters that determine the stream, for result records."""
        record: dict[str, Any] = {
            "distribution": self.distribution,
            "seed": self.seed,
            "event_rate_hz": self.event_rate_hz,
        }
        if self.distribution == "clustered":
            record["clusters"] = self.clusters
            record["cluster_sigma_fraction"] = self.cluster_sigma_fraction
        return record

    def _timestamps(self, first_index: int) -> NDArray[np.uint64]:
        index = np.arange(first_index, first_index + self.batch_size, dtype=np.int64)
        return ((index * 1_000_000) // self.event_rate_hz).astype(np.uint64)

    def _fill_uniform(self, rng: np.random.Generator, batch: EventArray) -> None:
        width, height = self.sensor_size
        batch["x"] = rng.integers(0, width, size=self.batch_size, dtype=np.uint16)
        batch["y"] = rng.integers(0, height, size=self.batch_size, dtype=np.uint16)

    def _sigma(self) -> float:
        return self.cluster_sigma_fraction * min(self.sensor_size)

    def _centres(self) -> NDArray[np.float64]:
        # Separate stream so the centres don't depend on the batch size.
        rng = np.random.default_rng([self.seed, 1])
        sigma = self._sigma()
        centres = np.empty((self.clusters, 2), dtype=np.float64)
        for axis, extent in enumerate(self.sensor_size):
            margin = min(3.0 * sigma, (extent - 1) / 2.0)
            centres[:, axis] = rng.uniform(margin, extent - 1 - margin, size=self.clusters)
        return centres

    def _fill_clustered(
        self, rng: np.random.Generator, batch: EventArray, centres: NDArray[np.float64]
    ) -> None:
        width, height = self.sensor_size
        which = rng.integers(0, self.clusters, size=self.batch_size)
        offsets = rng.normal(0.0, self._sigma(), size=(self.batch_size, 2))
        points = np.rint(centres[which] + offsets)
        batch["x"] = np.clip(points[:, 0], 0, width - 1).astype(np.uint16)
        batch["y"] = np.clip(points[:, 1], 0, height - 1).astype(np.uint16)
