"""Deterministic synthetic event generator for benchmarking.

Provides :func:`generate_batch` for single batches and
:func:`event_stream` for paced real-time-like streams.  All generators
are **seeded** so results are reproducible across machines.

Four standard profiles model real-world event camera workloads at
increasing intensity.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator

import numpy as np

from frames2py.core.types import EVENT_DTYPE, EventBatch


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Benchmark workload profile.

    Attributes:
        name: Human-readable name.
        rate: Target event rate in events/second.
        sensor_size: ``(width, height)``.
        batch_size: Events per batch.
        description: One-line description.
    """

    name: str
    rate: int
    sensor_size: tuple[int, int]
    batch_size: int
    description: str


PROFILES: dict[str, Profile] = {
    "low": Profile(
        name="low",
        rate=500_000,
        sensor_size=(346, 260),
        batch_size=5_000,
        description="DAVIS346, indoor static scene",
    ),
    "medium": Profile(
        name="medium",
        rate=2_000_000,
        sensor_size=(640, 480),
        batch_size=20_000,
        description="DVXplorer, moderate motion",
    ),
    "high": Profile(
        name="high",
        rate=5_000_000,
        sensor_size=(1280, 720),
        batch_size=50_000,
        description="IMX636, fast drone flight",
    ),
    "stress": Profile(
        name="stress",
        rate=10_000_000,
        sensor_size=(1280, 720),
        batch_size=100_000,
        description="Synthetic worst case, exceeds real sensors",
    ),
}


def generate_batch(
    n_events: int,
    sensor_size: tuple[int, int] = (1280, 720),
    t_start: int = 0,
    t_step: int = 1,
    seed: int = 42,
) -> EventBatch:
    """Generate a single batch of deterministic synthetic events.

    Events are uniformly distributed over the sensor area with
    monotonically increasing timestamps.

    Parameters:
        n_events: Number of events to generate.
        sensor_size: ``(width, height)`` of the virtual sensor.
        t_start: Timestamp of the first event (microseconds).
        t_step: Timestamp increment between consecutive events.
        seed: RNG seed for reproducibility.

    Returns:
        1-D structured array with dtype :data:`EVENT_DTYPE`.
    """
    rng = np.random.default_rng(seed)
    w, h = sensor_size

    events = np.empty(n_events, dtype=EVENT_DTYPE)
    events["t"] = np.arange(t_start, t_start + n_events * t_step, t_step, dtype=np.uint64)
    events["x"] = rng.integers(0, w, size=n_events, dtype=np.uint16)
    events["y"] = rng.integers(0, h, size=n_events, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n_events, dtype=np.uint8)

    return events


def event_stream(
    rate_events_per_sec: float,
    sensor_size: tuple[int, int] = (1280, 720),
    batch_size: int = 10_000,
    duration_sec: float = 10.0,
    seed: int = 42,
    paced: bool = True,
) -> Iterator[EventBatch]:
    """Yield event batches simulating a real-time camera stream.

    Parameters:
        rate_events_per_sec: Target event production rate.
        sensor_size: ``(width, height)`` of the virtual sensor.
        batch_size: Events per batch.
        duration_sec: Total stream duration in seconds.
        seed: RNG seed for reproducibility.
        paced: If ``True``, insert ``time.sleep`` between batches to
            approximate wall-clock rate.  If ``False``, yield as fast
            as possible (for throughput benchmarks).

    Yields:
        Event batches at the specified rate.
    """
    total_events = int(rate_events_per_sec * duration_sec)
    batch_interval_sec = batch_size / rate_events_per_sec if paced else 0.0

    t_cursor = 0
    # Timestamp step: microseconds between events at the target rate
    t_step = max(1, int(1_000_000 / rate_events_per_sec))
    events_yielded = 0
    batch_seed = seed

    while events_yielded < total_events:
        n = min(batch_size, total_events - events_yielded)
        batch = generate_batch(
            n_events=n,
            sensor_size=sensor_size,
            t_start=t_cursor,
            t_step=t_step,
            seed=batch_seed,
        )
        t_cursor += n * t_step
        events_yielded += n
        batch_seed += 1

        yield batch

        if paced and batch_interval_sec > 0:
            time.sleep(batch_interval_sec)


def profile_stream(
    profile_name: str,
    duration_sec: float = 10.0,
    seed: int = 42,
    paced: bool = True,
) -> Iterator[EventBatch]:
    """Yield a stream from a named :data:`PROFILES` entry.

    Parameters:
        profile_name: One of ``"low"``, ``"medium"``, ``"high"``,
            ``"stress"``.
        duration_sec: Total stream duration.
        seed: RNG seed.
        paced: Whether to pace batches at wall-clock rate.

    Yields:
        Event batches per the selected profile.
    """
    if profile_name not in PROFILES:
        raise KeyError(
            f"Unknown profile {profile_name!r}. "
            f"Available: {sorted(PROFILES)}"
        )
    p = PROFILES[profile_name]
    return event_stream(
        rate_events_per_sec=p.rate,
        sensor_size=p.sensor_size,
        batch_size=p.batch_size,
        duration_sec=duration_sec,
        seed=seed,
        paced=paced,
    )
