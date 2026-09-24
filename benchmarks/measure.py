"""Timing, latency percentiles and memory measurement for one prepared target.

Method:

- distinct pre-generated batches, one per call
- warmup calls first, discarded
- timed calls with the garbage collector disabled, one ``perf_counter_ns``
  interval per call
- per run: the median call time; across runs: the median of those medians
- throughput is ``batch_size / median call time``

Latency percentiles pool every timed call of every run and use the nearest-rank
definition, so each reported percentile is an observed call time. With few calls
the upper percentiles are close to the maximum; the sample count is recorded
next to them.

Memory is measured in a separate pass under ``tracemalloc``, which sees NumPy's
data allocations. Timing never runs with tracing on.
"""

from __future__ import annotations

import dataclasses
import gc
import math
import statistics
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from benchmarks.workloads import EventArray

DEFAULT_TIMED_CALLS: Final = {10_000: 50, 100_000: 20, 1_000_000: 7}
LATENCY_PERCENTILES: Final = (50, 95, 99)


@dataclasses.dataclass(frozen=True, slots=True)
class Policy:
    """How a cell is measured.

    Attributes:
        runs: Independent runs per cell, each on a freshly prepared target.
        warmup_calls: Calls per run before timing starts, discarded.
        timed_calls: Timed calls per run. ``None`` uses ``DEFAULT_TIMED_CALLS``
            for the cell's batch size.
        memory_calls: Calls measured in the memory pass. ``0`` skips it.
    """

    runs: int = 5
    warmup_calls: int = 1
    timed_calls: int | None = None
    memory_calls: int = 3

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError(f"runs must be >= 1, got {self.runs}")
        if self.warmup_calls < 0:
            raise ValueError(f"warmup_calls must be >= 0, got {self.warmup_calls}")
        if self.timed_calls is not None and self.timed_calls < 1:
            raise ValueError(f"timed_calls must be >= 1, got {self.timed_calls}")
        if self.memory_calls < 0:
            raise ValueError(f"memory_calls must be >= 0, got {self.memory_calls}")

    def timed_calls_for(self, batch_size: int) -> int:
        if self.timed_calls is not None:
            return self.timed_calls
        try:
            return DEFAULT_TIMED_CALLS[batch_size]
        except KeyError:
            raise ValueError(
                f"no default timed-call count for batch size {batch_size}; "
                "set Policy.timed_calls"
            ) from None

    def batches_needed(self, batch_size: int) -> int:
        return self.warmup_calls + max(self.timed_calls_for(batch_size), self.memory_calls)

    def to_record(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "warmup_calls": self.warmup_calls,
            "timed_calls": self.timed_calls,
            "default_timed_calls": {str(k): v for k, v in DEFAULT_TIMED_CALLS.items()},
            "memory_calls": self.memory_calls,
            "gc_disabled_while_timing": True,
            "clock": "time.perf_counter_ns",
            "latency_percentile_method": "nearest-rank",
        }


Call = Callable[[EventArray], object]
Counters = Callable[[], Mapping[str, int]]


def time_calls(
    call: Call,
    batches: Sequence[EventArray],
    warmup_calls: int,
    timed_calls: int,
    counters: Counters | None = None,
) -> tuple[list[int], dict[str, int]]:
    """Run warmup calls, then time *timed_calls* calls on distinct batches.

    Returns the per-call times in nanoseconds and, if *counters* is given, how
    much each counter changed across the timed calls.
    """
    if len(batches) < warmup_calls + timed_calls:
        raise ValueError(
            f"need {warmup_calls + timed_calls} batches, got {len(batches)}"
        )
    for batch in batches[:warmup_calls]:
        call(batch)
    before = dict(counters()) if counters is not None else {}
    timed = batches[warmup_calls : warmup_calls + timed_calls]
    samples: list[int] = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for batch in timed:
            start = time.perf_counter_ns()
            call(batch)
            samples.append(time.perf_counter_ns() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    after = dict(counters()) if counters is not None else {}
    return samples, {name: after[name] - before.get(name, 0) for name in after}


def nearest_rank(samples: Sequence[int], percentile: float) -> int:
    """The nearest-rank percentile: the smallest sample with at least
    ``percentile`` percent of the samples at or below it."""
    if not samples:
        raise ValueError("no samples")
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    ordered = sorted(samples)
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def summarize(batch_size: int, runs: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Throughput and latency figures for one cell from its per-run call times."""
    if not runs or any(not run for run in runs):
        raise ValueError("every run needs at least one timed call")
    run_medians = [statistics.median(run) for run in runs]
    median_ns = statistics.median(run_medians)
    pooled = [sample for run in runs for sample in run]
    return {
        "run_median_ns": run_medians,
        "median_ns": median_ns,
        "events_per_s": batch_size / (median_ns / 1e9),
        "events_per_s_run_range": [
            batch_size / (max(run_medians) / 1e9),
            batch_size / (min(run_medians) / 1e9),
        ],
        "latency_ns": {
            **{f"p{q}": nearest_rank(pooled, q) for q in LATENCY_PERCENTILES},
            "max": max(pooled),
            "samples": len(pooled),
        },
    }


def measure_memory(
    call: Call, batches: Sequence[EventArray], warmup_calls: int, memory_calls: int
) -> dict[str, Any]:
    """Temporary allocation per call and retained growth, under ``tracemalloc``.

    ``peak_temporary_bytes`` is, per call, the traced peak during the call minus
    what was traced when it started. ``retained_growth_bytes`` is how much more is
    traced after the last measured call than before the first. Allocations made
    before tracing started (target state, warmup) are not counted.
    """
    if len(batches) < warmup_calls + memory_calls:
        raise ValueError(f"need {warmup_calls + memory_calls} batches, got {len(batches)}")
    for batch in batches[:warmup_calls]:
        call(batch)
    if tracemalloc.is_tracing():
        raise RuntimeError("tracemalloc is already tracing; memory figures would be wrong")
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        peaks: list[int] = []
        for batch in batches[warmup_calls : warmup_calls + memory_calls]:
            tracemalloc.reset_peak()
            start = tracemalloc.get_traced_memory()[0]
            call(batch)
            peaks.append(tracemalloc.get_traced_memory()[1] - start)
        retained = tracemalloc.get_traced_memory()[0] - baseline
    finally:
        tracemalloc.stop()
    return {
        "method": "tracemalloc",
        "calls": memory_calls,
        "peak_temporary_bytes": peaks,
        "retained_growth_bytes": retained,
    }
