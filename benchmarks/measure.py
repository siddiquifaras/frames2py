"""Timing, latency percentiles and memory measurement for one prepared target.

Method:

- distinct pre-generated batches, one per call
- warmup calls first, discarded
- timed calls with the garbage collector disabled, one ``perf_counter_ns``
  interval per call; a target's per-call hooks run outside those intervals
- a per-run throughput, by one of two statistics:
  - ``median_call``: ``batch_size / median call time`` (kernel level)
  - ``sustained``: events in the timed calls / the sum of their call times
    (engine level), so occasional expensive calls, such as publications, count
- across runs: the median of the per-run throughputs

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
STATISTICS: Final = ("median_call", "sustained")


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
Hook = Callable[[], object]


def _nothing() -> None:
    pass


def time_calls(
    call: Call,
    batches: Sequence[EventArray],
    warmup_calls: int,
    timed_calls: int,
    counters: Counters | None = None,
    before_call: Hook | None = None,
    after_call: Hook | None = None,
    span: list[int] | None = None,
) -> tuple[list[int], dict[str, int]]:
    """Run warmup calls, then time *timed_calls* calls on distinct batches.

    *before_call* and *after_call* run around every call, warmup included, outside
    the timed interval. If *span* is given, the wall-clock time from the start of
    the first timed call to the end of the last, hooks included, is appended to it.

    Returns the per-call times in nanoseconds and, if *counters* is given, how
    much each counter changed across the timed calls.
    """
    if len(batches) < warmup_calls + timed_calls:
        raise ValueError(
            f"need {warmup_calls + timed_calls} batches, got {len(batches)}"
        )
    before = before_call or _nothing
    after = after_call or _nothing
    for batch in batches[:warmup_calls]:
        before()
        call(batch)
        after()
    start_counts = dict(counters()) if counters is not None else {}
    timed = batches[warmup_calls : warmup_calls + timed_calls]
    samples: list[int] = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        first = time.perf_counter_ns()
        for batch in timed:
            before()
            start = time.perf_counter_ns()
            call(batch)
            samples.append(time.perf_counter_ns() - start)
            after()
        last = time.perf_counter_ns()
    finally:
        if gc_was_enabled:
            gc.enable()
    if span is not None:
        span.append(last - first)
    end_counts = dict(counters()) if counters is not None else {}
    return samples, {name: end_counts[name] - start_counts.get(name, 0) for name in end_counts}


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


def run_throughput(batch_size: int, run: Sequence[int], statistic: str = "median_call") -> float:
    """One run's throughput in events/s by *statistic* (see the module docstring)."""
    if not run:
        raise ValueError("a run needs at least one timed call")
    if statistic == "median_call":
        return batch_size / (statistics.median(run) / 1e9)
    if statistic == "sustained":
        return batch_size * len(run) / (sum(run) / 1e9)
    raise ValueError(f"unknown statistic {statistic!r}; expected one of {STATISTICS}")


def summarize(
    batch_size: int, runs: Sequence[Sequence[int]], statistic: str = "median_call"
) -> dict[str, Any]:
    """Throughput and latency figures for one cell from its per-run call times.

    ``events_per_s`` is the median of the per-run throughputs by *statistic*.
    """
    if not runs or any(not run for run in runs):
        raise ValueError("every run needs at least one timed call")
    run_medians = [statistics.median(run) for run in runs]
    per_run = [run_throughput(batch_size, run, statistic) for run in runs]
    pooled = [sample for run in runs for sample in run]
    return {
        "statistic": statistic,
        "run_events_per_s": per_run,
        "run_median_ns": run_medians,
        "median_ns": statistics.median(run_medians),
        "events_per_s": statistics.median(per_run),
        "events_per_s_run_range": [min(per_run), max(per_run)],
        "latency_ns": {
            **{f"p{q}": nearest_rank(pooled, q) for q in LATENCY_PERCENTILES},
            "max": max(pooled),
            "samples": len(pooled),
        },
    }


def measure_memory(
    call: Call,
    batches: Sequence[EventArray],
    warmup_calls: int,
    memory_calls: int,
    before_call: Hook | None = None,
    after_call: Hook | None = None,
) -> dict[str, Any]:
    """Temporary allocation per call and retained growth, under ``tracemalloc``.

    ``peak_temporary_bytes`` is, per call, the traced peak during the call minus
    what was traced when it started. ``retained_growth_bytes`` is how much more is
    traced after the last measured call than before the first. Allocations made
    before tracing started (target state, warmup) are not counted.
    """
    if len(batches) < warmup_calls + memory_calls:
        raise ValueError(f"need {warmup_calls + memory_calls} batches, got {len(batches)}")
    before = before_call or _nothing
    after = after_call or _nothing
    for batch in batches[:warmup_calls]:
        before()
        call(batch)
        after()
    if tracemalloc.is_tracing():
        raise RuntimeError("tracemalloc is already tracing; memory figures would be wrong")
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        peaks: list[int] = []
        for batch in batches[warmup_calls : warmup_calls + memory_calls]:
            before()
            tracemalloc.reset_peak()
            start = tracemalloc.get_traced_memory()[0]
            call(batch)
            peaks.append(tracemalloc.get_traced_memory()[1] - start)
            after()
        retained = tracemalloc.get_traced_memory()[0] - baseline
    finally:
        tracemalloc.stop()
    return {
        "method": "tracemalloc",
        "calls": memory_calls,
        "peak_temporary_bytes": peaks,
        "retained_growth_bytes": retained,
    }
