"""Targets for the temporal-kernel gate (``benchmarks/temporal_gate_preregistration.md``).

The same method as the v1 targets (``benchmarks.targets.v1``), for ``StackedHistogram`` and
``VoxelGrid`` in the five parameter sets of the preregistration's section 4. A cell's
``kernel`` is the name of one of those sets (``TEMPORAL_KERNEL_CONFIGS``).

- ``temporal-kernel``: ``kernel.begin_call(state)`` then ``kernel.accumulate(events, state,
  watermark)``, with the running maximum timestamp computed before timing.
- ``temporal-engine``: ``Engine.ingest(events)`` at the cell's interval, on the v1 targets'
  virtual arrival clock.
- ``temporal-planes``: the kernel-level calls again, with the kernels' plane-clearing step
  (``PLANE_CLEARING``) wrapped in a ``perf_counter_ns`` hook. It is the preregistration's
  separate instrumented pass (section 9), never a gate run: the hook adds work to the calls.

Result checks (section 10): the kernel state, read at the final watermark, and the Engine's
last published frame equal bit for bit a reference built with ``np.bincount`` over the
events fed. Each run also records the power source and Low Power Mode at the start and end
of every cell, from its own process, for the environment rules of section 3.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

import frames2py
import frames2py._engine as _engine_module
import frames2py.kernels._temporal as _temporal_module
from frames2py import Engine, StackedHistogram, VoxelGrid
from frames2py.kernels import Kernel

from benchmarks import environment
from benchmarks.matrix import TEMPORAL_KERNEL_CONFIGS, Cell
from benchmarks.targets import Level, Prepared
from benchmarks.targets.v1 import (
    NS_PER_EVENT,
    TARGET_RATE_EVENTS_PER_S,
    VirtualTime,
    _check_clock_seam,
    _input_problems,
    _running_watermarks,
    engine_timed_calls,
    publication_schedule,
)
from benchmarks.workloads import EventArray

PLANE_CLEARING: Final = "frames2py.kernels._temporal._clear_planes"
"""The private function that holds both kernels' plane-clearing step."""


def make_kernel(name: str) -> Kernel:
    """The temporal kernel of a parameter set named in ``TEMPORAL_KERNEL_CONFIGS``."""
    kind, bins, bin_us = TEMPORAL_KERNEL_CONFIGS[name]
    if kind == "stacked_histogram":
        return StackedHistogram(bins=bins, bin_us=bin_us)
    return VoxelGrid(bins=bins, bin_us=bin_us)


def _parameters(name: str) -> dict[str, Any]:
    kind, bins, bin_us = TEMPORAL_KERNEL_CONFIGS[name]
    return {"kernel_class": kind, "bins": bins, "bin_us": bin_us}


def state_bytes(state: Any) -> int:
    """Bytes held by the arrays of a kernel state."""
    if isinstance(state, np.ndarray):
        return int(state.nbytes)
    if dataclasses.is_dataclass(state):
        return sum(state_bytes(getattr(state, field.name)) for field in dataclasses.fields(state))
    return 0


def _describe(name: str, level: Level, statistic: str) -> dict[str, Any]:
    return {
        "name": name,
        "level": level,
        "statistic": statistic,
        "implementation": "frames2py temporal kernels",
        "frames2py_version": frames2py.__version__,
        "kernel_configs": {k: _parameters(k) for k in TEMPORAL_KERNEL_CONFIGS},
    }


def _power() -> dict[str, Any]:
    record = environment._power()
    return {"source": record.get("source"), "low_power_mode": record.get("low_power_mode")}


# ---------------------------------------------------------------- reference


def reference(name: str, sensor_size: tuple[int, int], calls: Sequence[EventArray]) -> NDArray[Any]:
    """What the kernel of parameter set *name* reads at the final watermark after *calls*.

    Built with ``np.bincount``: histogram counts as exact integers, voxel numerators as float64
    sums of integers, exact below 2**53, then converted by the kernel's output rule.
    """
    kind, bins, bin_us = TEMPORAL_KERNEL_CONFIGS[name]
    width, height = sensor_size
    pixels = width * height
    events = np.concatenate([np.asarray(c) for c in calls]) if calls else np.empty(0, dtype=frames2py.EVENT_DTYPE)
    shape = (2, bins, height, width) if kind == "stacked_histogram" else (bins, height, width)
    if not len(events):
        return np.zeros(shape, dtype=np.uint32 if kind == "stacked_histogram" else np.float32)
    t = events["t"].astype(np.int64)
    pixel = events["y"].astype(np.int64) * width + events["x"].astype(np.int64)
    on = events["p"] != 0
    current = int(t.max()) // bin_us  # the bin in progress at the final watermark
    q = t // bin_us
    if kind == "stacked_histogram":
        j = q - (current - bins)
        keep = (j >= 0) & (j < bins)
        index = (on[keep].astype(np.int64) * bins + j[keep]) * pixels + pixel[keep]
        counts = np.bincount(index, minlength=2 * bins * pixels)
        return (counts % 2**32).astype(np.uint32).reshape(shape)
    first = current - bins + 1  # the first knot's bin
    keep = (q >= first) & (q < current)
    j, r = q[keep] - first, (t - q * bin_us)[keep]
    sign = np.where(on[keep], 1.0, -1.0)
    numerators = np.bincount(j * pixels + pixel[keep], weights=sign * (bin_us - r), minlength=bins * pixels)
    numerators += np.bincount((j + 1) * pixels + pixel[keep], weights=sign * r, minlength=bins * pixels)
    return (numerators / float(bin_us) + 0.0).astype(np.float32).reshape(shape)


def compare(observed: NDArray[Any], expected: NDArray[Any]) -> str | None:
    """``None`` if *observed* equals *expected* bit for bit, otherwise what differs."""
    if observed.shape != expected.shape or observed.dtype != expected.dtype:
        return f"{observed.dtype} {observed.shape}, expected {expected.dtype} {expected.shape}"
    differ = observed.view(np.uint8).reshape(observed.size, -1) != expected.view(np.uint8).reshape(expected.size, -1)
    bad = differ.any(axis=1)
    if bad.any():
        worst = int(np.argmax(bad))
        return (f"{int(bad.sum())} values differ; first at flat index {worst}: "
                f"{observed.flat[worst]!r} vs {expected.flat[worst]!r}")
    return None


# ---------------------------------------------------------------- kernel level


class TemporalKernelTarget:
    """``kernel.begin_call(state)`` then ``kernel.accumulate(events, state, watermark)``."""

    name = "temporal-kernel"
    level: Level = "kernel"
    statistic = "median_call"
    _hooked = False

    def supports(self, cell: Cell) -> bool:
        return cell.kernel in TEMPORAL_KERNEL_CONFIGS

    def timed_calls(self, cell: Cell, default: int, warmup_calls: int) -> int:
        return default

    def prepare(self, cell: Cell, batches: Sequence[EventArray]) -> Prepared:
        power_start = _power()
        kernel = make_kernel(cell.kernel)
        state = kernel.init_state(cell.sensor_size)
        shape, dtype = kernel.output_spec(cell.sensor_size)
        problems = _input_problems(batches, cell.sensor_size)
        marks = _running_watermarks(batches)
        begin_call, accumulate = kernel.begin_call, kernel.accumulate
        watermark: list[int | None] = [None]
        calls = [0]
        hook = _PlaneHook() if self._hooked else None

        def before_call() -> None:
            watermark[0] = marks[calls[0]]
            if hook is not None:
                hook.install()

        def call(events: EventArray) -> None:
            begin_call(state)
            accumulate(events, state, watermark[0])

        def after_call() -> None:
            if hook is not None:
                hook.uninstall()
            calls[0] += 1

        def finish() -> Mapping[str, Any]:
            failures = list(problems)
            out = np.empty(shape, dtype=dtype)
            kernel.read(state, out, watermark[0])
            mismatch = compare(out, reference(cell.kernel, cell.sensor_size, batches[: calls[0]]))
            if mismatch:
                failures.append(f"output: {mismatch}")
            checks: dict[str, Any] = {"valid": not failures, "failures": failures, "calls": calls[0],
                                      "power": {"start": power_start, "end": _power()}}
            if hook is not None:
                checks["plane_clearing_function"] = PLANE_CLEARING
                checks["plane_clearing_ns"] = hook.per_call
            return checks

        return Prepared(
            call=call,
            details={"output_dtype": str(dtype), "state_bytes": state_bytes(state), **_parameters(cell.kernel)},
            before_call=before_call,
            after_call=after_call,
            finish=finish,
        )

    def describe(self) -> dict[str, Any]:
        return _describe(self.name, self.level, self.statistic)


class _PlaneHook:
    """Times every call of the plane-clearing step while installed; one total per kernel call."""

    def __init__(self) -> None:
        self.original = _temporal_module._clear_planes
        self.per_call: list[int] = []
        self._current = 0

    def install(self) -> None:
        original = self.original
        self._current = 0

        def timed(state: Any, newest: int) -> None:
            start = time.perf_counter_ns()
            original(state, newest)
            self._current += time.perf_counter_ns() - start

        _temporal_module._clear_planes = timed

    def uninstall(self) -> None:
        _temporal_module._clear_planes = self.original
        self.per_call.append(self._current)


class TemporalPlanesTarget(TemporalKernelTarget):
    """The kernel-level calls with the plane-clearing step timed: a separate instrumented pass."""

    name = "temporal-planes"
    _hooked = True


def plane_share(record: Mapping[str, Any], warmup_calls: int) -> list[float]:
    """Per run of a ``temporal-planes`` cell record: plane-clearing time over the timed calls' time."""
    shares = []
    for calls, checks in zip(record["call_ns"], record["checks"]):
        hooked = checks["plane_clearing_ns"][warmup_calls : warmup_calls + len(calls)]
        shares.append(sum(hooked) / sum(calls))
    return shares


# ---------------------------------------------------------------- engine level


class TemporalEngineTarget:
    """``Engine.ingest(events)`` at the cell's interval, on the virtual clock."""

    name = "temporal-engine"
    level: Level = "engine"
    statistic = "sustained"

    def supports(self, cell: Cell) -> bool:
        return cell.kernel in TEMPORAL_KERNEL_CONFIGS

    def timed_calls(self, cell: Cell, default: int, warmup_calls: int) -> int:
        return engine_timed_calls(cell, default, warmup_calls)

    def prepare(self, cell: Cell, batches: Sequence[EventArray]) -> Prepared:
        power_start = _power()
        _check_clock_seam()
        clock = VirtualTime()
        module: Any = _engine_module
        module.time = clock
        try:
            kernel = make_kernel(cell.kernel)
            engine = Engine(cell.sensor_size, kernel, snapshot_interval_ms=cell.interval_ms)
        finally:
            module.time = time
        step = cell.batch_size * NS_PER_EVENT
        problems = _input_problems(batches, cell.sensor_size)
        calls = [0]
        published = [0]
        flags: list[bool] = []
        publications: list[dict[str, Any]] = []

        def before_call() -> None:
            clock.now_ns = calls[0] * step
            module.time = clock

        def after_call() -> None:
            module.time = time
            count = engine.stats.snapshots_published
            flags.append(count > published[0])
            if count > published[0]:
                meta = engine.snapshot().meta  # type: ignore[union-attr]
                publications.append({
                    "call": calls[0],
                    "virtual_ns": clock.now_ns,
                    "sequence": meta.sequence,
                    "watermark": meta.watermark,
                })
            published[0] = count
            calls[0] += 1

        def counters() -> Mapping[str, int]:
            return {"snapshots_published": engine.stats.snapshots_published}

        def finish() -> Mapping[str, Any]:
            failures = list(problems)
            n = calls[0]
            fed = batches[:n]
            expected = publication_schedule(n, cell.batch_size, cell.interval_ms)
            if flags != expected:
                failures.append(
                    f"publications at calls {[i for i, f in enumerate(flags) if f]}, "
                    f"expected {[i for i, f in enumerate(expected) if f]}"
                )
            stats = engine.stats
            if stats.events_ingested != sum(len(b) for b in fed):
                failures.append(f"events_ingested {stats.events_ingested}")
            if stats.events_out_of_bounds != 0:
                failures.append(f"events_out_of_bounds {stats.events_out_of_bounds}")
            if stats.snapshots_published != sum(flags):
                failures.append(f"snapshots_published {stats.snapshots_published}")
            marks = _running_watermarks(fed)
            for number, publication in enumerate(publications, 1):
                if publication["sequence"] != number:
                    failures.append(f"publication {number} has sequence {publication['sequence']}")
                if publication["watermark"] != marks[publication["call"]]:
                    failures.append(f"publication {number} has watermark {publication['watermark']}")
            snapshot = engine.snapshot()
            if snapshot is None or not publications:
                failures.append("nothing published")
            else:
                last = publications[-1]["call"]
                mismatch = compare(snapshot.frame, reference(cell.kernel, cell.sensor_size, fed[: last + 1]))
                if mismatch:
                    failures.append(f"last snapshot: {mismatch}")
            del snapshot
            return {
                "valid": not failures,
                "failures": failures,
                "calls": n,
                "virtual_step_ns": step,
                "virtual_elapsed_ns": (n - 1) * step if n else 0,
                "publication_calls": [i for i, f in enumerate(flags) if f],
                "publications": publications,
                "power": {"start": power_start, "end": _power()},
            }

        return Prepared(
            call=engine.ingest,
            counters=counters,
            details={
                "clock": "virtual",
                "virtual_step_ns": step,
                "target_rate_events_per_s": TARGET_RATE_EVENTS_PER_S,
                "state_bytes": state_bytes(engine._accumulator._state),
                **_parameters(cell.kernel),
            },
            before_call=before_call,
            after_call=after_call,
            finish=finish,
        )

    def describe(self) -> dict[str, Any]:
        return {
            **_describe(self.name, self.level, self.statistic),
            "clock": "virtual",
            "target_rate_events_per_s": TARGET_RATE_EVENTS_PER_S,
        }
