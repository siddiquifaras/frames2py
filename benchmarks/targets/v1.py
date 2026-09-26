"""Targets for the v1 core: the kernels alone, and ``Engine.ingest()``.

Kernel level times ``kernel.begin_call(state)`` followed by
``kernel.accumulate(events, state, watermark)`` through the public ``Kernel``
protocol, on one call's events. The watermark each call receives is the running
maximum timestamp, computed before timing starts: that is the Accumulator's work,
not the kernel's. Structural validation, the range and bounds checks, ``read`` and
publication are outside the timed region.

Engine level times ``Engine.ingest(events)`` with the cell's publication interval.
The Engine reads a virtual clock instead of the monotonic clock, advanced between
calls as if events arrived at ``TARGET_RATE_EVENTS_PER_S``: call ``k`` (warmup
included, from 0) sees ``k * batch_size * NS_PER_EVENT`` nanoseconds. The clock is
put in place of the ``time`` module inside ``frames2py._engine`` only for the
Engine's construction and for each call, and the real module is put back straight
after, so no other Engine in the process sees it. The Engine's
own interval check decides every publication. Cells with a positive interval run
enough timed calls for at least ``MIN_TIMED_PUBLICATIONS`` publications during the
timed calls, and every run checks the observed publications against the schedule
the contract gives for that clock.

Both targets check their result after the timed calls against references built
with different NumPy primitives from the kernels' (``bincount``, ``lexsort``, eager
decay). A run whose checks fail is marked invalid.

Kernel parameters are fixed here: ``ExpDecay(0.95)`` and ``TimestampDecay(10_000.0)``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

import frames2py
import frames2py._engine as _engine_module
from frames2py import Engine
from frames2py._events import TIMESTAMP_LIMIT, validate
from frames2py.kernels import EventCount, ExpDecay, Kernel, Polarity, TimeSurface, TimestampDecay

from benchmarks.matrix import V1_KERNELS, Cell
from benchmarks.targets import Level, Prepared
from benchmarks.workloads import EventArray

KERNEL_PARAMETERS: Final[Mapping[str, Mapping[str, float]]] = {
    "exp_decay": {"decay": 0.95},
    "timestamp_decay": {"tau_us": 10_000.0},
}

TARGET_RATE_EVENTS_PER_S: Final = 20_000_000
NS_PER_EVENT: Final = 1_000_000_000 // TARGET_RATE_EVENTS_PER_S
MIN_TIMED_PUBLICATIONS: Final = 10

FLOAT_RTOL: Final = 3e-7
"""About 2 float32 ULP: the kernel's float64 value and the reference's differ only in
summation order, then both round to float32."""
FLOAT_ATOL: Final = float(np.finfo(np.float32).tiny)


def make_kernel(name: str) -> Kernel:
    """A v1 kernel configured with the benchmark's parameters."""
    if name == "event_count":
        return EventCount()
    if name == "polarity":
        return Polarity()
    if name == "time_surface":
        return TimeSurface()
    if name == "exp_decay":
        return ExpDecay(KERNEL_PARAMETERS["exp_decay"]["decay"])
    if name == "timestamp_decay":
        return TimestampDecay(KERNEL_PARAMETERS["timestamp_decay"]["tau_us"])
    raise ValueError(f"not a v1 kernel: {name!r}")


def _describe(name: str, level: Level, statistic: str) -> dict[str, Any]:
    return {
        "name": name,
        "level": level,
        "statistic": statistic,
        "implementation": "frames2py v1",
        "frames2py_version": frames2py.__version__,
        "kernels": list(V1_KERNELS),
        "kernel_parameters": {k: dict(v) for k, v in KERNEL_PARAMETERS.items()},
    }


# ---------------------------------------------------------------- references


def _flat_index(batch: EventArray, width: int) -> NDArray[np.int64]:
    return batch["y"].astype(np.int64) * width + batch["x"].astype(np.int64)


def reference(
    kernel: str, sensor_size: tuple[int, int], calls: Sequence[EventArray]
) -> NDArray[Any]:
    """What *kernel* reads after accumulating *calls*, one ``accumulate`` per batch.

    Counts are exact; decay kernels are float64, compared within ``FLOAT_RTOL``.
    """
    width, height = sensor_size
    pixels = width * height
    if kernel in ("event_count", "polarity"):
        channels = 2 if kernel == "polarity" else 1
        counts = np.zeros(pixels * channels, dtype=np.int64)
        for batch in calls:
            index = _flat_index(batch, width)
            if channels == 2:
                index = index * 2 + (batch["p"] != 0)
            counts += np.bincount(index, minlength=pixels * channels)
        shape = (height, width, 2) if channels == 2 else (height, width)
        return (counts % 2**32).astype(np.uint32).reshape(shape)
    if kernel == "time_surface":
        index = np.concatenate([_flat_index(b, width) for b in calls]) if calls else np.empty(0, np.int64)
        t = np.concatenate([b["t"] for b in calls]) if calls else np.empty(0, np.uint64)
        surface = np.zeros(pixels, dtype=np.uint64)
        if len(t):
            order = np.lexsort((t, index))
            index, t = index[order], t[order]
            last = np.flatnonzero(np.append(index[1:] != index[:-1], True))
            surface[index[last]] = t[last]
        return surface.reshape(height, width)
    if kernel == "exp_decay":
        decay = KERNEL_PARAMETERS["exp_decay"]["decay"]
        decayed = np.zeros(pixels, dtype=np.float64)
        for batch in calls:
            decayed *= decay
            decayed += np.bincount(_flat_index(batch, width), minlength=pixels)
        return decayed.reshape(height, width)
    if kernel == "timestamp_decay":
        tau = KERNEL_PARAMETERS["timestamp_decay"]["tau_us"]
        summed = np.zeros(pixels, dtype=np.float64)
        if calls:
            watermark = max(int(b["t"].max()) for b in calls if len(b))
            for batch in calls:
                since = batch["t"].astype(np.int64) - np.int64(watermark)
                summed += np.bincount(
                    _flat_index(batch, width), weights=np.exp(since / tau), minlength=pixels
                )
        return summed.reshape(height, width)
    raise ValueError(f"not a v1 kernel: {kernel!r}")


def compare(kernel: str, observed: NDArray[Any], expected: NDArray[Any]) -> str | None:
    """``None`` if *observed* matches *expected*, otherwise what differs."""
    if observed.shape != expected.shape:
        return f"shape {observed.shape}, expected {expected.shape}"
    if kernel in ("exp_decay", "timestamp_decay"):
        diff = np.abs(observed.astype(np.float64) - expected)
        bad = diff > FLOAT_ATOL + FLOAT_RTOL * np.abs(expected)
        if bad.any():
            worst = int(np.argmax(np.where(bad, diff, 0)))
            return (f"{int(bad.sum())} pixels outside rtol {FLOAT_RTOL}; worst at flat index "
                    f"{worst}: {observed.flat[worst]!r} vs {expected.flat[worst]!r}")
        return None
    if not np.array_equal(observed, expected):
        return f"{int((observed != expected).sum())} pixels differ"
    return None


def _input_problems(batches: Sequence[EventArray], sensor_size: tuple[int, int]) -> list[str]:
    """Anything that would make the Accumulator do more than the kernel is timed on."""
    width, height = sensor_size
    problems = []
    for i, batch in enumerate(batches):
        try:
            validate(batch)
        except TypeError as error:
            problems.append(f"batch {i}: {error}")
            continue
        if len(batch) and (
            int(batch["x"].max()) >= width
            or int(batch["y"].max()) >= height
            or int(batch["t"].max()) >= TIMESTAMP_LIMIT
        ):
            problems.append(f"batch {i}: out-of-bounds coordinates or timestamps")
    return problems


def _running_watermarks(batches: Sequence[EventArray]) -> list[int | None]:
    marks: list[int | None] = []
    current: int | None = None
    for batch in batches:
        if len(batch):
            latest = int(batch["t"].max())
            current = latest if current is None else max(current, latest)
        marks.append(current)
    return marks


# ---------------------------------------------------------------- kernel level


class V1KernelTarget:
    """``kernel.begin_call(state)`` then ``kernel.accumulate(events, state, watermark)``."""

    name = "v1-kernel"
    level: Level = "kernel"
    statistic = "median_call"

    def supports(self, cell: Cell) -> bool:
        return cell.kernel in V1_KERNELS

    def timed_calls(self, cell: Cell, default: int, warmup_calls: int) -> int:
        return default

    def prepare(self, cell: Cell, batches: Sequence[EventArray]) -> Prepared:
        kernel = make_kernel(cell.kernel)
        state = kernel.init_state(cell.sensor_size)
        shape, dtype = kernel.output_spec(cell.sensor_size)
        problems = _input_problems(batches, cell.sensor_size)
        marks = _running_watermarks(batches)
        begin_call, accumulate = kernel.begin_call, kernel.accumulate
        watermark: list[int | None] = [None]
        calls = [0]

        def before_call() -> None:
            watermark[0] = marks[calls[0]]

        def call(events: EventArray) -> None:
            begin_call(state)
            accumulate(events, state, watermark[0])

        def after_call() -> None:
            calls[0] += 1

        def finish() -> Mapping[str, Any]:
            failures = list(problems)
            out = np.empty(shape, dtype=dtype)
            kernel.read(state, out, watermark[0])
            mismatch = compare(cell.kernel, out, reference(cell.kernel, cell.sensor_size, batches[: calls[0]]))
            if mismatch:
                failures.append(f"output: {mismatch}")
            return {"valid": not failures, "failures": failures, "calls": calls[0]}

        return Prepared(
            call=call,
            details={"output_dtype": str(dtype), **KERNEL_PARAMETERS.get(cell.kernel, {})},
            before_call=before_call,
            after_call=after_call,
            finish=finish,
        )

    def describe(self) -> dict[str, Any]:
        return _describe(self.name, self.level, self.statistic)


# ---------------------------------------------------------------- engine level


class VirtualTime:
    """Stands in for the ``time`` module inside ``frames2py._engine``: the Engine
    reads only ``time.monotonic_ns()``."""

    __slots__ = ("now_ns",)

    def __init__(self) -> None:
        self.now_ns = 0

    def monotonic_ns(self) -> int:
        return self.now_ns


def _check_clock_seam() -> None:
    if getattr(_engine_module, "time", None) is not time:
        raise RuntimeError(
            "frames2py._engine no longer reads the clock through the time module; "
            "the virtual clock can't be installed"
        )


def publication_schedule(calls: int, batch_size: int, interval_ms: float) -> list[bool]:
    """Which of the first *calls* ``ingest()`` calls publish under the virtual clock.

    From the contract: the first call publishes; after that a call publishes if at
    least the interval has passed since the last publication.
    """
    interval_ns = interval_ms * 1e6
    step = batch_size * NS_PER_EVENT
    schedule: list[bool] = []
    last: int | None = None
    for k in range(calls):
        now = k * step
        publish = last is None or now - last >= interval_ns
        if publish:
            last = now
        schedule.append(publish)
    return schedule


def engine_timed_calls(cell: Cell, default: int, warmup_calls: int) -> int:
    """*default*, raised for a positive interval until the timed calls contain at
    least ``MIN_TIMED_PUBLICATIONS`` publications."""
    if cell.interval_ms == 0:
        return default
    timed = 1
    while True:
        schedule = publication_schedule(warmup_calls + timed, cell.batch_size, cell.interval_ms)
        if sum(schedule[warmup_calls:]) >= MIN_TIMED_PUBLICATIONS:
            return max(default, timed)
        timed += 1


class V1EngineTarget:
    """``Engine.ingest(events)`` at the cell's interval, on the virtual clock."""

    name = "v1-engine"
    level: Level = "engine"
    statistic = "sustained"

    def supports(self, cell: Cell) -> bool:
        return cell.kernel in V1_KERNELS

    def timed_calls(self, cell: Cell, default: int, warmup_calls: int) -> int:
        return engine_timed_calls(cell, default, warmup_calls)

    def prepare(self, cell: Cell, batches: Sequence[EventArray]) -> Prepared:
        _check_clock_seam()
        clock = VirtualTime()
        module: Any = _engine_module
        module.time = clock
        try:
            engine = Engine(cell.sensor_size, make_kernel(cell.kernel), snapshot_interval_ms=cell.interval_ms)
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
                windowed = cell.kernel in ("event_count", "polarity")
                first = publications[-2]["call"] + 1 if windowed and len(publications) > 1 else 0
                mismatch = compare(
                    cell.kernel, snapshot.frame, reference(cell.kernel, cell.sensor_size, fed[first : last + 1])
                )
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
            }

        return Prepared(
            call=engine.ingest,
            counters=counters,
            details={
                "clock": "virtual",
                "virtual_step_ns": step,
                "target_rate_events_per_s": TARGET_RATE_EVENTS_PER_S,
                **KERNEL_PARAMETERS.get(cell.kernel, {}),
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
            "min_timed_publications": MIN_TIMED_PUBLICATIONS,
        }
