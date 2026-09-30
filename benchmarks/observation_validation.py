"""Validation stages V1, V3 (its metrics), V5 and V6 of the observation study (preregistration 22).

V1 and V6 drive each arm single-threaded, in lockstep: a producer step, then every consumer
ticks until it has nothing more to take. V1 advances a virtual cadence clock by one batch
period per step, and gives the Engine the same clock through the gate's substitution of
``frames2py._engine.time`` (21.1: V1 only). V6 paces steps on the real clock instead and
measures allocations with ``tracemalloc``. Neither is a timing measurement.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import statistics
import time
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from benchmarks.measure import nearest_rank
from benchmarks.observation import (
    POOL_SHA256,
    PREREGISTERED,
    VirtualClock,
    Condition,
    Context,
    Lockstep,
    Source,
    W5_TARGET_S,
    build_arm,
    make_work,
)
from frames2py.publish import Snapshot

V1_ARMS: Final = ("A", "B", "C", "E", "F", "G", "H", "RB")
V1_BATCHES: Final = 400
V1_INTERVALS_MS: Final = (16.0, 0.0)
V6_ARMS: Final = ("A", "B", "C", "E", "F", "G", "H", "H'", "RB", "EP-H", "EP-QB", "EP-QE")
V6_BATCHES: Final = 200
V5_UNIT_CALLS: Final = 50
V5_CHECK_CALLS: Final = 20
V5_TOLERANCE: Final = 0.03
V5_MAX_ITERATIONS: Final = 5


def _digest(frame: NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


@contextlib.contextmanager
def _engine_clock(clock: VirtualClock) -> Iterator[None]:
    """The gate's virtual-clock substitution (``benchmarks/targets/v1.py``), for V1 only."""
    import frames2py._engine

    module: Any = frames2py._engine
    if getattr(module, "time", None) is not time:
        raise RuntimeError("frames2py._engine no longer reads the clock through the time module")
    module.time = clock
    try:
        yield
    finally:
        module.time = time


def _lockstep_publications(arm_name: str, condition: Condition, source: Source, batches: int) -> dict[str, Any]:
    clock = VirtualClock()
    seen: list[tuple[str, int, int]] = []

    def record(frame: NDArray[Any], watermark: int, sequence: int, snapshot: Snapshot | None) -> None:
        seen.append((_digest(frame), watermark, sequence))

    period = round(condition.batch_period_ns)
    ctx = Context(condition, record, full=False, cadence_clock=clock)

    def before(k: int) -> None:
        clock.now_ns = k * period

    with _engine_clock(clock):
        arm = build_arm(arm_name, 1, ctx)
        Lockstep(arm, source, batches, before_step=before).run()
        pending: str | None = None
        if arm.engine is not None:
            before_stop = arm.engine.stats.snapshots_published
            arm.engine.stop()
            snap = arm.engine.snapshot()
            if arm.engine.stats.snapshots_published > before_stop and snap is not None:
                pending = _digest(snap.frame)
        elif arm_name == "RB":
            consumer = arm.consumers[0]
            pending = _digest(getattr(consumer, "acc").read())
        else:
            pending = _digest(getattr(arm, "acc").read())
    return {"publications": seen, "pending": pending}


def v1(condition: Condition = PREREGISTERED, batches: int = V1_BATCHES) -> dict[str, Any]:
    """V1: the published (frame bytes, watermark, sequence) are identical across the arms, and
    every window equals the independent reference. Frame bytes are compared by SHA-256."""
    from benchmarks.targets.v1 import reference

    out: dict[str, Any] = {"stage": "V1", "batches": batches, "intervals": {}}
    ok = True
    for interval in V1_INTERVALS_MS:
        c = dataclasses.replace(condition, interval_ms=interval)
        source = Source(c)
        same_pool = (c.sensor_size, c.batch_size, c.rate_hz, c.pool_events) == (
            PREREGISTERED.sensor_size, PREREGISTERED.batch_size, PREREGISTERED.rate_hz, PREREGISTERED.pool_events)
        digest_ok = not same_pool or source.digest() == POOL_SHA256
        results = {arm: _lockstep_publications(arm, c, source, batches) for arm in V1_ARMS}
        engine = results["H"]["publications"]
        schedule = _schedule(c, batches)
        windows = list(_windows(schedule))
        problems: list[str] = []
        if len(engine) != len(windows):
            problems.append(f"the Engine published {len(engine)} times; the schedule has {len(windows)}")
        for (digest, watermark, _), (first, last) in zip(engine, windows):
            expected = reference("event_count", c.sensor_size, [source.materialise(k) for k in range(first, last + 1)])
            if _digest(expected) != digest:
                problems.append(f"window {first}..{last}: the Engine's frame differs from the reference")
            if watermark != source.max_t(last):
                problems.append(f"window {first}..{last}: watermark {watermark} != m_{last}")
        tail = range(windows[-1][1] + 1 if windows else 0, batches)
        pending_ref = (_digest(reference("event_count", c.sensor_size, [source.materialise(k) for k in tail]))
                       if len(tail) else None)
        for arm, result in results.items():
            if result["publications"] != engine:
                problems.append(f"{arm}: published sequence differs from the Engine's")
            if pending_ref is not None and result["pending"] != pending_ref:
                problems.append(f"{arm}: the final pending window differs from the reference")
        if not digest_ok:
            problems.append("pool digest mismatch")
        ok = ok and not problems
        out["intervals"][str(interval)] = {
            "publications": len(engine), "pending_batches": len(tail), "pool_digest_ok": digest_ok,
            "problems": problems, "arms": list(results),
        }
    out["ok"] = ok
    return out


def _schedule(condition: Condition, batches: int) -> list[bool]:
    """Which steps publish under the virtual clock, from the contract's cadence rule."""
    period = round(condition.batch_period_ns)
    last: int | None = None
    out = []
    for k in range(batches):
        now = k * period
        publish = last is None or now - last >= condition.interval_ns
        if publish:
            last = now
        out.append(publish)
    return out


def _windows(schedule: list[bool]) -> Iterator[tuple[int, int]]:
    first = 0
    for k, published in enumerate(schedule):
        if published:
            yield first, k
            first = k + 1


def v6(arm_name: str, tmp_dir: Path, condition: Condition = PREREGISTERED, batches: int = V6_BATCHES) -> dict[str, Any]:
    """V6: peak temporary bytes per producer step and per observation, under ``tracemalloc``,
    one arm at N = 1 with W1, steps paced at the batch period on the real clock."""
    source = Source(condition)
    ctx = Context(condition, make_work("W1", None), full=False)
    recorder = None
    path = tmp_dir / f"v6_{arm_name.replace(chr(39), 'p')}.h5"
    if arm_name.startswith("EP-"):
        from frames2py import recorder as recorder_module

        tmp_dir.mkdir(parents=True, exist_ok=True)
        recorder = recorder_module.open(path, sensor_size=condition.sensor_size, compression="blosc", overwrite=True)
    arm = build_arm(arm_name, 1, ctx, recorder=recorder)
    steps: list[int] = []
    observations: list[int] = []
    recorder_writes: list[int] = []
    period = condition.batch_period_ns
    tracemalloc.start()
    try:
        start = time.perf_counter_ns()
        for k in range(batches):
            while (r := start + (k + 1) * period - time.perf_counter_ns()) > 0:
                time.sleep(min(r, 1_000_000) / 1e9)
            batch = source.materialise(k)
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            arm.step(k, batch)
            steps.append(tracemalloc.get_traced_memory()[1] - base)
            for consumer in arm.consumers:
                while True:
                    before = consumer.log.observations
                    tracemalloc.reset_peak()
                    base = tracemalloc.get_traced_memory()[0]
                    took = consumer.tick(None)
                    if consumer.log.observations > before:
                        observations.append(tracemalloc.get_traced_memory()[1] - base)
                    if not took or consumer.kind == "poll":
                        break
            rc = getattr(arm, "recorder_consumer", None)
            while rc is not None:
                tracemalloc.reset_peak()
                base = tracemalloc.get_traced_memory()[0]
                if not rc.tick(None):
                    break
                recorder_writes.append(tracemalloc.get_traced_memory()[1] - base)
    finally:
        tracemalloc.stop()
        if recorder is not None:
            recorder.close()
            path.unlink(missing_ok=True)
    return {"stage": "V6", "arm": arm_name, "batches": batches,
            "step_peak_bytes": _summary(steps), "observation_peak_bytes": _summary(observations),
            "recorder_write_peak_bytes": _summary(recorder_writes),
            "observations": sum(c.log.observations for c in arm.consumers) + sum(log.observations for log in arm.inline)}


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "p50": nearest_rank(values, 50), "p99": nearest_rank(values, 99), "max": max(values),
            "mean": statistics.fmean(values)}


def v5(target_s: float = W5_TARGET_S) -> dict[str, Any]:
    """V5: K5 so that W5 takes the target in isolation (runtime A, one thread, no producer)."""
    frame = np.zeros((PREREGISTERED.sensor_size[1], PREREGISTERED.sensor_size[0]), dtype=np.uint32)
    target_ns = target_s * 1e9
    sequence = [0]

    def median_ns(k: int, calls: int) -> tuple[float, list[int]]:
        work = make_work("W5", k)
        samples = []
        for _ in range(calls):
            sequence[0] += 1
            t = time.perf_counter_ns()
            work(frame, 0, sequence[0], None)
            samples.append(time.perf_counter_ns() - t)
        return statistics.median(samples), samples

    unit, unit_samples = median_ns(1, V5_UNIT_CALLS)
    k = max(1, round(target_ns / unit))
    iterations = []
    for _ in range(V5_MAX_ITERATIONS + 1):
        median, samples = median_ns(k, V5_CHECK_CALLS)
        within = abs(median - target_ns) <= V5_TOLERANCE * target_ns
        iterations.append({"k": k, "median_ns": median, "samples_ns": samples, "within": within})
        if within:
            break
        if len(iterations) > V5_MAX_ITERATIONS:
            break
        k = max(1, round(k * target_ns / median))
    ok = iterations[-1]["within"]
    return {"stage": "V5", "target_ns": target_ns, "unit_calls": V5_UNIT_CALLS, "unit_median_ns": unit,
            "unit_samples_ns": unit_samples, "iterations": iterations, "ok": ok, "k5": k if ok else None}


def w5_cost(k5: int, calls: int = V5_CHECK_CALLS) -> dict[str, Any]:
    """W5's cost at K5 on this runtime, measured and reported, not re-calibrated (8)."""
    frame = np.zeros((PREREGISTERED.sensor_size[1], PREREGISTERED.sensor_size[0]), dtype=np.uint32)
    work = make_work("W5", k5)
    samples = []
    for i in range(calls):
        t = time.perf_counter_ns()
        work(frame, 0, i + 1, None)
        samples.append(time.perf_counter_ns() - t)
    return {"k5": k5, "median_ns": statistics.median(samples), "samples_ns": samples}


def v3_metrics(record: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> dict[str, Any]:
    """V3's two figures from either instrumentation mode: busy time per event and step p99."""
    t0, t1 = record["t0"], record["t1"]
    s, e = arrays["producer_s"], arrays["producer_e"]
    b = Condition.from_record(record["request"]["condition"]).batch_size if "condition" in record["request"] \
        else PREREGISTERED.batch_size
    in_w = (s >= t0) & (s < t1)
    steps = (e - s)[in_w]
    events = int(in_w.sum()) * b
    return {"busy_ns_per_event": float(steps.sum()) / events if events else None,
            "step_p99_us": nearest_rank(steps.tolist(), 99) / 1e3 if len(steps) else None, "steps": int(len(steps))}


def validate_worker(request: dict[str, Any]) -> dict[str, Any]:
    """One validation stage in this (study-environment) process."""
    from benchmarks.observation import runtime_problems

    stage = request["stage"]
    problems = runtime_problems(request) if request.get("check_runtime", True) else []
    if problems:
        return {"stage": stage, "refused": problems}
    if stage == "V1":
        return v1()
    if stage == "V6":
        return {"stage": "V6", "arms": [v6(arm, Path(request["tmp_dir"])) for arm in V6_ARMS]}
    if stage == "V5":
        return v5()
    if stage == "W5_COST":
        return w5_cost(int(request["k5"]))
    raise ValueError(f"unknown validation stage {stage!r}")
