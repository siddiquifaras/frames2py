"""Consumer characterisation: the recorder's write rate, the viewer's rendering cost and its effect
on a live producer, and paced replay's timing. Characterisation, not a gate.

Run from the repo root (recordings from ``uv run python -m tests.recordings download NAME``):

    uv run --extra recorder python -m benchmarks recorder --out recorder.json
    uv run python -m benchmarks viewer --out viewer.json
    uv run --extra hdf5 python -m benchmarks replay --out replay.json
    uv run python -m benchmarks consumers-report recorder.json

Every run is a separate Python process, and the whole measurement runs inside
``power.hold_awake()``; a document whose power record shows sleep, or an end outside full
wake, or any failed check, is marked invalid. Cases are interleaved run by run.

**recorder** For each recording (its first ``--events`` events, decoded once into a
temporary ``.npy`` outside the repository), compression and write size: one untimed warm-up
recording; ``record``: ``recorder.open()``, every ``write()``, ``close()`` (the file is moved
into place, not fsynced; the page cache absorbs it), one ``perf_counter_ns`` interval with the
garbage collector off, plus each ``write()`` call's own time; ``record_ingest``: the same with
``Engine.ingest()`` (``event_count``, 16 ms) after each ``write()`` on the same thread. Checks:
the recording read back through ``frames2py.adapters.hdf5`` equals the source. The last run
adds a ``tracemalloc`` pass (peak traced bytes of recording alone).

**viewer** ``render``: for each kernel, resolution and distribution, a snapshot of an Engine
after one 16 ms window of the 20M events/s gate stream (320,000 events; the running kernels
after 1,600,000 events, 80 ms); 50 timed calls each of ``render()`` (automatic scaling),
``render(scale=...)`` (the same frame with its scale given, so no percentile) and the automatic
scale alone. Checks: the two renders are identical when given the same scale. ``impact``: a
producer thread paced at 20M events/s on the real clock ingests the uniform gate stream at
1280x720 into ``Engine(..., snapshot_interval_ms=16)``, for a warm-up and then a timed window,
once with no consumer and once with the viewer's own cadence loop on the main thread (16 ms,
rendering every new publication with ``render()``; no window), in alternating order across
runs. Per arm: achieved and busy producer rate, ``ingest()`` latency percentiles, lag behind
the arrival schedule, and the viewer's renders.

**replay** ``paced()`` over a recording's batches of 10,000 events (the first ``--seconds`` of it) on the real
clock, with and without ``Engine.ingest()`` as the consumer: each batch's lateness (yield
time minus its due time), and the replay's duration against the requested one. A fake-clock
pass over the same batches with a backward jump and a forward spike inserted checks that
every batch is yielded exactly at its due time.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np

from benchmarks import environment, power

SCHEMA: Final = "frames2py-consumer-benchmark/1"
_REPO = Path(__file__).resolve().parent.parent
_VERSIONS: Final = ("numpy", "h5py", "hdf5plugin", "pyglet")

RECORDER_RECORDINGS: Final = ("sparklers.raw", "active_marker.raw", "dsec_thun_01_a_events_left.h5")
RECORDER_COMPRESSIONS: Final = (None, "gzip", "blosc")
RECORDER_WRITE_SIZES: Final = (10_000, 100_000, 1_048_576)
RECORDER_EVENTS: Final = 20_000_000

VIEWER_KERNELS: Final = ("event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay")
VIEWER_RESOLUTIONS: Final = ((346, 260), (640, 480), (1280, 720))
VIEWER_DISTRIBUTIONS: Final = ("uniform", "clustered")
RENDER_CALLS: Final = 50
IMPACT_KERNELS: Final = ("event_count", "polarity", "time_surface", "timestamp_decay")
IMPACT_BATCHES: Final = (10_000, 100_000)
IMPACT_RATE: Final = 20_000_000
IMPACT_WARMUP_S: Final = 1.0
IMPACT_WINDOW_S: Final = 4.0

REPLAY_CASES: Final = (("active_marker.raw", 1.0), ("active_marker.raw", 4.0), ("sparklers.raw", 0.1))
REPLAY_SECONDS: Final = 5.0
REPLAY_BATCH: Final = 10_000


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _VERSIONS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _kernel(name: str) -> Any:
    import frames2py

    if name == "exp_decay":
        return frames2py.ExpDecay(0.95)
    if name == "timestamp_decay":
        return frames2py.TimestampDecay(10_000.0)
    return name


def _percentiles(values_ns: list[int]) -> dict[str, float]:
    ordered = np.sort(np.asarray(values_ns, dtype=np.float64))
    if ordered.size == 0:
        return {}
    q = np.percentile(ordered, [50, 95, 99])
    return {"p50_ns": float(q[0]), "p95_ns": float(q[1]), "p99_ns": float(q[2]), "max_ns": float(ordered[-1])}


def _timed(fn: Callable[[], Any]) -> tuple[int, Any]:
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        value = fn()
        return time.perf_counter_ns() - start, value
    finally:
        gc.enable()


# recorder ---------------------------------------------------------------------------


def _load_events(name: str, limit: int, directory: Path) -> tuple[Path, dict[str, Any]]:
    """The recording's first *limit* events as one ``.npy`` file, and what they are."""
    from tests import recordings

    recording = recordings.RECORDINGS[name]
    module = importlib.import_module(f"frames2py.adapters.{recording.adapter}")
    parts, count = [], 0
    with module.open(recordings.path(name), **recording.open_kwargs) as reader:
        size = reader.sensor_size
        for batch in reader:
            parts.append(batch[: limit - count])
            count += len(parts[-1])
            if count >= limit:
                break
    events = np.concatenate(parts)
    path = directory / f"{name}.npy"
    np.save(path, events)
    span_us = int(events["t"].max()) - int(events["t"].min())
    return path, {
        "name": name, "sha256": recording.file_sha256, "sensor_size": list(size or (0, 0)), "events": len(events),
        "span_us": span_us, "recording_events_per_s": len(events) / (span_us / 1e6) if span_us else None,
        "events_sha256": hashlib.sha256(events.tobytes()).hexdigest(),
    }


def _record(events: Any, size: tuple[int, int], compression: str | None, write_size: int, path: Path,
            engine: Any = None) -> list[int]:
    from frames2py import recorder

    calls = []
    with recorder.open(path, sensor_size=size, compression=compression, overwrite=True) as rec:
        for start in range(0, len(events), write_size):
            part = events[start : start + write_size]
            t = time.perf_counter_ns()
            rec.write(part)
            calls.append(time.perf_counter_ns() - t)
            if engine is not None:
                engine.ingest(part)
    return calls


def recorder_worker(request: dict[str, Any]) -> dict[str, Any]:
    import frames2py
    from frames2py.adapters import hdf5

    events = np.load(request["events_path"])
    size = tuple(request["sensor_size"])
    compression, write_size = request["compression"], request["write_size"]
    target = Path(request["scratch"]) / f"r{os.getpid()}.h5"
    _record(events[:100_000], size, compression, write_size, target)  # warm-up: imports, plugin load
    record_ns, calls = _timed(lambda: _record(events, size, compression, write_size, target))
    file_bytes = target.stat().st_size
    engine = frames2py.Engine(size, "event_count", snapshot_interval_ms=16.0)
    def record_and_ingest() -> None:
        _record(events, size, compression, write_size, target, engine)
        engine.stop()

    both_ns, _ = _timed(record_and_ingest)
    with hdf5.open(target, group="events") as reader:
        digest = hashlib.sha256()
        for batch in reader:
            digest.update(batch.tobytes())
    out: dict[str, Any] = {
        "runtime": environment.runtime(),
        "events": len(events),
        "record_ns": record_ns,
        "record_ingest_ns": both_ns,
        "write_call": _percentiles(calls),
        "file_bytes": file_bytes,
        "checks": {"round_trip": digest.hexdigest() == request["events_sha256"]},
    }
    if request["memory"]:
        tracemalloc.start()
        _record(events, size, compression, write_size, target)
        out["record_peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    target.unlink()
    out["max_rss_bytes"] = environment.max_rss_bytes()
    return out


# viewer -----------------------------------------------------------------------------


def _snapshot(kernel: str, size: tuple[int, int], distribution: str) -> Any:
    import frames2py
    from benchmarks.workloads import Workload

    running = kernel in ("time_surface", "exp_decay", "timestamp_decay")
    total = 1_600_000 if running else 320_000
    engine = frames2py.Engine(size, _kernel(kernel), snapshot_interval_ms=1e12)
    for batch in Workload(distribution, size, 100_000, seed=7).batches(total // 100_000):
        engine.ingest(batch)
    engine.stop()
    return engine.snapshot()


def render_worker(request: dict[str, Any]) -> dict[str, Any]:
    from frames2py.viewer import _render, render

    size = (int(request["sensor_size"][0]), int(request["sensor_size"][1]))
    snapshot = _snapshot(request["kernel"], size, request["distribution"])
    frame = snapshot.frame
    time_surface = frame.dtype == np.uint64
    scale = None if time_surface else _render._scale(frame, None)
    render(snapshot)
    auto, given, scale_only = [], [], []
    for _ in range(RENDER_CALLS):
        auto.append(_timed(lambda: render(snapshot))[0])
        given.append(_timed(lambda: render(snapshot, scale=scale))[0])
        if not time_surface:
            scale_only.append(_timed(lambda: _render._scale(frame, None))[0])
    tracemalloc.start()
    render(snapshot)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {
        "runtime": environment.runtime(),
        "nonzero_values": int(np.count_nonzero(frame)),
        "values": int(frame.size),
        "render_auto": _percentiles(auto),
        "render_given_scale": _percentiles(given),
        "auto_scale_only": _percentiles(scale_only),
        "render_peak_traced_bytes": peak,
        "checks": {"same_scale_same_image": time_surface or bool(np.array_equal(render(snapshot),
                                                                               render(snapshot, scale=scale)))},
    }


class _NoWindow:
    """The viewer loop's window, without a window: it counts what would be shown."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.shown = 0
        self.presents = 0

    def dispatch(self) -> bool:
        return not self.stop.is_set()

    def show(self, image: Any) -> None:
        self.shown += image is not None

    def present(self) -> None:
        self.presents += 1


def _impact_arm(kernel: str, batch_size: int, with_viewer: bool) -> dict[str, Any]:
    import frames2py
    from benchmarks.workloads import Workload
    from frames2py.viewer import _run, render

    size = (1280, 720)
    period_ns = batch_size * 1_000_000_000 // IMPACT_RATE
    count = math.ceil((IMPACT_WARMUP_S + IMPACT_WINDOW_S) * IMPACT_RATE / batch_size)
    pool = Workload("uniform", size, batch_size, seed=3).batches(min(count, 64))
    stride = batch_size * 1_000_000 // IMPACT_RATE  # µs of event time per batch
    engine = frames2py.Engine(size, _kernel(kernel), snapshot_interval_ms=16.0)
    stop = threading.Event()
    warmup = int(IMPACT_WARMUP_S * IMPACT_RATE / batch_size)
    latencies: list[int] = []
    lags: list[int] = []
    busy = [0]
    span = [0, 0]  # the first timed batch's due time, the last one's completion
    errors: list[BaseException] = []

    def produce() -> None:
        try:
            start = time.perf_counter_ns()
            for k in range(count):
                batch = pool[k % len(pool)].copy()
                batch["t"] += np.uint64((k // len(pool)) * len(pool) * stride)
                due = start + (k + 1) * period_ns
                while (now := time.perf_counter_ns()) < due:
                    time.sleep(min((due - now) / 1e9, 0.001))
                t = time.perf_counter_ns()
                engine.ingest(batch)
                done = time.perf_counter_ns()
                if k == warmup:
                    span[0] = due - period_ns
                if k >= warmup:
                    span[1] = done
                    latencies.append(done - t)
                    lags.append(done - due)
                    busy[0] += done - t
        except BaseException as exc:  # noqa: BLE001 - reported in the record
            errors.append(exc)
        finally:
            stop.set()

    producer = threading.Thread(target=produce)
    renders: list[int] = []

    def draw(snapshot: Any) -> Any:
        t = time.perf_counter_ns()
        image = render(snapshot)
        renders.append(time.perf_counter_ns() - t)
        return image

    window = _NoWindow(stop)
    wall = time.perf_counter_ns()
    producer.start()
    if with_viewer:
        _run.loop(engine.snapshot, window, interval_ns=16_000_000, draw=draw, clock=time.monotonic_ns,
                  sleep=time.sleep)
    producer.join()
    wall = time.perf_counter_ns() - wall
    timed = count - warmup
    return {
        "viewer": with_viewer,
        "errors": [repr(e) for e in errors],
        "timed_batches": timed,
        "achieved_events_per_s": timed * batch_size / ((span[1] - span[0]) / 1e9) if timed else None,
        "busy_events_per_s": timed * batch_size / (busy[0] / 1e9) if busy[0] else None,
        "ingest": _percentiles(latencies),
        "lag": {**_percentiles(lags), "final_ns": float(lags[-1]) if lags else None},
        "wall_ns": wall,
        "viewer_renders": len(renders),
        "viewer_presents": window.presents,
        "render": _percentiles(renders),
        "snapshots_published": engine.stats.snapshots_published,
    }


def impact_worker(request: dict[str, Any]) -> dict[str, Any]:
    order = [False, True] if request["viewer_first"] is False else [True, False]
    arms = {("viewer" if v else "none"): _impact_arm(request["kernel"], request["batch_size"], v) for v in order}
    return {"runtime": environment.runtime(), "order": ["viewer" if v else "none" for v in order], "arms": arms,
            "checks": {"no_errors": not any(a["errors"] for a in arms.values()),
                       "viewer_rendered": arms["viewer"]["viewer_renders"] > 0}}


# replay -----------------------------------------------------------------------------


def _replay_batches(name: str, seconds: float) -> tuple[list[Any], tuple[int, int] | None]:
    from tests import recordings

    recording = recordings.RECORDINGS[name]
    module = importlib.import_module(f"frames2py.adapters.{recording.adapter}")
    out: list[Any] = []
    with module.open(recordings.path(name), **{**recording.open_kwargs, "batch_size": REPLAY_BATCH}) as reader:
        size = reader.sensor_size
        t0 = None
        for batch in reader:
            t0 = int(batch["t"].min()) if t0 is None else t0
            out.append(batch)
            if int(batch["t"].max()) - t0 > seconds * 1e6:
                break
    return out, size


def _due_offsets(batches: list[Any], speed: float) -> list[int]:
    """Each batch's due time after the start, by the running maximum (an independent recomputation)."""
    t0: int | None = None
    top, due = 0, list[int]()
    for batch in batches:
        if len(batch) == 0:
            due.append(0 if not due else due[-1])
            continue
        lo, hi = int(batch["t"].min()), int(batch["t"].max())
        t0 = lo if t0 is None else t0
        top = max(top, hi)
        due.append(math.ceil((top - t0) * 1000 / speed))
    return due


def replay_worker(request: dict[str, Any]) -> dict[str, Any]:
    import frames2py
    from frames2py.replay import paced

    batches, size = _replay_batches(request["recording"], request["seconds"])
    speed = request["speed"]
    due = _due_offsets(batches, speed)
    engine = frames2py.Engine(size or (1280, 720), "event_count") if request["ingest"] else None
    late: list[int] = []
    readings: list[int] = []

    def clock() -> int:  # the real clock, keeping paced()'s first reading: its start
        now = time.monotonic_ns()
        if not readings:
            readings.append(now)
        return now

    for k, batch in enumerate(paced(batches, speed, clock=clock)):
        late.append(time.monotonic_ns() - (readings[0] + due[k]))
        if engine is not None:
            engine.ingest(batch)
    duration = time.monotonic_ns() - readings[0] if readings else 0

    # Fake clock: the same batches with a backward jump and a forward spike inserted.
    jumped = [b.copy() for b in batches]
    middle = len(jumped) // 2
    for b in jumped[middle : middle + 3]:
        b["t"] -= np.minimum(b["t"], np.uint64(500_000))
    at = min(middle + 5, len(jumped) - 1)
    spike = jumped[at].copy()
    spike["t"][:1] = spike["t"].max() + np.uint64(2_000_000)
    jumped[at] = spike
    fake_due = _due_offsets(jumped, speed)
    fake = [10**12]

    def sleep(seconds: float) -> None:
        fake[0] += round(seconds * 1e9)

    exact = 0
    for k, _ in enumerate(paced(jumped, speed, clock=lambda: fake[0], sleep=sleep)):
        exact += fake[0] - 10**12 == fake_due[k]
    return {
        "runtime": environment.runtime(),
        "batches": len(batches),
        "events": sum(len(b) for b in batches),
        "requested_ns": due[-1],
        "duration_ns": duration,
        "lateness": _percentiles(late),
        "early_batches": sum(1 for x in late if x < 0),
        "fake_clock_exact": exact,
        "checks": {"never_early": all(x >= 0 for x in late), "fake_clock_exact": exact == len(jumped)},
    }


# driver -----------------------------------------------------------------------------

WORKERS: Final[dict[str, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "recorder": recorder_worker, "render": render_worker, "impact": impact_worker, "replay": replay_worker,
}


def worker(request: dict[str, Any]) -> dict[str, Any]:
    return WORKERS[request["kind"]](request)


def _spawn(request: dict[str, Any], python: str) -> dict[str, Any]:
    done = subprocess.run([python, "-m", "benchmarks", "consumers-worker"], input=json.dumps(request),
                          stdout=subprocess.PIPE, text=True, cwd=_REPO, check=True)
    record: dict[str, Any] = json.loads(done.stdout)
    return record


def _cases(kind: str, args: dict[str, Any], scratch: Path) -> Iterator[dict[str, Any]]:
    if kind == "recorder":
        for name in args.get("recordings") or RECORDER_RECORDINGS:
            path, source = _load_events(name, args.get("events") or RECORDER_EVENTS, scratch)
            for compression in RECORDER_COMPRESSIONS:
                for write_size in RECORDER_WRITE_SIZES:
                    yield {"kind": "recorder", "source": source, "events_path": str(path), "scratch": str(scratch),
                           "sensor_size": source["sensor_size"], "events_sha256": source["events_sha256"],
                           "compression": compression, "write_size": write_size}
    elif kind == "viewer":
        for kernel in VIEWER_KERNELS:
            for size in VIEWER_RESOLUTIONS:
                for distribution in VIEWER_DISTRIBUTIONS:
                    yield {"kind": "render", "kernel": kernel, "sensor_size": list(size), "distribution": distribution}
        for kernel in IMPACT_KERNELS:
            for batch_size in IMPACT_BATCHES:
                yield {"kind": "impact", "kernel": kernel, "batch_size": batch_size}
    elif kind == "replay":
        for name, speed in REPLAY_CASES:
            for ingest in (False, True):
                yield {"kind": "replay", "recording": name, "speed": speed, "ingest": ingest,
                       "seconds": args.get("seconds") or REPLAY_SECONDS}
    else:
        raise ValueError(f"unknown characterisation {kind!r}")


def run(kind: str, *, runs: int = 5, **args: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="frames2py-consumers-") as tmp, \
            power.hold_awake(f"frames2py {kind} characterisation") as power_record:
        env = environment.capture()
        cases = list(_cases(kind, args, Path(tmp)))
        results: list[list[dict[str, Any]]] = [[] for _ in cases]
        for i in range(runs):
            for c, case in enumerate(cases):
                request = {**case, "memory": i == runs - 1, "viewer_first": i % 2 == 1}
                results[c].append(_spawn(request, sys.executable))
                print(f"run {i + 1}/{runs} case {c + 1}/{len(cases)} {kind}", file=sys.stderr, flush=True)
        env_end = environment.capture()
    checks = all(all(r["checks"].values()) for rs in results for r in rs)
    valid = not power.slept_during(power_record) and not power.ended_outside_full_wake(power_record) and checks
    return {
        "schema": SCHEMA, "kind": kind, "policy": {"runs": runs, "process_per_run": True, **args},
        "versions": _versions(), "environment": env, "environment_end": env_end, "power": power_record,
        "valid": valid, "cases": [{"case": case, "runs": rs} for case, rs in zip(cases, results)],
    }


def _median(values: list[float | None]) -> float:
    """The median of the values present; NaN if none is."""
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else math.nan


def report(document: dict[str, Any]) -> str:
    if document.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document")
    env = document["environment"]
    lines = [
        f"{document['kind']} characterisation{'' if document['valid'] else '  INVALID'}: {env['chip']}, {env['os']}, "
        f"Python {env['python']} (GIL {'on' if env['gil_enabled'] else 'off'}), NumPy {env['numpy']}, "
        f"commit {str(env['commit'])[:7]}; {document['policy']['runs']} runs, medians",
        "  " + ", ".join(f"{k} {v}" for k, v in document["versions"].items() if v),
    ]
    for entry in document["cases"]:
        case, rs = entry["case"], entry["runs"]
        if case["kind"] == "recorder":
            src, n = case["source"], rs[0]["events"]
            rate = _median([n / (r["record_ns"] / 1e9) for r in rs])
            both = _median([n / (r["record_ingest_ns"] / 1e9) for r in rs])
            p99 = _median([r["write_call"].get("p99_ns") for r in rs])
            peak = rs[-1].get("record_peak_traced_bytes")
            lines.append(
                f"  {src['name']:<34} {str(case['compression']):<6} write {case['write_size']:>9,}: "
                f"record {rate / 1e6:7.1f} M/s ({rate / src['recording_events_per_s']:6.1f}x its "
                f"{src['recording_events_per_s'] / 1e6:.2f} M/s), with ingest {both / 1e6:7.1f} M/s, "
                f"write p99 {p99 / 1e6:7.2f} ms, {rs[0]['file_bytes'] / n:5.2f} B/event"
                + (f", peak {peak / 2**20:.1f} MiB" if peak is not None else "")
            )
        elif case["kind"] == "render":
            auto = _median([r["render_auto"]["p50_ns"] for r in rs])
            given = _median([r["render_given_scale"]["p50_ns"] for r in rs])
            scale = _median([r["auto_scale_only"].get("p50_ns") for r in rs])
            w, h = case["sensor_size"]
            lines.append(
                f"  render {case['kernel']:<16} {w}x{h} {case['distribution']:<9} nonzero {rs[0]['nonzero_values']:>8,}: "
                f"auto {auto / 1e6:6.2f} ms, given scale {given / 1e6:6.2f} ms"
                + (f", auto scale alone {scale / 1e6:6.2f} ms" if scale is not None else "")
                + f", peak {rs[-1]['render_peak_traced_bytes'] / 2**20:.1f} MiB"
            )
        elif case["kind"] == "impact":
            parts = []
            for arm in ("none", "viewer"):
                busy = _median([r["arms"][arm]["busy_events_per_s"] for r in rs])
                p99 = _median([r["arms"][arm]["ingest"]["p99_ns"] for r in rs])
                lag = _median([r["arms"][arm]["lag"]["max_ns"] for r in rs])
                parts.append(f"{arm}: busy {busy / 1e6:6.1f} M/s, ingest p99 {p99 / 1e3:7.1f} us, max lag {lag / 1e6:6.2f} ms")
            renders = _median([r["arms"]["viewer"]["viewer_renders"] for r in rs])
            ratio = _median([r["arms"]["viewer"]["busy_events_per_s"] / r["arms"]["none"]["busy_events_per_s"]
                             for r in rs])
            lines.append(f"  impact {case['kernel']:<16} {case['batch_size']:>7,} @ 16 ms: " + "; ".join(parts)
                         + f"; busy viewer/none {ratio:.2f}; {renders:.0f} renders")
        elif case["kind"] == "replay":
            late = [r["lateness"] for r in rs]
            ratio = _median([r["duration_ns"] / r["requested_ns"] for r in rs if r["requested_ns"]])
            lines.append(
                f"  replay {case['recording']:<20} x{case['speed']:<4} {'ingest' if case['ingest'] else 'no consumer'}: "
                f"{rs[0]['batches']} batches, duration/requested {ratio:.4f}, lateness p50 "
                f"{_median([x['p50_ns'] for x in late]) / 1e3:.0f} us, p99 {_median([x['p99_ns'] for x in late]) / 1e3:.0f} us, "
                f"max {_median([x['max_ns'] for x in late]) / 1e3:.0f} us; fake clock exact "
                f"{rs[0]['fake_clock_exact']}/{rs[0]['batches']}"
            )
    return "\n".join(lines)
