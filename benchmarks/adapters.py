"""Adapter characterisation: what reading a real recording costs, and how that compares with ingesting it.

Characterisation, not a gate: nothing here passes or fails on speed. Run from the repo root:

    uv run python -m tests.recordings download active_marker.raw
    uv run python -m benchmarks adapters --recording active_marker.raw --out result.json
    uv run python -m benchmarks adapters-report result.json

Method, per recording. Each run is a separate Python process; inside it, in this order:

1. warm-up: one untimed pass of the adapter over the whole file (imports, page cache);
2. ``source_read``: the file's bytes read in 1 MiB ``read()`` calls, nothing decoded;
3. ``decode``: ``open()`` and iteration over every batch, nothing else;
4. an untimed pass that keeps the batches in memory;
5. ``ingest``: ``Engine.ingest()`` on those batches, back to back, then ``stop()``;
6. ``end_to_end``: the adapter feeding ``Engine.ingest()`` batch by batch, then ``stop()``.

Timed passes run with the garbage collector disabled; each is one ``perf_counter_ns``
interval. The reader uses its defaults (``batch_size=None``). The Engine uses the chosen
kernel and interval against the real clock, so publications follow wall time. The last run
adds an untimed ``tracemalloc`` pass of the adapter alone (peak traced bytes).

Per run: decode rate (events / decode time), the recording's own rate (events / its
timestamp span), real-time factor (timestamp span / decode time), ingest and end-to-end
rates, and the adapter's share of end-to-end time, ``1 - ingest / end_to_end``. Across runs:
medians. Every run checks the decoded event count against the registry.

The whole measurement runs inside ``power.hold_awake()``; the document records the power
state at start and end, and a run that slept or ended outside full wake is marked invalid.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Final

from benchmarks import environment, power

SCHEMA: Final = "frames2py-adapter-benchmark/1"
READ_SIZE: Final = 1 << 20
BACKENDS: Final = ("numpy", "dv-processing", "h5py", "hdf5plugin")
_REPO = Path(__file__).resolve().parent.parent


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in BACKENDS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    try:
        out["hdf5 library"] = importlib.import_module("h5py").version.hdf5_version
    except ImportError:
        out["hdf5 library"] = None
    return out


def _open(adapter: str, path: Path, kwargs: dict[str, Any]) -> Any:
    module = importlib.import_module(f"frames2py.adapters.{adapter}")
    return module.open(path, **kwargs)


def _timed(fn: Any) -> tuple[int, Any]:
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        value = fn()
        return time.perf_counter_ns() - start, value
    finally:
        gc.enable()


def measure(adapter: str, path: Path, open_kwargs: dict[str, Any], kernel: str, interval_ms: float,
            memory: bool) -> dict[str, Any]:
    """One run in this process."""
    from frames2py import Engine

    def decode() -> tuple[int, int, int]:
        n, t_min, t_max = 0, None, None
        with _open(adapter, path, open_kwargs) as reader:
            for batch in reader:
                n += len(batch)
                lo, hi = int(batch["t"].min()), int(batch["t"].max())
                t_min = lo if t_min is None else min(t_min, lo)
                t_max = hi if t_max is None else max(t_max, hi)
        return n, t_min or 0, t_max or 0

    def source_read() -> int:
        size = 0
        with path.open("rb") as f:
            while block := f.read(READ_SIZE):
                size += len(block)
        return size

    decode()
    read_ns, file_bytes = _timed(source_read)
    decode_ns, (events, t_min, t_max) = _timed(decode)
    with _open(adapter, path, open_kwargs) as reader:
        sensor_size = reader.sensor_size
        batches = list(reader)

    def ingest(parts: list[Any]) -> int:
        engine = Engine(sensor_size, kernel, snapshot_interval_ms=interval_ms)
        for batch in parts:
            engine.ingest(batch)
        engine.stop()
        return int(engine.stats.snapshots_published)

    def end_to_end() -> int:
        engine = Engine(sensor_size, kernel, snapshot_interval_ms=interval_ms)
        with _open(adapter, path, open_kwargs) as reader:
            for batch in reader:
                engine.ingest(batch)
        engine.stop()
        return int(engine.stats.snapshots_published)

    ingest_ns, published = _timed(lambda: ingest(batches))
    batches.clear()
    e2e_ns, published_e2e = _timed(end_to_end)
    record: dict[str, Any] = {
        "events": events,
        "file_bytes": file_bytes,
        "t_min": t_min,
        "t_max": t_max,
        "sensor_size": list(sensor_size) if sensor_size else None,
        "source_read_ns": read_ns,
        "decode_ns": decode_ns,
        "ingest_ns": ingest_ns,
        "end_to_end_ns": e2e_ns,
        "snapshots_published": {"ingest": published, "end_to_end": published_e2e},
        "runtime": environment.runtime(),
    }
    if memory:
        tracemalloc.start()
        try:
            decode()
            record["decode_peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    record["max_rss_bytes"] = environment.max_rss_bytes()
    return record


def derived(record: dict[str, Any]) -> dict[str, float | None]:
    span_s = (record["t_max"] - record["t_min"]) / 1e6
    decode_s, e2e_s = record["decode_ns"] / 1e9, record["end_to_end_ns"] / 1e9
    events = record["events"]
    return {
        "source_read_mb_per_s": record["file_bytes"] / 1e6 / (record["source_read_ns"] / 1e9),
        "decode_events_per_s": events / decode_s,
        "recording_events_per_s": events / span_s if span_s > 0 else None,
        "real_time_factor": span_s / decode_s,
        "ingest_events_per_s": events / (record["ingest_ns"] / 1e9),
        "end_to_end_events_per_s": events / e2e_s,
        "adapter_share_of_end_to_end": 1 - record["ingest_ns"] / record["end_to_end_ns"],
    }


def worker(request: dict[str, Any]) -> dict[str, Any]:
    return measure(request["adapter"], Path(request["path"]), request["open_kwargs"], request["kernel"],
                   request["interval_ms"], request["memory"])


def run(name: str, *, runs: int = 5, kernel: str = "event_count", interval_ms: float = 16.0,
        process_per_run: bool = True, path: Path | None = None, adapter: str | None = None,
        open_kwargs: dict[str, Any] | None = None, expected_events: int | None = None) -> dict[str, Any]:
    """Measure a registered recording (or, for tests, any file given by *path* and *adapter*)."""
    from tests import recordings

    if path is None:
        recording = recordings.RECORDINGS[name]
        path, adapter = recordings.path(name), recording.adapter
        open_kwargs, expected_events = dict(recording.open_kwargs), recording.events
        source = {"name": name, "sha256": recording.file_sha256, "licence": recording.licence,
                  "source": recording.source, "format": recording.format}
    else:
        source = {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert adapter is not None and open_kwargs is not None
    records = []
    with power.hold_awake(f"frames2py adapter benchmark {name}") as power_record:
        env = environment.capture()
        for i in range(runs):
            request = {"adapter": adapter, "path": str(path), "open_kwargs": open_kwargs, "kernel": kernel,
                       "interval_ms": interval_ms, "memory": i == runs - 1}
            if process_per_run:
                done = subprocess.run([sys.executable, "-m", "benchmarks", "adapters-worker"], input=json.dumps(request),
                                      stdout=subprocess.PIPE, text=True, cwd=_REPO, check=True)
                record = json.loads(done.stdout)
            else:
                record = worker(request)
            record["checks"] = {"events_match": expected_events is None or record["events"] == expected_events}
            records.append(record)
            print(f"run {i + 1}/{runs} {name}: {record['events']} events, decode {record['decode_ns'] / 1e6:.1f} ms",
                  file=sys.stderr, flush=True)
        env_end = environment.capture()
    valid = (not power.slept_during(power_record) and not power.ended_outside_full_wake(power_record)
             and all(r["checks"]["events_match"] for r in records))
    per_run = [derived(r) for r in records]
    summary = {key: statistics.median(v for v in (d[key] for d in per_run) if v is not None)
               for key in per_run[0] if per_run[0][key] is not None}
    return {
        "schema": SCHEMA,
        "recording": source,
        "adapter": adapter,
        "open_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in open_kwargs.items()},
        "read_size": READ_SIZE,
        "engine": {"kernel": kernel, "snapshot_interval_ms": interval_ms},
        "policy": {"runs": runs, "process_per_run": process_per_run},
        "backends": _versions(),
        "environment": env,
        "environment_end": env_end,
        "power": power_record,
        "valid": valid,
        "runs": records,
        "per_run": per_run,
        "summary": summary,
    }


def report(document: dict[str, Any]) -> str:
    if document.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document")
    env, s, first = document["environment"], document["summary"], document["runs"][0]
    lines = [
        f"{document['recording']['name']} via frames2py.adapters.{document['adapter']}"
        f"{'' if document['valid'] else '  INVALID'}",
        f"  {env['chip']}, {env['os']}, Python {env['python']} (GIL {'on' if env['gil_enabled'] else 'off'}), "
        f"NumPy {env['numpy']}, commit {str(env['commit'])[:7]}",
        f"  backends: {', '.join(f'{k} {v}' for k, v in document['backends'].items() if v)}",
        f"  {first['events']:,} events, {first['file_bytes']:,} bytes, span {(first['t_max'] - first['t_min']) / 1e6:.2f} s; "
        f"{document['policy']['runs']} runs, medians; Engine {document['engine']['kernel']} "
        f"@ {document['engine']['snapshot_interval_ms']:g} ms",
        f"  source read      {s['source_read_mb_per_s']:10.1f} MB/s",
        f"  decode           {s['decode_events_per_s'] / 1e6:10.2f} M events/s   "
        f"({s['real_time_factor']:.1f}x the recording's {s.get('recording_events_per_s', 0) / 1e6:.3f} M events/s)",
        f"  ingest only      {s['ingest_events_per_s'] / 1e6:10.2f} M events/s",
        f"  end to end       {s['end_to_end_events_per_s'] / 1e6:10.2f} M events/s   "
        f"(adapter {100 * s['adapter_share_of_end_to_end']:.0f}% of the time)",
    ]
    peak = document["runs"][-1].get("decode_peak_traced_bytes")
    if peak is not None:
        lines.append(f"  decode peak traced memory {peak / 2**20:.1f} MiB; worker max RSS "
                     f"{max(r['max_rss_bytes'] for r in document['runs']) / 2**20:.0f} MiB")
    return "\n".join(lines)
