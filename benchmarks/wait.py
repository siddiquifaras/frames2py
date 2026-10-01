"""The ``wait_for_newer`` producer-cost measurement (``benchmarks/wait_preregistration.md``).

What is fixed here, by the preregistration: the builds, labels, runtimes, cells, repetitions
and pass order; and the worker, which makes one run of one cell under one label in its own
process, with ``frames2py`` imported from that label's build. The driver
(``benchmarks/wait_driver.py``) runs the workers and checks the machine around them; the
analysis (``benchmarks/wait_analysis.py``) applies the preregistered rule.

Run the worker as ``python -m benchmarks.wait worker`` with the request as JSON on stdin and
``PYTHONPATH`` set to the build's ``src``. It needs only NumPy and the build.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import random
import statistics
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parent.parent
PREREGISTRATION: Final = Path(__file__).with_name("wait_preregistration.md")
MEASURE_DIR: Final = REPO / "scratch" / "v11_phase1" / "measure"

BUILDS: Final = {
    "baseline": "e63150e1da26d1177d0f7246ab6123d0140d214a",
    "feature": "c0b6d9a4df7d87eafb27ea6b791e01b0418f3904",
}
"""The commit each build is checked out at, in its own worktree under ``MEASURE_DIR/builds``."""
BUILD_DIRS: Final = {name: MEASURE_DIR / "builds" / name for name in BUILDS}

LABELS: Final = {"B": ("baseline", 0), "B'": ("baseline", 0), "F0": ("feature", 0),
                 "F1": ("feature", 1), "F4": ("feature", 4), "F8": ("feature", 8)}
"""label: (build, waiters). B' is the A/A control: the baseline build again, under a second label."""
LABEL_ORDER: Final = ("B", "B'", "F0", "F1", "F4", "F8")

RUNTIMES: Final = ("A", "B")
ENVIRONMENTS: Final = {"A": MEASURE_DIR / "envs" / "py311" / "bin" / "python",
                       "B": MEASURE_DIR / "envs" / "py314t" / "bin" / "python"}
RUNTIME_VERSIONS: Final = {"A": "3.11.14", "B": "3.14.2"}
FREE_THREADED: Final = {"A": False, "B": True}
NUMPY_VERSION: Final = "2.4.6"

EXPERIMENTS: Final = ("M1", "M2")
REPETITIONS: Final = 5
ORDER_SEED: Final = 20261002

M1_RESOLUTIONS: Final = ((346, 260), (1280, 720))
M1_WARMUP: Final = 50
M1_PUBLICATIONS: Final = 2_000
M2_RESOLUTIONS: Final = ((346, 260), (640, 480), (1280, 720))
M2_CONDITIONS: Final = ((10_000, 0.0), (100_000, 0.0), (100_000, 16.0))
"""(events per call, publication interval in ms)."""

REGISTRATION_TIMEOUT_S: Final = 10.0
SHUTDOWN_TIMEOUT_S: Final = 10.0


@dataclasses.dataclass(frozen=True, slots=True)
class Spec:
    """One cell of one experiment on one runtime under one label."""

    experiment: str
    runtime: str
    width: int
    height: int
    batch_size: int
    interval_ms: float
    label: str

    @property
    def cell(self) -> str:
        return f"{self.experiment} {self.width}x{self.height} {self.batch_size} @ {self.interval_ms:g} ms"

    @property
    def id(self) -> str:
        label = self.label.replace("'", "p")
        return f"{self.experiment}_{self.runtime}_{self.width}x{self.height}_{self.batch_size}_{self.interval_ms:g}_{label}"

    @property
    def build(self) -> str:
        return LABELS[self.label][0]

    @property
    def waiters(self) -> int:
        return LABELS[self.label][1]

    def to_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Spec:
        return cls(**record)


def specs() -> list[Spec]:
    """Every preregistered (experiment, runtime, cell, label), in a fixed order."""
    out = []
    for runtime in RUNTIMES:
        for label in LABEL_ORDER:
            for width, height in M1_RESOLUTIONS:
                out.append(Spec("M1", runtime, width, height, 1, 0.0, label))
            for width, height in M2_RESOLUTIONS:
                for batch_size, interval_ms in M2_CONDITIONS:
                    out.append(Spec("M2", runtime, width, height, batch_size, interval_ms, label))
    return out


def pass_order(pass_index: int) -> list[Spec]:
    """Pass *pass_index* (1 to ``REPETITIONS``): every spec once, in a seeded random order."""
    if not 1 <= pass_index <= REPETITIONS:
        raise ValueError(f"pass must be 1..{REPETITIONS}, got {pass_index}")
    order = specs()
    random.Random(ORDER_SEED * 100 + pass_index).shuffle(order)
    return order


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- in the worker


class WaiterPool:
    """*count* threads, each looping ``s = engine.wait_for_newer(seq); seq = s.meta.sequence``
    with no work in between and no timeout, counting its returns."""

    def __init__(self, engine: Any, count: int) -> None:
        self._engine = engine
        self.count = count
        self.returns = [0] * count
        self.errors: list[str] = []
        self._stop = False
        self._threads = [threading.Thread(target=self._loop, args=(i,), name=f"waiter-{i}", daemon=True)
                         for i in range(count)]

    def _loop(self, i: int) -> None:
        sequence = None
        try:
            while not self._stop:
                snapshot = self._engine.wait_for_newer(sequence)
                sequence = snapshot.meta.sequence
                self.returns[i] += 1
        except BaseException:  # noqa: BLE001 - reported in the run record
            self.errors.append(traceback.format_exc())

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def registered(self) -> int:
        return len(self._engine._waiters) if self.count else 0

    def until_registered(self) -> None:
        """Spin until every waiter is registered. Never inside a timed call."""
        if not self.count:
            return
        deadline = time.monotonic() + REGISTRATION_TIMEOUT_S
        while len(self._engine._waiters) < self.count:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.count} waiters not registered after {REGISTRATION_TIMEOUT_S} s")
            time.sleep(0)

    def close(self, publish: Callable[[], None]) -> list[str]:
        """Stop the loops: set the flag, publish once to wake the registered, join. Returns
        the names of threads still alive after ``SHUTDOWN_TIMEOUT_S``."""
        if not self.count:
            return []
        self._stop = True
        publish()
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return [thread.name for thread in self._threads if thread.is_alive()]


class _FarFuture:
    """Stands in for ``time`` inside ``frames2py._engine`` for the shutdown publication, so
    that it publishes whatever the cell's interval."""

    @staticmethod
    def monotonic_ns() -> int:
        return 2**62


def _publish_now(engine_module: Any, engine: Any, batch: Any) -> None:
    engine_module.time = _FarFuture()
    try:
        engine.ingest(batch)
    finally:
        engine_module.time = time


def _rusage() -> dict[str, int] | None:
    from benchmarks import rusage

    sample = rusage.sample()
    if sample is None:
        return None
    keep = ("user_time", "system_time", "user_ptime", "system_ptime", "cycles", "instructions",
            "pcycles", "pinstructions", "runnable_time")
    return {k: sample[k] for k in keep}


def _percentiles(call_ns: Sequence[int]) -> dict[str, int]:
    from benchmarks.measure import nearest_rank

    return {f"p{q}": nearest_rank(call_ns, q) for q in (50, 95, 99)}


def measure_m2(spec: Spec) -> dict[str, Any]:
    """The gate's Engine-level method (``v1-engine``) on one cell, with the label's waiters."""
    import frames2py._engine as engine_module
    from benchmarks.matrix import Cell, default_seed
    from benchmarks.measure import DEFAULT_TIMED_CALLS, time_calls
    from benchmarks.targets.v1 import V1EngineTarget

    size = (spec.width, spec.height)
    cell = Cell("event_count", size, spec.batch_size, spec.interval_ms, "uniform",
                default_seed(size, spec.batch_size, "uniform"))
    target = V1EngineTarget()
    warmup = 1
    timed = target.timed_calls(cell, DEFAULT_TIMED_CALLS[spec.batch_size], warmup)
    batches = cell.workload().batches(warmup + timed + 1)
    measured, final = batches[:-1], batches[-1]
    prepared = target.prepare(cell, measured)
    engine = prepared.call.__self__  # type: ignore[attr-defined]
    pool = WaiterPool(engine, spec.waiters)
    pool.start()
    pool.until_registered()
    registered_at_start = pool.registered()
    before = _rusage()
    span: list[int] = []
    call_ns, counters = time_calls(prepared.call, measured, warmup, timed, prepared.counters,
                                   prepared.before_call, prepared.after_call, span)
    after = _rusage()
    returns = sum(pool.returns)
    checks = dict(prepared.finish()) if prepared.finish is not None else {"valid": True}
    alive = pool.close(lambda: _publish_now(engine_module, engine, final))
    return {
        "cell": cell.to_record(), "warmup_calls": warmup, "timed_calls": timed, "call_ns": call_ns,
        "timed_span_ns": span[0], "counters": counters, "checks": checks,
        "events_per_s": spec.batch_size * len(call_ns) / (sum(call_ns) / 1e9),
        **_percentiles(call_ns),
        "waiters": spec.waiters, "registered_at_start": registered_at_start,
        "waiter_returns_measured_calls": returns, "waiter_errors": pool.errors, "waiters_alive": alive,
        "rusage_before": before, "rusage_after": after,
    }


def measure_m1(spec: Spec) -> dict[str, Any]:
    """``ingest()`` of one event at interval 0, every call a publication, each with exactly
    the label's waiters registered when it starts."""
    import numpy as np

    import frames2py
    import frames2py._engine as engine_module
    from benchmarks.matrix import Cell, default_seed
    from benchmarks.measure import time_calls

    size = (spec.width, spec.height)
    cell = Cell("event_count", size, 1, 0.0, "uniform", default_seed(size, 1, "uniform"))
    batches = cell.workload().batches(M1_WARMUP + M1_PUBLICATIONS + 1)
    measured, final = batches[:-1], batches[-1]
    engine = frames2py.Engine(size, frames2py.EventCount(), snapshot_interval_ms=0.0)
    pool = WaiterPool(engine, spec.waiters)
    pool.start()
    short: list[int] = []

    def before_call() -> None:
        pool.until_registered()
        if pool.registered() != spec.waiters:
            short.append(pool.registered())

    before = _rusage()
    span: list[int] = []
    call_ns, _ = time_calls(engine.ingest, measured, M1_WARMUP, M1_PUBLICATIONS, None, before_call, None, span)
    after = _rusage()
    pool.until_registered()  # a waiter counts its return before it registers again
    returns = sum(pool.returns)
    stats = engine.stats
    snapshot = engine.snapshot()
    failures = []
    if stats.snapshots_published != len(measured):
        failures.append(f"snapshots_published {stats.snapshots_published}, expected {len(measured)}")
    if stats.events_ingested != len(measured) or stats.events_out_of_bounds != 0:
        failures.append(f"events_ingested {stats.events_ingested}, out of bounds {stats.events_out_of_bounds}")
    if snapshot is None:
        failures.append("nothing published")
    else:
        last = measured[-1]
        expected = np.zeros((spec.height, spec.width), dtype=np.uint32)
        expected[int(last["y"][0]), int(last["x"][0])] = 1
        if not np.array_equal(snapshot.frame, expected) or snapshot.meta.watermark != int(last["t"][0]):
            failures.append("the last snapshot is not the last call's window")
    if short:
        failures.append(f"{len(short)} calls started with other than {spec.waiters} waiters registered")
    if spec.waiters and returns != spec.waiters * len(measured):
        failures.append(f"{returns} waiter returns, expected {spec.waiters * len(measured)}")
    del snapshot
    alive = pool.close(lambda: _publish_now(engine_module, engine, final))
    return {
        "cell": cell.to_record(), "warmup_calls": M1_WARMUP, "timed_calls": M1_PUBLICATIONS, "call_ns": call_ns,
        "timed_span_ns": span[0], "checks": {"valid": not failures, "failures": failures},
        "median_ns": statistics.median(call_ns), **_percentiles(call_ns),
        "waiters": spec.waiters, "waiter_returns_measured_calls": returns,
        "waiter_errors": pool.errors, "waiters_alive": alive,
        "rusage_before": before, "rusage_after": after,
    }


def _refusals(spec: Spec) -> list[str]:
    import numpy as np

    import frames2py

    problems = []
    version = sys.version.split()[0]
    if version != RUNTIME_VERSIONS[spec.runtime]:
        problems.append(f"Python {version}, expected {RUNTIME_VERSIONS[spec.runtime]}")
    if np.__version__ != NUMPY_VERSION:
        problems.append(f"NumPy {np.__version__}, expected {NUMPY_VERSION}")
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    if gil == FREE_THREADED[spec.runtime]:
        problems.append(f"GIL enabled: {gil}")
    src = (BUILD_DIRS[spec.build] / "src").resolve()
    if Path(frames2py.__file__).resolve().parent.parent != src:
        problems.append(f"frames2py from {frames2py.__file__}, expected the {spec.build} build at {src}")
    if hasattr(frames2py.Engine, "wait_for_newer") != (spec.build == "feature"):
        problems.append(f"the {spec.build} build {'lacks' if spec.build == 'feature' else 'has'} wait_for_newer")
    return problems


def run(request: dict[str, Any]) -> dict[str, Any]:
    """One run: checks, then the measurement under the power guard. Writes and returns the record."""
    import frames2py

    from benchmarks import environment, power

    spec = Spec.from_record(request["spec"])
    record: dict[str, Any] = {"run_id": request["run_id"], "request": request, "started_at": time.time()}
    out = Path(request["out_dir"]) / f"{request['run_id']}.json"
    try:
        record.update(runtime=environment.runtime(), frames2py_file=frames2py.__file__,
                      preregistration_sha256=sha256(PREREGISTRATION),
                      switch_interval=sys.getswitchinterval(), gc_thresholds=gc.get_threshold(),
                      threads_before=threading.active_count())
        refused = _refusals(spec)
        if refused:
            record["refused"] = refused
        else:
            try:
                with power.hold_awake(f"frames2py wait_for_newer {request['run_id']}") as awake:
                    measure = measure_m1 if spec.experiment == "M1" else measure_m2
                    record["result"] = measure(spec)
                record["power"] = awake
            except power.PowerStateError as error:
                record["power_refused"] = str(error)
        record["threads_after"] = threading.active_count()
        record["gil_enabled_at_end"] = getattr(sys, "_is_gil_enabled", lambda: True)()
    except BaseException:  # noqa: BLE001 - classified by the driver
        record["fatal"] = traceback.format_exc()
    record["ended_at"] = time.time()
    out.write_text(json.dumps(record, default=str) + "\n")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["worker"]:
        print("usage: python -m benchmarks.wait worker < request.json", file=sys.stderr)
        return 2
    record = run(json.load(sys.stdin))
    return 0 if "fatal" not in record else 1


if __name__ == "__main__":
    sys.exit(main())
