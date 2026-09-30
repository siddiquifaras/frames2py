"""The observation-architecture study's harness (``benchmarks/observation_preregistration.md``).

One run is one child process and one cell: an arm, a consumer workload, a consumer count and
a runtime. The child runs the common producer loop (preregistration 6.2) on its own thread,
the arm's consumers on theirs, and the monitor (6.8) on its main thread. It writes the raw
per-batch, per-observation and monitor logs to ``<run id>.npz`` and a JSON record next to
them, and checks the run's integrity accounting (21.4) after the window. Metrics are computed
later, from those files, by ``benchmarks.observation_analysis``.

Arms are built from consumer objects with a non-blocking ``tick()``, so the same code runs
threaded in a measured run and in lockstep in validation (V1, V6).

Every parameter is the preregistration's. ``Condition`` exists so that validation and tests
can run the same code at a smaller size; the driver refuses a campaign run at anything other
than the preregistered condition.
"""

from __future__ import annotations

import dataclasses
import enum
import gc
import hashlib
import json
import math
import os
import random
import sys
import sysconfig
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

import frames2py
from frames2py import Accumulator, Engine, EventCount, SnapshotMeta
from frames2py.publish import Snapshot

from benchmarks import rusage
from benchmarks.matrix import default_seed
from benchmarks.workloads import EventArray, Workload

REPO: Final = Path(__file__).resolve().parent.parent
PREREGISTRATION: Final = Path(__file__).with_name("observation_preregistration.md")
PREREGISTRATION_COMMIT: Final = "cdb1ee3d8a234d7c159c1603a611c10863f932cc"
"""The commit that added the approved protocol; an ancestor of every measured commit."""
STUDY_DIR: Final = REPO / "scratch" / "observation_study"
ENVIRONMENTS: Final = {
    "A": STUDY_DIR / "envs" / "py311" / "bin" / "python",
    "B": STUDY_DIR / "envs" / "py314t" / "bin" / "python",
}
RUNTIME_VERSIONS: Final = {"A": "3.11.14", "B": "3.14.2"}
PACKAGE_VERSIONS: Final = {"frames2py": "1.0.0", "numpy": "2.4.6", "h5py": "3.16.0", "hdf5plugin": "7.1.0"}

P1_ARMS: Final = ("A", "B", "C", "E", "F", "G", "H", "H'", "RB")
"""The fixed arm order of every table (16.5)."""
P2_ARMS: Final = ("EP-H", "EP-QB", "EP-QE")
WORKLOADS: Final = ("W1", "W3", "W5")
CONSUMER_COUNTS: Final = (1, 4)
REFERENCE_ARMS: Final = ("A", "F", "G", "H", "H'")
"""Arms with an N = 0 cell."""
RUNTIMES: Final = ("A", "B")
EXPERIMENT_INDEX: Final = {"P1": 1, "P2": 2}
ORDER_SEED: Final = 20261001
REPETITIONS: Final = 5
QUEUE_ARMS: Final = frozenset({"B", "C", "E", "RB"})
POLL_ARMS: Final = frozenset({"F", "G", "H", "H'"})
ENGINE_ARMS: Final = frozenset({"H", "H'", "EP-H", "EP-QB", "EP-QE"})

W3_SLEEP_S: Final = 0.030
W5_TARGET_S: Final = 0.250
W5_MASK: Final = 0x7FFFFFFF
MEMORY_CEILING_BYTES: Final = 4 * 2**30
SHUTDOWN_GRACE_S: Final = 10.0
DRAIN_CAP_S: Final = 120.0
MONITOR_PERIOD_NS: Final = 100_000_000
QUEUE_TIMEOUT_S: Final = 0.1

POOL_SHA256: Final = "d002851ebd9342a2bd76551b75404b67292223e0be91b1a94ed0fde39c6a5e75"
"""SHA-256 of the preregistered condition's pool bytes (11.1), computed on both study
environments before any run; every run's pool must match it."""


# ---------------------------------------------------------------- the condition


@dataclasses.dataclass(frozen=True, slots=True)
class Condition:
    """Input, cadence and timing. The defaults are the preregistered condition (11.3)."""

    sensor_size: tuple[int, int] = (1280, 720)
    batch_size: int = 100_000
    rate_hz: int = 20_000_000
    interval_ms: float = 16.0
    warmup_s: float = 5.0
    window_s: float = 10.0
    pool_events: int = 6_400_000
    k_ff: int = 4

    @property
    def pool_batches(self) -> int:
        return self.pool_events // self.batch_size

    @property
    def batch_period_ns(self) -> float:
        return self.batch_size * 1e9 / self.rate_hz

    @property
    def k_raw(self) -> int:
        """``ceil(64 ms / batch period)`` (6.5.1)."""
        return math.ceil(64e6 / self.batch_period_ns - 1e-9)

    @property
    def interval_ns(self) -> float:
        return self.interval_ms * 1e6

    @property
    def stream_ns(self) -> int:
        return round((self.warmup_s + self.window_s) * 1e9)

    @property
    def warmup_ns(self) -> int:
        return round(self.warmup_s * 1e9)

    def to_record(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["sensor_size"] = list(self.sensor_size)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Condition:
        values = dict(record)
        values["sensor_size"] = tuple(values["sensor_size"])
        return cls(**values)


PREREGISTERED: Final = Condition()


# ---------------------------------------------------------------- cells and pass orders


@dataclasses.dataclass(frozen=True, slots=True)
class Cell:
    experiment: str
    arm: str
    workload: str
    n: int
    runtime: str

    @property
    def id(self) -> str:
        return f"{self.experiment}_{arm_slug(self.arm)}_{self.workload}_N{self.n}_{self.runtime}"

    def to_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Cell:
        return cls(record["experiment"], record["arm"], record["workload"], int(record["n"]), record["runtime"])


def arm_slug(arm: str) -> str:
    return arm.replace("'", "p")


def p1_cells() -> list[Cell]:
    """The 118 P1 cells, in listing order (11.4)."""
    cells = [Cell("P1", arm, w, n, rt) for arm in P1_ARMS for w in WORKLOADS for n in CONSUMER_COUNTS
             for rt in RUNTIMES]
    cells += [Cell("P1", arm, "none", 0, rt) for arm in REFERENCE_ARMS for rt in RUNTIMES]
    return cells


def p2_cells() -> list[Cell]:
    """The 12 P2 cells; ``n`` is N_SO and the workload of its consumer is W1."""
    return [Cell("P2", arm, "W1" if n else "none", n, rt) for arm in P2_ARMS for n in (0, 1) for rt in RUNTIMES]


def cells(experiment: str) -> list[Cell]:
    if experiment == "P1":
        return p1_cells()
    if experiment == "P2":
        return p2_cells()
    raise ValueError(f"unknown experiment {experiment!r}")


def pass_order(experiment: str, pass_index: int) -> list[Cell]:
    """One pass's run order: the listing order shuffled with the preregistered seed (11.4)."""
    order = cells(experiment)
    random.Random(ORDER_SEED + 100 * EXPERIMENT_INDEX[experiment] + pass_index).shuffle(order)
    return order


# ---------------------------------------------------------------- the source (11.1)


class Source:
    """The synthetic stream: a pool of distinct batches, materialised afresh for every batch k."""

    def __init__(self, condition: Condition) -> None:
        c = condition
        if c.pool_events % c.batch_size:
            raise ValueError("the pool must hold a whole number of batches")
        self.condition = c
        self.pool_batches = c.pool_batches
        cycle = self.pool_batches * c.batch_size * 1_000_000
        if cycle % c.rate_hz:
            raise ValueError("the per-cycle time offset must be a whole number of microseconds")
        self.cycle_offset_us = cycle // c.rate_hz
        workload = Workload("uniform", c.sensor_size, c.batch_size,
                            seed=default_seed(c.sensor_size, c.batch_size, "uniform"), event_rate_hz=c.rate_hz)
        self.pool: list[EventArray] = workload.batches(self.pool_batches)
        for batch in self.pool:
            batch.flags.writeable = False
        self._pool_max = [int(b["t"].max()) for b in self.pool]
        self.t0 = int(self.pool[0]["t"].min())

    def digest(self) -> str:
        h = hashlib.sha256()
        for batch in self.pool:
            h.update(batch.tobytes())
        return h.hexdigest()

    def offset_rule(self) -> str:
        return f"t += (k div {self.pool_batches}) * {self.cycle_offset_us} us"

    def max_t(self, k: int) -> int:
        """``m_k``, the largest timestamp in batch k."""
        return self._pool_max[k % self.pool_batches] + (k // self.pool_batches) * self.cycle_offset_us

    def materialise(self, k: int) -> EventArray:
        batch = self.pool[k % self.pool_batches].copy()
        batch["t"] += np.uint64((k // self.pool_batches) * self.cycle_offset_us)
        return batch

    def schedule(self) -> tuple[list[int], list[int]]:
        """``(offsets, m)``: ``A_k - T_start`` in ns and ``m_k``, for every batch whose scheduled
        availability lies within the stream (6.2)."""
        offsets: list[int] = []
        maxima: list[int] = []
        running = -1
        k = 0
        while True:
            m = self.max_t(k)
            running = max(running, m)
            offset = 1000 * (running - self.t0 + 1)
            if offset > self.condition.stream_ns:
                return offsets, maxima
            offsets.append(offset)
            maxima.append(m)
            k += 1


# ---------------------------------------------------------------- queues (6.5.1)


class Policy(enum.Enum):
    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    UNBOUNDED = "unbounded"


T = TypeVar("T")


class PolicyQueue(Generic[T]):
    """A FIFO of ``(key, item)`` pairs with one of four full-queue policies.

    The structure of ``queue.Queue``: one lock, two conditions on it and a deque. ``key``
    identifies an item in the logs (a publication's sequence number or a batch's k).
    """

    def __init__(self, capacity: int | None, policy: Policy, stop: threading.Event) -> None:
        if (capacity is None) != (policy is Policy.UNBOUNDED):
            raise ValueError("an UNBOUNDED queue has no capacity; every other policy needs one")
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.policy = policy
        self._stop = stop
        self._items: deque[tuple[int, T]] = deque()
        lock = threading.Lock()
        self._not_empty = threading.Condition(lock)
        self._not_full = threading.Condition(lock)
        self.offered = 0
        self.enqueued = 0
        self.dequeued = 0
        self.abandoned: list[int] = []
        """Keys whose blocked put was given up because ``stop`` was set."""
        self.evicted: list[tuple[int, int]] = []
        """DROP_OLDEST: ``(evicted key, key whose put evicted it)``."""
        self.rejected: list[int] = []
        """DROP_NEWEST: keys refused because the queue was full."""

    def put(self, key: int, item: T) -> bool:
        """Offer an item; ``True`` if it was enqueued."""
        with self._not_full:
            self.offered += 1
            if self.capacity is not None and len(self._items) >= self.capacity:
                if self.policy is Policy.BLOCK:
                    while len(self._items) >= self.capacity:
                        if self._stop.is_set():
                            self.abandoned.append(key)
                            return False
                        self._not_full.wait(QUEUE_TIMEOUT_S)
                elif self.policy is Policy.DROP_OLDEST:
                    old_key, _ = self._items.popleft()
                    self.evicted.append((old_key, key))
                else:
                    self.rejected.append(key)
                    return False
            self._items.append((key, item))
            self.enqueued += 1
            self._not_empty.notify()
            return True

    def get(self, timeout: float) -> tuple[int, T] | None:
        """The oldest item, waiting up to *timeout* seconds once; ``None`` if still empty."""
        with self._not_empty:
            if not self._items:
                self._not_empty.wait(timeout)
                if not self._items:
                    return None
            got = self._items.popleft()
            self.dequeued += 1
            self._not_full.notify()
            return got

    def get_nowait(self) -> tuple[int, T] | None:
        with self._not_empty:
            if not self._items:
                return None
            got = self._items.popleft()
            self.dequeued += 1
            self._not_full.notify()
            return got

    def remaining(self) -> list[int]:
        """Keys still queued. For after the run, never during it (10.2: no depth polling)."""
        with self._not_empty:
            return [key for key, _ in self._items]

    def log(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value, "capacity": self.capacity, "offered": self.offered,
            "enqueued": self.enqueued, "dequeued": self.dequeued, "abandoned": list(self.abandoned),
            "evicted": [list(e) for e in self.evicted], "rejected": list(self.rejected),
            "remaining": self.remaining(),
        }


# ---------------------------------------------------------------- workloads (8)


Work = Callable[[NDArray[Any], int, int, Snapshot | None], None]
"""``work(frame, watermark, sequence, snapshot)``; *snapshot* is the Engine's, for arm H."""


def lcg(seed: int, k: int) -> int:
    """W5's pure-Python loop: *k* steps of the LCG from ``seed & 0x7FFFFFFF``."""
    a = seed & W5_MASK
    for _ in range(k):
        a = (a * 1103515245 + 12345) & W5_MASK
    return a


def make_work(workload: str, k5: int | None) -> Work:
    from frames2py.viewer import render

    if workload == "W1":
        def w1(frame: NDArray[Any], watermark: int, sequence: int, snapshot: Snapshot | None) -> None:
            render(snapshot if snapshot is not None else Snapshot(frame, SnapshotMeta(watermark, sequence)),
                   scale=None)
        return w1
    if workload == "W3":
        def w3(frame: NDArray[Any], watermark: int, sequence: int, snapshot: Snapshot | None) -> None:
            time.sleep(W3_SLEEP_S)  # the caller holds the state's references meanwhile
        return w3
    if workload == "W5":
        if k5 is None or k5 < 1:
            raise ValueError("W5 needs the calibrated K5")
        steps = k5

        def w5(frame: NDArray[Any], watermark: int, sequence: int, snapshot: Snapshot | None) -> None:
            lcg(sequence, steps)
        return w5
    if workload == "none":
        def none(frame: NDArray[Any], watermark: int, sequence: int, snapshot: Snapshot | None) -> None:
            pass
        return none
    raise ValueError(f"unknown workload {workload!r}")


# ---------------------------------------------------------------- run context and logs


Clock = Callable[[], int]


class Context:
    """What every thread of one run shares. The window bounds are set once, before ``go``."""

    def __init__(self, condition: Condition, work: Work, *, full: bool = True,
                 cadence_clock: Clock = time.monotonic_ns) -> None:
        self.condition = condition
        self.work = work
        self.full = full
        self.cadence_clock = cadence_clock
        self.stop = threading.Event()
        self.go = threading.Event()
        self.producer_done = threading.Event()
        self.t_start = 0
        self.t0 = 0
        self.t1 = 0
        self.errors: list[dict[str, str]] = []
        self.memory_error = False

    def set_start(self, t_start: int) -> None:
        self.t_start = t_start
        self.t0 = t_start + self.condition.warmup_ns
        self.t1 = t_start + self.condition.stream_ns

    def fail(self, where: str, exc: BaseException) -> None:
        if isinstance(exc, MemoryError):
            self.memory_error = True
        self.errors.append({"thread": where, "type": type(exc).__name__,
                            "traceback": "".join(traceback.format_exception(exc))})
        self.stop.set()


class ConsumerLog:
    """One consumer's records: observations, loop samples and, in the raw loop, blocks."""

    def __init__(self, index: int, full: bool) -> None:
        self.index = index
        self.full = full
        self.o: list[int] = []
        self.w: list[int] = []
        self.co: list[int] = []
        self.cw: list[int] = []
        self.seq: list[int] = []
        self.wm: list[int] = []
        self.observations = 0
        self.first: tuple[int, int] | None = None
        self.last: tuple[int, int] | None = None
        self.polls = 0
        self.keys: list[int] = []
        """Keys in the order this consumer took them from its queue."""
        self.blocks: list[tuple[int, int, int, int, int, int]] = []
        """Raw loop: ``(start, start_cpu, end, end_cpu, batches, events)`` per block."""

    def mark(self, now: int, ctx: Context) -> None:
        if not self.full:
            return
        if self.first is None and now >= ctx.t0:
            self.first = (now, time.thread_time_ns())
        if now < ctx.t1:
            self.last = (now, time.thread_time_ns())

    def observe(self, ctx: Context, observed_at: int, frame: NDArray[Any], watermark: int, sequence: int,
                snapshot: Snapshot | None) -> None:
        """Work on one observed state, whose instant O the caller took."""
        if self.full:
            co = time.thread_time_ns()
            ctx.work(frame, watermark, sequence, snapshot)
            self.w.append(time.perf_counter_ns())
            self.cw.append(time.thread_time_ns())
            self.o.append(observed_at)
            self.co.append(co)
            self.seq.append(sequence)
            self.wm.append(watermark)
        else:
            ctx.work(frame, watermark, sequence, snapshot)
        self.observations += 1

    def arrays(self) -> dict[str, NDArray[Any]]:
        first = self.first or (0, 0)
        last = self.last or (0, 0)
        return {
            "o": np.array(self.o, dtype=np.int64), "w": np.array(self.w, dtype=np.int64),
            "co": np.array(self.co, dtype=np.int64), "cw": np.array(self.cw, dtype=np.int64),
            "seq": np.array(self.seq, dtype=np.int64), "wm": np.array(self.wm, dtype=np.int64),
            "samples": np.array([first[0], first[1], last[0], last[1], int(self.first is not None),
                                 int(self.last is not None), self.polls, self.observations], dtype=np.int64),
            "keys": np.array(self.keys, dtype=np.int64),
            "blocks": np.array(self.blocks, dtype=np.int64).reshape(-1, 6),
        }


class Cadence:
    """The publication rule of 6.3, read after each accumulation call."""

    def __init__(self, interval_ns: float, clock: Clock) -> None:
        self._interval = interval_ns
        self._clock = clock
        self._last: int | None = None

    def due(self) -> bool:
        now = self._clock()
        if self._last is None or now - self._last >= self._interval:
            self._last = now
            return True
        return False


Item = tuple[NDArray[Any], int, int]
"""A finished-frame item: ``(frame, watermark, sequence)``."""


class Publisher:
    """``publish_state()`` of 6.4 over one Accumulator."""

    def __init__(self, acc: Accumulator) -> None:
        self.acc = acc
        self.sequence = 0

    def publish(self) -> Item:
        frame = self.acc.read()
        watermark = self.acc.watermark
        if watermark is None:
            raise RuntimeError("a publication with no in-bounds event: every window here is non-empty")
        self.acc.reset()
        self.sequence += 1
        return frame, watermark, self.sequence


# ---------------------------------------------------------------- consumers


class Consumer:
    """One consumer. ``tick(timeout)`` is one loop iteration; ``None`` never waits."""

    kind = "poll"

    def __init__(self, ctx: Context, log: ConsumerLog) -> None:
        self.ctx = ctx
        self.log = log

    def tick(self, timeout: float | None) -> bool:
        raise NotImplementedError


class LatestConsumer(Consumer):
    """Poll-loop consumer of F, G, H and H′: ``read(last)`` returns a new state or ``None``."""

    kind = "poll"

    def __init__(self, ctx: Context, log: ConsumerLog,
                 read: Callable[[int | None], tuple[NDArray[Any], int, int, Snapshot | None] | None]) -> None:
        super().__init__(ctx, log)
        self._read = read
        self._last: int | None = None

    def tick(self, timeout: float | None) -> bool:
        state = self._read(self._last)
        if state is None:
            return False
        observed_at = time.perf_counter_ns()
        frame, watermark, sequence, snapshot = state
        self.log.observe(self.ctx, observed_at, frame, watermark, sequence, snapshot)
        self._last = sequence
        return True


class QueueConsumer(Consumer):
    """Queue-loop consumer of B, C and E."""

    kind = "queue"

    def __init__(self, ctx: Context, log: ConsumerLog, queue: PolicyQueue[Item]) -> None:
        super().__init__(ctx, log)
        self.queue = queue

    def tick(self, timeout: float | None) -> bool:
        got = self.queue.get_nowait() if timeout is None else self.queue.get(timeout)
        if got is None:
            return False
        observed_at = time.perf_counter_ns()
        sequence, (frame, watermark, _) = got
        if self.log.full:
            self.log.keys.append(sequence)
        self.log.observe(self.ctx, observed_at, frame, watermark, sequence, None)
        return True


class RawConsumer(Consumer):
    """Raw-loop consumer of RB: its own Accumulator, drained before it observes."""

    kind = "queue"

    def __init__(self, ctx: Context, log: ConsumerLog, queue: PolicyQueue[EventArray]) -> None:
        super().__init__(ctx, log)
        self.queue = queue
        self.acc = Accumulator(ctx.condition.sensor_size, EventCount())
        self.publisher = Publisher(self.acc)
        self.cadence = Cadence(ctx.condition.interval_ns, ctx.cadence_clock)
        self.events = 0
        self.last_key: int | None = None
        self.last_published_watermark: int | None = None

    def tick(self, timeout: float | None) -> bool:
        got = self.queue.get_nowait() if timeout is None else self.queue.get(timeout)
        if got is None:
            return False
        full = self.log.full
        start, start_cpu = (time.perf_counter_ns(), time.thread_time_ns()) if full else (0, 0)
        batches = events = 0
        while got is not None:
            key, batch = got
            self.acc.accumulate(batch)
            if full:
                self.log.keys.append(key)
            self.last_key = key
            batches += 1
            events += len(batch)
            got = self.queue.get_nowait()
        self.events += events
        if full:
            self.log.blocks.append((start, start_cpu, time.perf_counter_ns(), time.thread_time_ns(), batches, events))
        if self.cadence.due():
            frame = self.acc.read()
            observed_at = time.perf_counter_ns()
            watermark = self.acc.watermark
            if watermark is None:
                raise RuntimeError("a raw consumer published with no in-bounds event")
            self.acc.reset()
            self.publisher.sequence += 1
            self.last_published_watermark = watermark
            self.log.observe(self.ctx, observed_at, frame, watermark, self.publisher.sequence, None)
        return True


class RecorderConsumer(Consumer):
    """EP-QB and EP-QE's recorder thread: ``rec.write(b)`` for every queued batch."""

    kind = "recorder"

    def __init__(self, ctx: Context, log: ConsumerLog, queue: PolicyQueue[EventArray], recorder: Any) -> None:
        super().__init__(ctx, log)
        self.queue = queue
        self.recorder = recorder

    def tick(self, timeout: float | None) -> bool:
        got = self.queue.get_nowait() if timeout is None else self.queue.get(timeout)
        if got is None:
            return False
        key, batch = got
        self.recorder.write(batch)
        self.log.keys.append(key)
        self.log.o.append(time.perf_counter_ns())
        return True


# ---------------------------------------------------------------- arms (6.4)


class Arm:
    """An arm's producer step, publication counter and consumers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.consumers: list[Consumer] = []
        self.inline: list[ConsumerLog] = []
        self.queues: list[PolicyQueue[Any]] = []
        self.engine: Engine | None = None
        self.recorder: Any = None
        self.written: list[int] = []
        """EP-H: the batches ``rec.write()`` completed, in order."""
        self.ingested_events = 0

    def step(self, k: int, batch: EventArray) -> None:
        raise NotImplementedError

    def published_sequence(self) -> int:
        raise NotImplementedError


def build_arm(name: str, n: int, ctx: Context, *, recorder: Any = None,
              engine_factory: Callable[[], Engine] | None = None) -> Arm:
    """The arm *name* with *n* consumers (N_SO for the EP arms)."""
    c = ctx.condition
    full = ctx.full
    logs = [ConsumerLog(i, full) for i in range(n)]
    make_engine = engine_factory or (lambda: Engine(c.sensor_size, EventCount(), snapshot_interval_ms=c.interval_ms))

    if name in ("A", "B", "C", "E", "F", "G"):
        acc = Accumulator(c.sensor_size, EventCount())
        publisher = Publisher(acc)
        cadence = Cadence(c.interval_ns, ctx.cadence_clock)
        arm = _BaselineArm(name, acc, publisher, cadence)
        if name == "A":
            arm.inline = logs
        elif name in ("B", "C", "E"):
            policy = {"B": Policy.BLOCK, "C": Policy.DROP_OLDEST, "E": Policy.UNBOUNDED}[name]
            capacity = None if policy is Policy.UNBOUNDED else c.k_ff
            for log in logs:
                q: PolicyQueue[Item] = PolicyQueue(capacity, policy, ctx.stop)
                arm.queues.append(q)
                arm.consumers.append(QueueConsumer(ctx, log, q))
        elif name == "F":
            height, width = c.sensor_size[1], c.sensor_size[0]
            arm.shared = np.zeros((height, width), dtype=np.uint32)
            for log in logs:
                arm.consumers.append(LatestConsumer(ctx, log, arm.f_reader(np.zeros_like(arm.shared))))
        else:
            for log in logs:
                arm.consumers.append(LatestConsumer(ctx, log, arm.g_read))
        arm.ctx = ctx
        return arm
    if name in ("H", "H'"):
        engine_arm = _EngineArm(name, make_engine())
        for log in logs:
            engine_arm.consumers.append(LatestConsumer(ctx, log, engine_arm.snapshot_read))
        return engine_arm
    if name == "RB":
        raw = _RawArm(name)
        for log in logs:
            rq: PolicyQueue[EventArray] = PolicyQueue(c.k_raw, Policy.BLOCK, ctx.stop)
            raw.queues.append(rq)
            raw.consumers.append(RawConsumer(ctx, log, rq))
        return raw
    if name in P2_ARMS:
        if recorder is None:
            raise ValueError("an EP arm needs a recorder")
        ep = _PreservationArm(name, make_engine(), recorder)
        if name != "EP-H":
            policy = Policy.BLOCK if name == "EP-QB" else Policy.UNBOUNDED
            eq: PolicyQueue[EventArray] = PolicyQueue(c.k_raw if policy is Policy.BLOCK else None, policy, ctx.stop)
            ep.queues.append(eq)
            ep.recorder_consumer = RecorderConsumer(ctx, ConsumerLog(-1, full), eq, recorder)
        for log in logs:
            ep.consumers.append(LatestConsumer(ctx, log, ep.snapshot_read))
        return ep
    raise ValueError(f"unknown arm {name!r}")


class _BaselineArm(Arm):
    """A, B, C, E, F and G: the public Accumulator and the harness's own publication."""

    def __init__(self, name: str, acc: Accumulator, publisher: Publisher, cadence: Cadence) -> None:
        super().__init__(name)
        self.acc = acc
        self.publisher = publisher
        self.cadence = cadence
        self.lock = threading.Lock()
        self.slot: Item | None = None
        self.shared: NDArray[Any] = np.zeros(0, dtype=np.uint32)
        self.meta: tuple[int, int] | None = None
        self.ctx: Context | None = None

    def step(self, k: int, batch: EventArray) -> None:
        self.acc.accumulate(batch)
        if not self.cadence.due():
            return
        name = self.name
        if name == "A":
            item = self.publisher.publish()
            frame, watermark, sequence = item
            assert self.ctx is not None
            for log in self.inline:
                log.observe(self.ctx, time.perf_counter_ns(), frame, watermark, sequence, None)
        elif name == "F":
            frame, watermark, sequence = self.publisher.publish()
            with self.lock:
                np.copyto(self.shared, frame)
                self.meta = (watermark, sequence)
        elif name == "G":
            item = self.publisher.publish()
            with self.lock:
                self.slot = item
        else:
            item = self.publisher.publish()
            for q in self.queues:
                q.put(item[2], item)

    def published_sequence(self) -> int:
        return self.publisher.sequence

    def g_read(self, last: int | None) -> tuple[NDArray[Any], int, int, Snapshot | None] | None:
        with self.lock:
            item = self.slot
        if item is None or item[2] == last:
            return None
        return item[0], item[1], item[2], None

    def f_reader(self, private: NDArray[Any]) -> Callable[[int | None], tuple[NDArray[Any], int, int, Snapshot | None] | None]:
        def read(last: int | None) -> tuple[NDArray[Any], int, int, Snapshot | None] | None:
            with self.lock:
                meta = self.meta
                if meta is None or meta[1] == last:
                    return None
                np.copyto(private, self.shared)
            return private, meta[0], meta[1], None
        return read


class _EngineArm(Arm):
    """H and H′: ``Engine.ingest()`` and ``Engine.snapshot()``, the public API only."""

    def __init__(self, name: str, engine: Engine) -> None:
        super().__init__(name)
        self.engine = engine
        self._ingest = engine.ingest
        self._snapshot = engine.snapshot

    def step(self, k: int, batch: EventArray) -> None:
        self._ingest(batch)
        self.ingested_events += len(batch)

    def published_sequence(self) -> int:
        s = self._snapshot()
        return 0 if s is None else s.meta.sequence

    def snapshot_read(self, last: int | None) -> tuple[NDArray[Any], int, int, Snapshot | None] | None:
        s = self._snapshot()
        if s is None or s.meta.sequence == last:
            return None
        watermark = s.meta.watermark
        assert watermark is not None
        return s.frame, watermark, s.meta.sequence, s


class _RawArm(Arm):
    """RB: the producer only puts each raw batch into every consumer's queue."""

    def step(self, k: int, batch: EventArray) -> None:
        for q in self.queues:
            q.put(k, batch)

    def published_sequence(self) -> int:
        return 0


class _PreservationArm(_EngineArm):
    """EP-H records on the producer thread; EP-QB and EP-QE hand batches to a recorder thread."""

    def __init__(self, name: str, engine: Engine, recorder: Any) -> None:
        super().__init__(name, engine)
        self.recorder = recorder
        self.recorder_consumer: RecorderConsumer | None = None

    def step(self, k: int, batch: EventArray) -> None:
        if self.name == "EP-H":
            self.recorder.write(batch)
            self.written.append(k)
        else:
            self.queues[0].put(k, batch)
        self._ingest(batch)
        self.ingested_events += len(batch)


# ---------------------------------------------------------------- the threaded run


class ProducerLog:
    """Per-batch records in preallocated storage (6.2)."""

    FIELDS: Final = ("q", "s", "e", "cq", "cs", "ce", "p")

    def __init__(self, size: int) -> None:
        self.size = size
        self.columns: dict[str, list[int]] = {f: [0] * size for f in self.FIELDS}
        self.count = 0

    def arrays(self) -> dict[str, NDArray[np.int64]]:
        return {f"producer_{f}": np.array(v[: self.count], dtype=np.int64) for f, v in self.columns.items()}


def _producer(ctx: Context, arm: Arm, source: Source, offsets: Sequence[int], log: ProducerLog,
              ready: threading.Semaphore, consumers: int) -> None:
    try:
        for _ in range(consumers):
            ready.acquire()
        t_start = time.perf_counter_ns()
        ctx.set_start(t_start)
        ctx.go.set()
        perf, cpu, sleep = time.perf_counter_ns, time.thread_time_ns, time.sleep
        stop = ctx.stop
        end = ctx.t1
        step, published = arm.step, arm.published_sequence
        materialise = source.materialise
        q, s, e, cq, cs, ce, p = (log.columns[f] for f in ProducerLog.FIELDS)
        full = ctx.full
        for k, offset in enumerate(offsets):
            due = t_start + offset
            if due > end or stop.is_set():
                break
            while (r := due - perf()) > 0:
                sleep(min(r, 1_000_000) / 1e9)
            if full:
                q[k] = perf()
                cq[k] = cpu()
                batch = materialise(k)
                s[k] = perf()
                cs[k] = cpu()
                step(k, batch)
                e[k] = perf()
                ce[k] = cpu()
                p[k] = published()
            else:
                batch = materialise(k)
                s[k] = perf()
                step(k, batch)
                e[k] = perf()
            log.count = k + 1
    except BaseException as exc:  # noqa: BLE001 - recorded, classified by the driver
        ctx.fail("producer", exc)
    finally:
        ctx.go.set()
        ctx.producer_done.set()


def _consumer_thread(ctx: Context, consumer: Consumer, ready: threading.Semaphore) -> None:
    try:
        ready.release()
        ctx.go.wait()
        log = consumer.log
        stop = ctx.stop
        perf = time.perf_counter_ns
        if consumer.kind == "poll":
            poll_ns = ctx.condition.interval_ns
            deadline: float = perf()
            t0, t1 = ctx.t0, ctx.t1
            while not stop.is_set():
                now = perf()
                log.mark(now, ctx)
                consumer.tick(None)
                if t0 <= now < t1:
                    log.polls += 1
                deadline += poll_ns
                now = perf()
                if deadline > now:
                    time.sleep((deadline - now) / 1e9)
                else:
                    deadline = now
        else:
            while not stop.is_set():
                log.mark(perf(), ctx)
                consumer.tick(QUEUE_TIMEOUT_S)
    except BaseException as exc:  # noqa: BLE001
        ctx.fail(f"consumer {consumer.log.index}", exc)


def _recorder_thread(ctx: Context, consumer: RecorderConsumer, result: dict[str, Any]) -> None:
    """Records until the producer has finished and the queue is empty, or the drain cap."""
    try:
        ctx.go.wait()
        queue = consumer.queue
        while True:
            if consumer.tick(QUEUE_TIMEOUT_S):
                continue
            if ctx.producer_done.is_set() and not queue.remaining():
                result["drained_at"] = time.perf_counter_ns()
                return
            if ctx.t1 and time.perf_counter_ns() > ctx.t1 + DRAIN_CAP_S * 1e9:
                result["drain_timeout"] = True
                return
    except BaseException as exc:  # noqa: BLE001
        ctx.fail("recorder", exc)


class Monitor:
    """The child's main thread (6.8): resource samples every 100 ms, the stop, the ceiling."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.times: list[int] = []
        self.samples: list[dict[str, int] | None] = []
        self.getrusage: list[tuple[float, float, int]] = []
        self.gc_t0: list[dict[str, int]] | None = None
        self.gc_t1: list[dict[str, int]] | None = None
        self.footprint_start: int | None = None
        self.memory_ceiling = False
        self.stopped_at: int | None = None

    def _sample(self) -> dict[str, int] | None:
        import resource

        now = time.perf_counter_ns()
        usage = rusage.sample()
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.times.append(now)
        self.samples.append(usage)
        self.getrusage.append((r.ru_utime, r.ru_stime, int(r.ru_maxrss)))
        return usage

    def run(self) -> None:
        ctx = self.ctx
        ctx.go.wait()
        if not ctx.t_start:
            return
        first = self._sample()
        self.footprint_start = None if first is None else first["phys_footprint"]
        deadline = ctx.t_start
        while True:
            deadline += MONITOR_PERIOD_NS
            now = time.perf_counter_ns()
            if deadline > now:
                time.sleep((deadline - now) / 1e9)
            else:
                deadline = now
            usage = self._sample()
            now = self.times[-1]
            if self.gc_t0 is None and now >= ctx.t0:
                self.gc_t0 = gc.get_stats()
            if now >= ctx.t1:
                self.gc_t1 = gc.get_stats()
                break
            if (usage is not None and self.footprint_start is not None
                    and usage["phys_footprint"] - self.footprint_start > MEMORY_CEILING_BYTES):
                self.memory_ceiling = True
                break
            if ctx.stop.is_set():
                break
        self.stopped_at = time.perf_counter_ns()
        ctx.stop.set()

    def arrays(self) -> dict[str, NDArray[Any]]:
        out: dict[str, NDArray[Any]] = {
            "monitor_t": np.array(self.times, dtype=np.int64),
            "monitor_getrusage": np.array(self.getrusage, dtype=np.float64).reshape(-1, 3),
        }
        if self.samples and all(s is not None for s in self.samples):
            for field in rusage.FIELDS:
                out[f"rusage_{field}"] = np.array([s[field] for s in self.samples if s is not None],
                                                  dtype=np.uint64)
        return out


# ---------------------------------------------------------------- the child


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for name in PACKAGE_VERSIONS:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def runtime_problems(request: dict[str, Any]) -> list[str]:
    """What makes this interpreter the wrong one for the request (12), after every import."""
    import importlib

    for module in ("h5py", "hdf5plugin", "frames2py.adapters.hdf5", "frames2py.recorder", "frames2py.viewer"):
        importlib.import_module(module)  # before the GIL check, as 12 requires
    problems = []
    runtime = request["runtime"]
    version = sys.version.split()[0]
    free_threaded = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    if version != RUNTIME_VERSIONS[runtime]:
        problems.append(f"python {version}, runtime {runtime} is {RUNTIME_VERSIONS[runtime]}")
    if runtime == "A" and (free_threaded or not gil):
        problems.append("runtime A must be a standard build with the GIL")
    if runtime == "B" and (not free_threaded or gil or os.environ.get("PYTHON_GIL") is not None):
        problems.append("runtime B must be free-threaded with the GIL disabled and PYTHON_GIL unset")
    versions = _versions()
    for name, wanted in PACKAGE_VERSIONS.items():
        if versions[name] != wanted:
            problems.append(f"{name} {versions[name]}, the study pins {wanted}")
    site = Path(sys.prefix).resolve()
    if not Path(frames2py.__file__).resolve().is_relative_to(site):
        problems.append(f"frames2py is imported from {frames2py.__file__}, not the study environment {site}")
    return problems


def _environment_record(request: dict[str, Any]) -> dict[str, Any]:
    from benchmarks import environment

    try:
        numpy_config: Any = np.show_config(mode="dicts")
    except (TypeError, ValueError):
        numpy_config = None
    return {
        "runtime": environment.runtime(),
        "capture": environment.capture(),
        "frames2py_version": frames2py.__version__,
        "frames2py_file": frames2py.__file__,
        "versions": _versions(),
        "numpy_config": numpy_config,
        "switch_interval_s": sys.getswitchinterval(),
        "gc_thresholds": list(gc.get_threshold()),
        "gc_enabled": gc.isenabled(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLOSC_NTHREADS",
                        "PYTHON_GIL")},
        "python_env": sorted(k for k in os.environ if k.startswith("PYTHON")),
        "preregistration_sha256": _sha256_file(PREREGISTRATION),
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    """One run in this process (the child). Writes ``<out_dir>/<run_id>.npz`` and returns the
    run's record, which the caller writes as JSON."""
    started = time.perf_counter_ns()
    condition = Condition.from_record(request["condition"]) if "condition" in request else PREREGISTERED
    run_id = request["run_id"]
    out_dir = Path(request["out_dir"])
    record: dict[str, Any] = {"schema": "frames2py-observation-run/1", "request": request, "run_id": run_id}
    if request.get("check_runtime", True):
        problems = runtime_problems(request)
        if problems:
            record["refused"] = problems
            return record
    record["environment"] = _environment_record(request)
    record["threads_before"] = threading.active_count()
    npz = out_dir / f"{run_id}.npz"
    if npz.exists():
        raise FileExistsError(f"{npz} exists; results are never overwritten")

    source = Source(condition)
    record["pool_sha256"] = source.digest()
    record["offset_rule"] = source.offset_rule()
    expected = request.get("expected_pool_sha256")
    record["pool_digest_ok"] = expected is None or expected == record["pool_sha256"]
    offsets, maxima = source.schedule()

    from benchmarks import power

    try:
        with power.hold_awake(f"frames2py observation {run_id}") as awake:
            record.update(_measure(request, condition, source, offsets, maxima, npz))
    except power.PowerStateError as exc:
        record["power_refused"] = str(exc)
        return record
    record["power"] = awake
    record["threads_after"] = threading.active_count()
    record["child_duration_ns"] = time.perf_counter_ns() - started
    return record


def _measure(request: dict[str, Any], condition: Condition, source: Source, offsets: list[int],
             maxima: list[int], npz: Path) -> dict[str, Any]:
    arm_name, n = request["arm"], int(request["n"])
    full = request.get("instrumentation", "full") == "full"
    ctx = Context(condition, make_work(request["workload"], request.get("k5")), full=full)
    recorder = None
    h5_path: Path | None = None
    out: dict[str, Any] = {}
    if arm_name in P2_ARMS:
        from frames2py import recorder as recorder_module

        tmp = Path(request["tmp_dir"])
        tmp.mkdir(parents=True, exist_ok=True)
        h5_path = tmp / f"{request['run_id']}.h5"
        recorder = recorder_module.open(h5_path, sensor_size=condition.sensor_size, compression="blosc")
    arm = build_arm(arm_name, n, ctx, recorder=recorder)
    producer_log = ProducerLog(len(offsets))
    ready = threading.Semaphore(0)
    threads = [threading.Thread(target=_consumer_thread, args=(ctx, c, ready), name=f"consumer-{i}", daemon=True)
               for i, c in enumerate(arm.consumers)]
    recorder_result: dict[str, Any] = {}
    recorder_thread = None
    if isinstance(arm, _PreservationArm) and arm.recorder_consumer is not None:
        recorder_thread = threading.Thread(target=_recorder_thread, args=(ctx, arm.recorder_consumer, recorder_result),
                                           name="recorder", daemon=True)
    producer = threading.Thread(target=_producer, args=(ctx, arm, source, offsets, producer_log, ready,
                                                        len(threads)), name="producer", daemon=True)
    monitor = Monitor(ctx)
    for t in threads:
        t.start()
    if recorder_thread is not None:
        recorder_thread.start()
    producer.start()
    try:
        monitor.run()
    except BaseException as exc:  # noqa: BLE001
        ctx.fail("monitor", exc)
    ctx.stop.set()
    deadline = time.monotonic() + SHUTDOWN_GRACE_S
    for t in [producer, *threads]:
        t.join(max(0.0, deadline - time.monotonic()))
    alive = [t.name for t in (producer, *threads) if t.is_alive()]
    if recorder_thread is not None:
        recorder_thread.join(DRAIN_CAP_S + SHUTDOWN_GRACE_S)
        if recorder_thread.is_alive():
            alive.append(recorder_thread.name)
    out["alive_after_grace"] = alive
    out["t_start"] = ctx.t_start
    out["t0"] = ctx.t0
    out["t1"] = ctx.t1
    out["errors"] = ctx.errors
    out["memory_error"] = ctx.memory_error
    out["memory_ceiling"] = monitor.memory_ceiling
    out["monitor_stopped_at"] = monitor.stopped_at
    out["footprint_start"] = monitor.footprint_start
    out["gc_t0"] = monitor.gc_t0
    out["gc_t1"] = monitor.gc_t1
    out["batches"] = producer_log.count
    out["scheduled_batches"] = len(offsets)
    if alive:
        return out  # threads still hold the logs; nothing further is safe to read

    if recorder is not None:
        close_start = time.perf_counter_ns()
        recorder.close()
        out["recorder_close_ns"] = time.perf_counter_ns() - close_start
        out["drain_timeout"] = bool(recorder_result.get("drain_timeout"))
        drained = recorder_result.get("drained_at")
        out["completion_delay_ns"] = None if drained is None or not ctx.t1 else max(0, drained - ctx.t1)

    arrays: dict[str, NDArray[Any]] = {
        **producer_log.arrays(),
        "schedule_offset": np.array(offsets, dtype=np.int64),
        "m": np.array(maxima, dtype=np.int64),
        **monitor.arrays(),
    }
    logs = [c.log for c in arm.consumers] + arm.inline
    for log in logs:
        for key, value in log.arrays().items():
            arrays[f"c{log.index}_{key}"] = value
    out["consumers"] = len(logs)
    out["queues"] = [q.log() for q in arm.queues]
    if isinstance(arm, _PreservationArm):
        rc = arm.recorder_consumer
        written = arm.written if rc is None else rc.log.keys
        arrays["recorder_written"] = np.array(written, dtype=np.int64)
        arrays["recorder_o"] = np.array([] if rc is None else rc.log.o, dtype=np.int64)
    if arm.engine is not None:
        stats = arm.engine.stats
        out["engine_stats"] = {"events_ingested": stats.events_ingested,
                               "events_out_of_bounds": stats.events_out_of_bounds,
                               "snapshots_published": stats.snapshots_published}
        out["ingested_events"] = arm.ingested_events
    raw = [c for c in arm.consumers if isinstance(c, RawConsumer)]
    out["raw"] = [{"events": c.events, "last_key": c.last_key, "watermark": c.acc.watermark,
                   "last_published_watermark": c.last_published_watermark} for c in raw]
    np.savez(npz, **arrays)  # type: ignore[arg-type]  # numpy's stub types **kwds as allow_pickle
    out["npz"] = npz.name
    out["integrity"] = integrity(arm_name, n, condition, source, out, arrays, full)
    if h5_path is not None:
        out["readback"] = readback(h5_path, source, arrays["recorder_written"], condition)
        if out["readback"]["ok"]:
            h5_path.unlink()
    return out


def readback(path: Path, source: Source, written: NDArray[np.int64], condition: Condition) -> dict[str, Any]:
    """Read the recording back and compare it with the batches the recorder was given (21.4)."""
    from frames2py.adapters import hdf5

    got = hashlib.sha256()
    events = 0
    with hdf5.open(path, group="events", sensor_size=condition.sensor_size) as reader:
        for batch in reader:
            got.update(batch.tobytes())
            events += len(batch)
    fed = hashlib.sha256()
    fed_events = 0
    for k in written.tolist():
        batch = source.materialise(k)
        fed.update(batch.tobytes())
        fed_events += len(batch)
    return {"ok": got.hexdigest() == fed.hexdigest(), "events": events, "fed_events": fed_events,
            "sha256": got.hexdigest(), "fed_sha256": fed.hexdigest(), "file_bytes": path.stat().st_size}


# ---------------------------------------------------------------- integrity (21.4)


def resolver(m: Sequence[int], s: Sequence[int]) -> Callable[[int, int], int | None]:
    """``find(w, O)``: ``k* = min{k : m_k = w and S_k < O}`` (9.2), or ``None`` if no batch
    matches. *s* holds the released batches' S_k; later batches never match."""
    by_value: dict[int, list[int]] = {}
    for k, mk in enumerate(m[: len(s)]):
        by_value.setdefault(mk, []).append(k)

    def find(watermark: int, observed_at: int) -> int | None:
        for k in by_value.get(watermark, ()):
            if s[k] < observed_at:
                return k
        return None

    return find


def integrity(arm: str, n: int, condition: Condition, source: Source, out: dict[str, Any],
              arrays: dict[str, NDArray[Any]], full: bool) -> dict[str, Any]:
    """The accounting of 21.4. ``harness`` problems are HARNESS_FAILURE; the rest integrity."""
    harness: list[str] = []
    problems: list[str] = []
    count = out["batches"]
    s = arrays["producer_s"]
    e = arrays["producer_e"]
    if count and (np.any(s[:count] == 0) or np.any(e[:count] == 0)):
        harness.append("per-batch log has unfilled entries")
    if full and count:
        q = arrays["producer_q"]
        if np.any(np.diff(q) < 0) or np.any(s < q) or np.any(e < s) or np.any(np.diff(s) < 0):
            harness.append("producer clock samples are not monotonic")
    if not out.get("pool_digest_ok", True):
        problems.append("pool digest mismatch")
    consumers = out["consumers"]
    m = arrays["m"].tolist()
    find = resolver(m, s.tolist())
    for i in range(consumers):
        seq = arrays[f"c{i}_seq"]
        if len(seq) and np.any(np.diff(seq) <= 0):
            problems.append(f"consumer {i}: observed sequences do not increase strictly")
        if full:
            o, w = arrays[f"c{i}_o"], arrays[f"c{i}_w"]
            if len(o) and (np.any(w < o) or np.any(np.diff(o) < 0)):
                harness.append(f"consumer {i}: clock samples are not monotonic")
            unresolved = sum(find(int(wm), int(ob)) is None for wm, ob in zip(arrays[f"c{i}_wm"].tolist(), o.tolist()))
            if unresolved:
                problems.append(f"consumer {i}: {unresolved} watermarks resolve to no batch")
        keys = arrays[f"c{i}_keys"]
        if arm == "RB" and len(keys) and np.any(np.diff(keys) != 1):
            problems.append(f"consumer {i}: raw batches not taken in FIFO order without gaps")
        if arm in ("B", "C", "E") and len(keys) and np.any(np.diff(keys) <= 0):
            problems.append(f"consumer {i}: items not taken in FIFO order")
    stepped_events = count * condition.batch_size
    if arm in ENGINE_ARMS and "engine_stats" in out:
        stats = out["engine_stats"]
        if stats["events_ingested"] != out["ingested_events"] or out["ingested_events"] != stepped_events:
            problems.append(f"events_ingested {stats['events_ingested']}, ingested {out['ingested_events']}, "
                            f"stepped {stepped_events}")
        if stats["events_out_of_bounds"]:
            problems.append(f"events_out_of_bounds {stats['events_out_of_bounds']}")
    p = arrays.get("producer_p")
    published = int(p[count - 1]) if full and p is not None and count else 0
    for i, ql in enumerate(out["queues"]):
        dropped = len(ql["evicted"]) + len(ql["rejected"])
        if ql["offered"] != ql["enqueued"] + len(ql["abandoned"]) + len(ql["rejected"]):
            problems.append(f"queue {i}: offered {ql['offered']} != enqueued + abandoned + rejected")
        if ql["enqueued"] != ql["dequeued"] + len(ql["evicted"]) + len(ql["remaining"]):
            problems.append(f"queue {i}: enqueued {ql['enqueued']} != dequeued + evicted + remaining")
        if arm in ("B", "E", "RB", "EP-QB", "EP-QE") and dropped:
            problems.append(f"queue {i}: {dropped} drops in an arm whose policy declares none")
        if arm in ("B", "C", "E") and full and ql["offered"] != published:
            problems.append(f"queue {i}: offered {ql['offered']} of {published} publications")
        if arm in ("RB", "EP-QB", "EP-QE") and ql["offered"] != count:
            problems.append(f"queue {i}: offered {ql['offered']} of {count} batches")
        if arm in ("B", "C", "E") and ql["dequeued"] != len(arrays[f"c{i}_keys"]):
            problems.append(f"queue {i}: dequeued {ql['dequeued']}, consumer took {len(arrays[f'c{i}_keys'])}")
    if arm == "RB":
        for i, (ql, r) in enumerate(zip(out["queues"], out["raw"])):
            enqueued_events = ql["enqueued"] * condition.batch_size
            if enqueued_events != r["events"] + len(ql["remaining"]) * condition.batch_size:
                problems.append(f"consumer {i}: events enqueued != accumulated + remaining")
            if r["last_key"] is not None:
                final = r["watermark"] if r["watermark"] is not None else r["last_published_watermark"]
                if final != source.max_t(r["last_key"]):
                    problems.append(f"consumer {i}: final watermark {final} != m of the last batch")
    if arm in P2_ARMS:
        written = arrays["recorder_written"].tolist()
        if written != list(range(len(written))):
            problems.append("the recorder was not given batches 0, 1, 2, ... in order")
        fed = count - (len(out["queues"][0]["abandoned"]) if out["queues"] else 0)
        if not out.get("drain_timeout") and len(written) != fed:
            problems.append(f"recorded {len(written)} batches of {fed} fed")
    return {"harness": harness, "problems": problems}


# ---------------------------------------------------------------- lockstep (V1, V6)


class VirtualClock:
    """A cadence clock advanced by the caller: one batch period per step in V1."""

    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def monotonic_ns(self) -> int:
        return self.now_ns


@dataclasses.dataclass
class Lockstep:
    """One arm driven single-threaded: after every producer step each consumer ticks until it
    has nothing more to take. ``after_step`` runs after each step and its ticks."""

    arm: Arm
    source: Source
    batches: int
    before_step: Callable[[int], None] | None = None
    after_step: Callable[[int], None] | None = None

    def run(self) -> None:
        for k in range(self.batches):
            if self.before_step is not None:
                self.before_step(k)
            self.arm.step(k, self.source.materialise(k))
            for consumer in self.arm.consumers:
                while consumer.tick(None) and consumer.kind != "poll":
                    pass
            if isinstance(self.arm, _PreservationArm) and self.arm.recorder_consumer is not None:
                while self.arm.recorder_consumer.tick(None):
                    pass
            if self.after_step is not None:
                self.after_step(k)



# ---------------------------------------------------------------- CLI entry


def worker_main() -> int:
    request = json.load(sys.stdin)
    try:
        record = run(request)
    except BaseException as exc:  # noqa: BLE001 - reported to the driver as a harness failure
        record = {"schema": "frames2py-observation-run/1", "request": request, "run_id": request.get("run_id"),
                  "fatal": "".join(traceback.format_exception(exc))}
    out = Path(request["out_dir"]) / f"{request['run_id']}.json"
    if out.exists():
        print(json.dumps({"error": f"{out} exists"}))
        return 2
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({"record": str(out)}))
    sys.stdout.flush()
    alive = record.get("alive_after_grace")
    if alive:
        os._exit(3)  # threads are still running; don't wait for them
    return 0
