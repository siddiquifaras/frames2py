"""Concurrent observation: one producer thread, several consumer threads, public API only.

No sleeps. Threads are ordered with barriers and events, and every loop is bounded by a
count. Readers record a summary of every snapshot they receive; after the threads join, each
record is checked against what the producer knows it published under that sequence number.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.contract.api import EVENT_DTYPE, impl
from tests.contract.helpers import GIL_ENABLED, SUPPORTED_RUNTIME

publish = importlib.import_module(f"{impl.__name__}.publish")

requires_supported_runtime = pytest.mark.skipif(
    not SUPPORTED_RUNTIME,
    reason="free-threaded build not verified for this minor version; the Engine refuses it",
)

HANG_GUARD_S = 120
"""pytest-timeout watchdog against deadlock. Not a timing requirement."""

SENSOR = (320, 240)
READERS = 4
HOUR_MS = 3_600_000.0

Summary = tuple[tuple[int, int], ...]
Record = tuple[int, int | None, Summary] | None


def summary(frame: NDArray[Any]) -> Summary:
    """(min, max) per channel. A frame that mixes two publications has min != max."""
    channels = [frame[..., c] for c in range(frame.shape[-1])] if frame.ndim == 3 else [frame]
    return tuple((int(ch.min()), int(ch.max())) for ch in channels)


def record(snapshot: Any) -> Record:
    if snapshot is None:
        return None
    return snapshot.meta.sequence, snapshot.meta.watermark, summary(snapshot.frame)


def problems(reads: list[Record], published: dict[int, tuple[int | None, Summary]]) -> list[str]:
    """What is wrong with one reader's reads, in order.

    Every snapshot must be exactly what was published under its sequence number, frame and
    metadata together. Sequences never go backwards, and after a ``None`` the next snapshot
    is newer than any seen before it.
    """
    found: list[str] = []
    last = 0
    after_none = False
    for i, read in enumerate(reads):
        if read is None:
            after_none = True
            continue
        sequence, watermark, got = read
        if sequence not in published:
            found.append(f"read {i}: sequence {sequence} was never published")
        elif published[sequence] != (watermark, got):
            found.append(f"read {i}: sequence {sequence} got {(watermark, got)}, published {published[sequence]}")
        if sequence < last or (after_none and sequence == last):
            found.append(f"read {i}: sequence {sequence} after {last}")
        last = max(last, sequence)
        after_none = False
    return found


class StampedStream:
    """Ingest ``k`` puts ``1 + k % 2`` events on every pixel at ``t = k``, all OFF when
    ``k % 3 == 0`` and all ON otherwise. Every frame the kernels publish is then uniform per
    channel, and ``expected`` says which values it must hold."""

    def __init__(self, kernel: str, sensor: tuple[int, int] = SENSOR) -> None:
        width, height = sensor
        ys, xs = np.divmod(np.arange(width * height), width)
        self._batches = {}
        for copies in (1, 2):
            batch = np.empty(width * height * copies, dtype=EVENT_DTYPE)
            batch["x"] = np.tile(xs, copies)
            batch["y"] = np.tile(ys, copies)
            self._batches[copies] = batch
        self._kernel = kernel
        self.k = 0
        self._off = self._on = 0
        self._latest: int | None = None

    def next(self) -> NDArray[Any]:
        """The next ingest's events. Valid until the following call."""
        self.k += 1
        copies = 1 + self.k % 2
        on = self.k % 3 != 0
        batch = self._batches[copies]
        batch["t"] = self.k
        batch["p"] = 1 if on else 0
        if on:
            self._on += copies
        else:
            self._off += copies
        self._latest = self.k
        return batch

    def expected(self) -> tuple[int | None, Summary]:
        """Watermark and summary of a frame published now."""
        if self._kernel == "event_count":
            n = self._off + self._on
            return self._latest, ((n, n),)
        if self._kernel == "polarity":
            return self._latest, ((self._off, self._off), (self._on, self._on))
        assert self._latest is not None
        return self._latest, ((self._latest, self._latest),)

    def window_closed(self) -> None:
        if self._kernel != "time_surface":
            self._off = self._on = 0

    def reset(self) -> None:
        self._off = self._on = 0
        self._latest = None


class Producer:
    """Drives an Engine from one thread and records every publication it causes."""

    def __init__(self, kernel: str, interval_ms: float) -> None:
        self.engine = impl.Engine(SENSOR, kernel, snapshot_interval_ms=interval_ms)
        self.stream = StampedStream(kernel)
        self.published: dict[int, tuple[int | None, Summary]] = {}
        self._sequence = 0

    def _after(self, published_before: int) -> None:
        if self.engine.stats.snapshots_published != published_before:
            self._sequence += 1
            self.published[self._sequence] = self.stream.expected()
            self.stream.window_closed()

    def ingest(self) -> None:
        before = self.engine.stats.snapshots_published
        self.engine.ingest(self.stream.next())
        self._after(before)

    def stop(self) -> None:
        before = self.engine.stats.snapshots_published
        self.engine.stop()
        self._after(before)

    def reset(self) -> None:
        self.engine.reset()
        self.stream.reset()


def run_concurrently(
    produce: Callable[[], None],
    readers: list[Callable[[threading.Event], None]],
    steps: tuple[threading.Barrier, ...] = (),
) -> None:
    """Run *readers* on their own threads and *produce* on this one, all released together.
    Readers get an event that is set once *produce* returns. Re-raises any thread's error;
    on an error, *steps* are broken so no thread waits on them forever."""
    done = threading.Event()
    start = threading.Barrier(len(readers) + 1)
    errors: list[BaseException] = []

    def run(reader: Callable[[threading.Event], None]) -> None:
        try:
            start.wait()
            reader(done)
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            errors.append(error)
            for step in steps:
                step.abort()

    threads = [threading.Thread(target=run, args=(r,), daemon=True) for r in readers]
    for thread in threads:
        thread.start()
    try:
        start.wait()
        produce()
    except BaseException:
        for step in steps:
            step.abort()
        raise
    finally:
        done.set()
        for thread in threads:
            thread.join()
    if errors:
        raise errors[0]


def reading_until_done(snapshot: Callable[[], Any], reads: list[Record]) -> Callable[[threading.Event], None]:
    def reader(done: threading.Event) -> None:
        while not done.is_set():
            reads.append(record(snapshot()))
        reads.append(record(snapshot()))

    return reader


class _TearingPublisher:
    """A deliberately broken publisher: one buffer rewritten in place, metadata read after
    the frame. Used only to show that this harness detects torn snapshots."""

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self._buffer = np.zeros(shape, dtype=dtype)
        self._meta: Any = None

    def begin_write(self) -> NDArray[Any]:
        return self._buffer

    def end_write(self, meta: Any) -> None:
        self._meta = meta

    def read(self) -> Any:
        if self._meta is None:
            return None
        frame = self._buffer.copy()
        return publish.Snapshot(frame, self._meta)


def _publish_stamped(publisher: Any, publications: int) -> dict[int, tuple[int | None, Summary]]:
    published: dict[int, tuple[int | None, Summary]] = {}
    for k in range(1, publications + 1):
        buffer = publisher.begin_write()
        buffer[...] = k
        publisher.end_write(impl.SnapshotMeta(watermark=k, sequence=k))
        published[k] = (k, ((k, k),))
    return published


def _publisher_run(publisher: Any, publications: int = 300) -> list[str]:
    reads: list[list[Record]] = [[] for _ in range(READERS)]
    published: dict[int, tuple[int | None, Summary]] = {}

    def produce() -> None:
        published.update(_publish_stamped(publisher, publications))

    run_concurrently(produce, [reading_until_done(publisher.read, r) for r in reads])
    assert all(any(r is not None for r in rs) for rs in reads)
    return [p for rs in reads for p in problems(rs, published)]


class TestHarness:
    def test_problems_flags_mixed_frames_and_mismatched_metadata(self) -> None:
        published = {1: (1, ((1, 1),)), 2: (2, ((2, 2),))}
        mixed = np.ones((4, 6), dtype=np.uint32)
        mixed[2:] = 2
        assert problems([(1, 1, summary(np.ones((4, 6), dtype=np.uint32)))], published) == []
        assert problems([(1, 1, summary(mixed))], published)
        assert problems([(1, 1, summary(np.full((4, 6), 2, dtype=np.uint32)))], published)
        assert problems([(2, 2, ((2, 2),)), (1, 1, ((1, 1),))], published)
        assert problems([(1, 1, ((1, 1),)), None, (1, 1, ((1, 1),))], published)

    @pytest.mark.timeout(HANG_GUARD_S)
    def test_detects_a_publisher_that_tears(self) -> None:
        assert _publisher_run(_TearingPublisher((240, 320), np.dtype(np.uint32)))


@requires_supported_runtime
@pytest.mark.timeout(HANG_GUARD_S)
class TestConcurrentReaders:
    def test_publisher_readers_get_only_complete_publications(self) -> None:
        assert _publisher_run(publish.ImmutablePublisher((240, 320), np.dtype(np.uint32))) == []

    @pytest.mark.parametrize("interval_ms", [0.0, 5.0])
    @pytest.mark.parametrize("kernel", ["event_count", "polarity", "time_surface"])
    def test_every_snapshot_is_a_published_state(self, kernel: str, interval_ms: float) -> None:
        producer = Producer(kernel, interval_ms)
        reads: list[list[Record]] = [[] for _ in range(READERS)]

        def produce() -> None:
            for _ in range(150):
                producer.ingest()
            producer.stop()

        run_concurrently(produce, [reading_until_done(producer.engine.snapshot, r) for r in reads])
        assert len(producer.published) > 1
        for rs in reads:
            assert problems(rs, producer.published) == []
            first = next(i for i, r in enumerate(rs) if r is not None)
            assert None not in rs[first:], "None after a snapshot, without reset()"
            assert rs[-1] is not None and rs[-1][0] == max(producer.published)

    @pytest.mark.parametrize("readers", [1, 4])
    def test_readers_finish_while_the_producer_never_pauses(self, readers: int) -> None:
        # The measured starvation case for a retrying reader: interval 0, small batches, a
        # large frame. The producer keeps ingesting until every reader has its snapshots;
        # the cap only turns starvation into a failure. A read here loads the published
        # snapshot once, with no copy and no retry. When the cap was calibrated, readers that
        # also copied every snapshot but never retried finished within 5 to 15 ingests; a
        # two-buffer seqlock that retried missed 20,000.
        sensor, batch, wanted, cap = (1280, 720), 10_000, 20, 2_000
        rng = np.random.default_rng(4)
        pool = []
        for _ in range(5):
            ev = np.empty(batch, dtype=EVENT_DTYPE)
            ev["t"] = np.arange(batch)
            ev["x"] = rng.integers(0, sensor[0], batch)
            ev["y"] = rng.integers(0, sensor[1], batch)
            ev["p"] = rng.integers(0, 2, batch)
            pool.append(ev)
        engine = impl.Engine(sensor, "event_count", snapshot_interval_ms=0)
        engine.ingest(pool[0])
        finished = threading.Semaphore(0)
        ingests = 0

        def reader(done: threading.Event) -> None:
            got = 0
            while got < wanted and not done.is_set():
                got += engine.snapshot() is not None
            if got == wanted:
                finished.release()

        def produce() -> None:
            nonlocal ingests
            remaining = readers
            while remaining and ingests < cap:
                engine.ingest(pool[ingests % 5])
                ingests += 1
                while remaining and finished.acquire(blocking=False):
                    remaining -= 1
            assert remaining == 0, f"{remaining} of {readers} readers starved over {ingests} ingests"

        run_concurrently(produce, [reader] * readers)

    def test_reads_overlapping_reset(self) -> None:
        producer = Producer("event_count", 0.0)
        reads: list[list[Record]] = [[] for _ in range(READERS)]

        def produce() -> None:
            for _ in range(30):
                for _ in range(5):
                    producer.ingest()
                producer.reset()

        run_concurrently(produce, [reading_until_done(producer.engine.snapshot, r) for r in reads])
        for rs in reads:
            assert problems(rs, producer.published) == []
            assert rs[-1] is None

    def test_a_read_after_reset_returns_gets_none_until_the_next_publication(self) -> None:
        producer = Producer("time_surface", 0.0)
        steps = [threading.Barrier(READERS + 1) for _ in range(2)]
        seen: list[list[tuple[Record, Record]]] = [[] for _ in range(READERS)]
        cycles = 50

        def reader(slot: list[tuple[Record, Record]]) -> Callable[[threading.Event], None]:
            def read(done: threading.Event) -> None:
                for _ in range(cycles):
                    steps[0].wait()
                    after_reset = record(producer.engine.snapshot())
                    steps[1].wait()
                    steps[0].wait()
                    after_ingest = record(producer.engine.snapshot())
                    steps[1].wait()
                    slot.append((after_reset, after_ingest))

            return read

        def produce() -> None:
            for _ in range(cycles):
                producer.ingest()
                producer.reset()
                steps[0].wait()
                steps[1].wait()
                producer.ingest()
                steps[0].wait()
                steps[1].wait()

        run_concurrently(produce, [reader(s) for s in seen], steps)
        for slot in seen:
            assert len(slot) == cycles
            for cycle, (after_reset, after_ingest) in enumerate(slot):
                assert after_reset is None
                assert after_ingest is not None
                sequence, watermark, got = after_ingest
                assert sequence == 2 * cycle + 2
                assert producer.published[sequence] == (watermark, got)

    def test_stop_publishes_the_pending_window_to_readers(self) -> None:
        producer = Producer("event_count", HOUR_MS)
        step = threading.Barrier(READERS + 1)
        seen: list[list[Record]] = [[] for _ in range(READERS)]

        def reader(slot: list[Record]) -> Callable[[threading.Event], None]:
            def read(done: threading.Event) -> None:
                for _ in range(3):
                    step.wait()
                    slot.append(record(producer.engine.snapshot()))
                    step.wait()

            return read

        def produce() -> None:
            producer.ingest()  # the first ingest publishes
            step.wait()
            step.wait()
            for _ in range(4):
                producer.ingest()  # pending: the interval hasn't elapsed
            producer.stop()
            step.wait()
            step.wait()
            stats = producer.engine.stats
            producer.engine.ingest(producer.stream.next())  # stopped: a no-op
            assert producer.engine.stats.events_ingested == stats.events_ingested
            step.wait()
            step.wait()

        run_concurrently(produce, [reader(s) for s in seen], (step,))
        assert sorted(producer.published) == [1, 2]
        for slot in seen:
            assert [r[0] if r else None for r in slot] == [1, 2, 2]
            assert problems(slot, producer.published) == []


@requires_supported_runtime
@pytest.mark.timeout(HANG_GUARD_S)
def test_ingest_never_waits_for_consumers_holding_snapshots() -> None:
    engine = impl.Engine(SENSOR, "time_surface", snapshot_interval_ms=0)
    stream = StampedStream("time_surface")
    engine.ingest(stream.next())
    held = threading.Barrier(READERS + 1)
    release = threading.Event()
    kept: list[tuple[NDArray[Any], NDArray[Any]]] = []
    lock = threading.Lock()

    def consumer(done: threading.Event) -> None:
        snapshot = engine.snapshot()
        assert snapshot is not None
        frame = snapshot.frame
        with lock:
            kept.append((frame, frame.copy()))
        held.wait()
        release.wait()

    def produce() -> None:
        try:
            held.wait()  # every consumer holds a snapshot and is parked
            before = engine.stats.snapshots_published
            for _ in range(50):
                engine.ingest(stream.next())
            assert engine.stats.snapshots_published == before + 50
        finally:
            release.set()

    run_concurrently(produce, [consumer] * READERS, (held,))
    for frame, copy in kept:
        np.testing.assert_array_equal(frame, copy)


class _AcquireRecorder:
    """Wraps the Engine's lifecycle lock and records which threads acquire it."""

    def __init__(self, lock: Any) -> None:
        self._lock = lock
        self.threads: set[int] = set()

    def __enter__(self) -> Any:
        self.threads.add(threading.get_ident())
        return self._lock.__enter__()

    def __exit__(self, *exc: object) -> Any:
        return self._lock.__exit__(*exc)


@requires_supported_runtime
@pytest.mark.skipif(GIL_ENABLED, reason="exercises truly parallel stats reads; with the GIL they are "
                    "serialised and covered by test_interleavings.py")
@pytest.mark.timeout(HANG_GUARD_S)
def test_stats_read_in_parallel_with_ingest() -> None:
    batch_size, least, cap = 1_000, 400, 20_000
    rng = np.random.default_rng(7)
    batch = np.empty(batch_size, dtype=EVENT_DTYPE)
    batch["t"] = np.arange(batch_size)
    batch["x"] = rng.integers(0, SENSOR[0] + 40, batch_size)  # some out of bounds
    batch["y"] = rng.integers(0, SENSOR[1], batch_size)
    batch["p"] = 1
    out_of_bounds = int((batch["x"] >= SENSOR[0]).sum())
    engine = impl.Engine(SENSOR, "event_count", snapshot_interval_ms=0)
    recorder = _AcquireRecorder(engine._lifecycle)
    engine._lifecycle = recorder
    seen: list[list[Any]] = [[] for _ in range(READERS)]

    def reader(slot: list[Any]) -> Callable[[threading.Event], None]:
        def read(done: threading.Event) -> None:
            slot.append(threading.get_ident())
            while not done.is_set():
                slot.append(engine.stats)
            slot.append(engine.stats)

        return read

    ingested = [0]

    def produce() -> None:
        # At least `least` ingests, then more until every reader has read twice during
        # ingestion; the cap only turns a reader that never gets to run into a failure.
        while ingested[0] < least or (ingested[0] < cap and not all(len(s) > 2 for s in seen)):
            engine.ingest(batch)
            ingested[0] += 1

    run_concurrently(produce, [reader(s) for s in seen])
    ingests = ingested[0]
    reader_threads = {s[0] for s in seen}
    assert not reader_threads & recorder.threads, "a stats reader took the lifecycle lock"
    for slot in seen:
        stats = slot[1:]
        assert len(stats) > 2, f"a reader read {len(stats) - 1} times during {ingests} ingests"
        for field in ("events_ingested", "events_out_of_bounds", "snapshots_published", "uptime_ns"):
            values = [getattr(s, field) for s in stats]
            assert all(isinstance(v, int) and v >= 0 for v in values), field
            assert all(a <= b for a, b in zip(values, values[1:])), f"{field} went backwards"
        for s in stats:
            assert s.events_ingested % batch_size == 0 and s.events_ingested <= ingests * batch_size
            assert s.events_out_of_bounds % out_of_bounds == 0 if out_of_bounds else s.events_out_of_bounds == 0
            assert s.events_out_of_bounds <= ingests * out_of_bounds
            assert s.snapshots_published <= ingests
        final = stats[-1]
        assert (final.events_ingested, final.events_out_of_bounds, final.snapshots_published) == (
            ingests * batch_size, ingests * out_of_bounds, ingests)

