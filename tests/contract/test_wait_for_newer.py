"""``Engine.wait_for_newer``: arguments, results, and real threads.

Every ``ingest()`` runs on one dedicated producer thread. Tests that need a waiter to be
blocked before something happens wait until it has registered with the Engine; that is
synchronisation for the test, not what the test asserts. The orderings of a wait's steps
against publication are covered exhaustively by ``test_wait_interleavings.py``.
"""

from __future__ import annotations

import collections
import concurrent.futures
import decimal
import fractions
import math
import signal
import sys
import threading
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

import numpy as np
import pytest

from tests.contract.api import impl
from tests.contract.helpers import SUPPORTED_RUNTIME, events
from tests.contract.test_concurrency import Producer as RecordingProducer
from tests.contract.test_concurrency import Record, problems, record

pytestmark = [
    pytest.mark.skipif(not SUPPORTED_RUNTIME, reason="the Engine refuses this free-threaded runtime"),
    pytest.mark.timeout(120),
]

HANG_GUARD_S = 60.0
"""Bound on waiting for a thread that should finish. Not a timing requirement."""
HOUR_MS = 3_600_000.0
SENSOR = (6, 4)


def batch(k: int) -> Any:
    """One ON event at t = k on a pixel of its own."""
    return events((k, k % SENSOR[0], (k // SENSOR[0]) % SENSOR[1], 1))


class ProducerThread:
    """Runs calls one at a time on one thread, which becomes the Engine's producer."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __call__(self, call: Callable[..., Any], *args: Any) -> Any:
        return self._pool.submit(call, *args).result(timeout=HANG_GUARD_S)

    def close(self) -> None:
        self._pool.shutdown()


@pytest.fixture
def producer() -> Any:
    thread = ProducerThread()
    yield thread
    thread.close()


class Waiter(threading.Thread):
    """One ``wait_for_newer`` call on its own thread."""

    def __init__(self, engine: Any, sequence: Any, **kwargs: Any) -> None:
        super().__init__(daemon=True)
        self._call = lambda: engine.wait_for_newer(sequence, **kwargs)
        self.result: Any = None
        self.error: BaseException | None = None
        self.elapsed = 0.0

    def run(self) -> None:
        start = time.monotonic()
        try:
            self.result = self._call()
        except BaseException as error:  # noqa: BLE001 - re-raised by outcome()
            self.error = error
        self.elapsed = time.monotonic() - start

    def outcome(self) -> Any:
        self.join(HANG_GUARD_S)
        assert not self.is_alive(), "the waiter did not return"
        if self.error is not None:
            raise self.error
        return self.result


def until_registered(engine: Any, count: int) -> None:
    """Spin until *count* waiters are registered with *engine*: test synchronisation only."""
    deadline = time.monotonic() + HANG_GUARD_S
    while len(engine._waiters) < count:
        assert time.monotonic() < deadline, f"{count} waiters never registered"
        time.sleep(0)


def published(engine: Any, producer: ProducerThread, k: int) -> Any:
    producer(engine.ingest, batch(k))
    snapshot = engine.snapshot()
    assert snapshot is not None
    return snapshot


def engine(interval_ms: float = 0.0, kernel: str = "event_count") -> Any:
    return impl.Engine(SENSOR, kernel, snapshot_interval_ms=interval_ms)


# ------------------------------------------------------------------ arguments


class TestArguments:
    @pytest.mark.parametrize("sequence", [True, False, 1.0, "1", np.int64(1), [1]],
                             ids=["True", "False", "1.0", "'1'", "np.int64", "list"])
    def test_sequence_must_be_an_int_or_none(self, sequence: Any) -> None:
        with pytest.raises(TypeError):
            engine().wait_for_newer(sequence, timeout=0)

    def test_a_negative_sequence_is_refused(self) -> None:
        with pytest.raises(ValueError):
            engine().wait_for_newer(-1, timeout=0)

    @pytest.mark.parametrize("timeout", [-1, -0.5, -math.inf, math.nan],
                             ids=["-1", "-0.5", "-inf", "nan"])
    def test_a_negative_or_nan_timeout_is_refused(self, timeout: float) -> None:
        with pytest.raises(ValueError):
            engine().wait_for_newer(None, timeout=timeout)

    @pytest.mark.parametrize("timeout", ["1", 1j, decimal.Decimal("0.1"), [0]],
                             ids=["'1'", "1j", "Decimal", "list"])
    def test_a_non_real_timeout_is_refused(self, timeout: Any) -> None:
        with pytest.raises(TypeError):
            engine().wait_for_newer(None, timeout=timeout)

    def test_timeout_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            engine().wait_for_newer(None, 0)  # type: ignore[misc]

    def test_bad_arguments_raise_even_when_a_newer_snapshot_exists(self, producer: ProducerThread) -> None:
        e = engine()
        published(e, producer, 1)
        with pytest.raises(TypeError):
            e.wait_for_newer(True)
        with pytest.raises(ValueError):
            e.wait_for_newer(None, timeout=-1)

    @pytest.mark.parametrize(
        "timeout", [0, 0.0, 0.25, 2, fractions.Fraction(1, 4), np.float32(0.25), math.inf,
                    threading.TIMEOUT_MAX * 4, 10**400],
        ids=["0", "0.0", "0.25", "2", "Fraction", "np.float32", "inf", "4xTIMEOUT_MAX", "10**400"])
    def test_real_timeouts_are_accepted(self, producer: ProducerThread, timeout: Any) -> None:
        e = engine()
        snapshot = published(e, producer, 1)
        assert e.wait_for_newer(None, timeout=timeout) is snapshot


# ------------------------------------------------------------------ results without contention


class TestResults:
    def test_nothing_published_and_no_wait(self) -> None:
        assert engine().wait_for_newer(None, timeout=0) is None

    def test_an_already_newer_snapshot_is_returned_at_once(self, producer: ProducerThread) -> None:
        e = engine()
        snapshot = published(e, producer, 1)
        assert e.wait_for_newer(None) is snapshot
        assert e.wait_for_newer(snapshot.meta.sequence - 1) is snapshot

    def test_the_same_sequence_is_not_newer(self, producer: ProducerThread) -> None:
        e = engine()
        snapshot = published(e, producer, 1)
        assert e.wait_for_newer(snapshot.meta.sequence, timeout=0) is None

    def test_the_latest_is_returned_and_publications_in_between_are_skipped(self, producer: ProducerThread) -> None:
        e = engine()
        first = published(e, producer, 1)
        published(e, producer, 2)
        latest = published(e, producer, 3)
        assert e.wait_for_newer(first.meta.sequence, timeout=0) is latest

    def test_a_timeout_returns_none_after_it_has_elapsed(self, producer: ProducerThread) -> None:
        e = engine()
        snapshot = published(e, producer, 1)
        start = time.monotonic()
        assert e.wait_for_newer(snapshot.meta.sequence, timeout=0.05) is None
        assert time.monotonic() - start >= 0.05

    def test_a_sequence_from_before_reset_is_satisfied_by_the_next_publication(self, producer: ProducerThread) -> None:
        e = engine()
        before = published(e, producer, 1)
        producer(e.reset)
        assert e.wait_for_newer(before.meta.sequence, timeout=0) is None
        assert e.wait_for_newer(None, timeout=0) is None
        after = published(e, producer, 2)
        assert e.wait_for_newer(before.meta.sequence, timeout=0) is after
        assert after.meta.sequence > before.meta.sequence and after.meta.watermark == 2

    def test_a_stopped_engine_keeps_its_snapshot_and_publishes_nothing_more(self, producer: ProducerThread) -> None:
        e = engine()
        snapshot = published(e, producer, 1)
        producer(e.stop)
        producer(e.ingest, batch(2))
        assert e.wait_for_newer(None, timeout=0) is snapshot
        assert e.wait_for_newer(snapshot.meta.sequence, timeout=0.01) is None


class TestProducerThread:
    def test_the_producer_thread_is_refused(self) -> None:
        e = engine()
        e.ingest(batch(1))
        with pytest.raises(RuntimeError):
            e.wait_for_newer(None, timeout=0)
        with pytest.raises(RuntimeError):
            e.wait_for_newer(None)

    def test_a_no_op_ingest_while_stopped_also_makes_the_producer(self) -> None:
        e = engine()
        e.stop()
        e.ingest(batch(1))
        with pytest.raises(RuntimeError):
            e.wait_for_newer(None, timeout=0)

    def test_any_thread_may_wait_before_a_producer_exists(self) -> None:
        e = engine()
        assert e.wait_for_newer(None, timeout=0) is None
        with pytest.raises(TypeError):
            e.ingest(np.zeros(3))  # rejected: doesn't make this thread the producer
        assert e.wait_for_newer(None, timeout=0) is None

    def test_other_threads_wait_normally(self, producer: ProducerThread) -> None:
        e = engine()
        snapshot = published(e, producer, 1)
        assert e.wait_for_newer(None) is snapshot


# ------------------------------------------------------------------ real threads


class TestBlocking:
    def test_a_waiter_blocks_until_the_next_publication(self, producer: ProducerThread) -> None:
        e = engine()
        first = published(e, producer, 1)
        waiter = Waiter(e, first.meta.sequence)
        waiter.start()
        until_registered(e, 1)
        second = published(e, producer, 2)
        assert waiter.outcome() is second

    @pytest.mark.parametrize("timeout", [None, math.inf, 30.0, threading.TIMEOUT_MAX * 4, 10**400],
                             ids=["None", "inf", "30", "4xTIMEOUT_MAX", "10**400"])
    def test_a_publication_ends_any_wait(self, producer: ProducerThread, timeout: Any) -> None:
        e = engine()
        waiter = Waiter(e, None, timeout=timeout)
        waiter.start()
        until_registered(e, 1)
        snapshot = published(e, producer, 1)
        assert waiter.outcome() is snapshot

    def test_every_waiter_is_woken_by_one_publication(self, producer: ProducerThread) -> None:
        e = engine()
        first = published(e, producer, 1)
        waiters = [Waiter(e, first.meta.sequence) for _ in range(8)]
        for waiter in waiters:
            waiter.start()
        until_registered(e, len(waiters))
        second = published(e, producer, 2)
        assert all(waiter.outcome() is second for waiter in waiters)

    def test_a_publication_that_is_not_newer_leaves_the_waiter_waiting(self, producer: ProducerThread) -> None:
        e = engine()
        first = published(e, producer, 1)
        second = published(e, producer, 2)
        waiter = Waiter(e, second.meta.sequence)  # first + 1 publications from now
        waiter.start()
        until_registered(e, 1)
        third = published(e, producer, 3)
        assert waiter.outcome() is third
        assert first.meta.sequence < second.meta.sequence < third.meta.sequence


class TestLifecycle:
    def test_stop_from_another_thread_wakes_waiters_with_the_pending_window(self, producer: ProducerThread) -> None:
        e = engine(HOUR_MS)
        first = published(e, producer, 1)
        producer(e.ingest, batch(2))  # pending: the interval hasn't elapsed
        waiter = Waiter(e, first.meta.sequence)
        waiter.start()
        until_registered(e, 1)
        e.stop()
        result = waiter.outcome()
        assert result is e.snapshot() and result.meta.sequence > first.meta.sequence
        assert result.meta.watermark == 2 and int(result.frame.sum()) == 1  # the pending window only

    def test_stop_with_nothing_pending_wakes_nobody(self, producer: ProducerThread) -> None:
        e = engine(HOUR_MS)
        first = published(e, producer, 1)
        waiter = Waiter(e, first.meta.sequence, timeout=0.2)
        waiter.start()
        until_registered(e, 1)
        e.stop()
        assert waiter.outcome() is None
        assert waiter.elapsed >= 0.2

    def test_reset_from_another_thread_wakes_nobody_and_the_next_publication_does(
        self, producer: ProducerThread
    ) -> None:
        e = engine()
        before = published(e, producer, 1)
        waiter = Waiter(e, before.meta.sequence)
        waiter.start()
        until_registered(e, 1)
        e.reset()
        assert waiter.is_alive()
        after = published(e, producer, 7)
        result = waiter.outcome()
        assert result is after
        assert result.meta.sequence > before.meta.sequence and result.meta.watermark == 7
        assert int(result.frame.sum()) == 1

    def test_reset_then_a_timeout_returns_none(self, producer: ProducerThread) -> None:
        e = engine()
        published(e, producer, 1)
        waiter = Waiter(e, None, timeout=0.2)
        producer(e.reset)
        waiter.start()
        assert waiter.outcome() is None
        assert waiter.elapsed >= 0.2


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs signal.pthread_kill")
class TestInterruption:
    @pytest.mark.parametrize("timeout", [None, 30.0], ids=["no timeout", "timeout"])
    def test_sigint_raises_keyboard_interrupt_in_a_main_thread_waiter(
        self, producer: ProducerThread, timeout: float | None
    ) -> None:
        # CPython's Lock.acquire() acts on a SIGINT that lands in the instant before it blocks only
        # when the wait next wakes (scratch/v11_phase1/probes/probe_sigint_window.py), so the
        # signal is repeated until the handler runs. The handler raises KeyboardInterrupt once,
        # as the default one does, and ignores the repeats, which could otherwise land after the
        # call has returned.
        assert threading.current_thread() is threading.main_thread()
        e = engine()
        first = published(e, producer, 1)
        stats, snapshot = e.stats, e.snapshot()
        main = threading.get_ident()
        handled = threading.Event()

        def handler(signum: int, frame: FrameType | None) -> None:
            if not handled.is_set():
                handled.set()
                raise KeyboardInterrupt

        def interrupt() -> None:
            until_registered(e, 1)
            while not handled.is_set():
                signal.pthread_kill(main, signal.SIGINT)
                handled.wait(0.05)

        previous = signal.signal(signal.SIGINT, handler)
        sender = threading.Thread(target=interrupt, daemon=True)
        try:
            sender.start()
            with pytest.raises(KeyboardInterrupt):
                e.wait_for_newer(first.meta.sequence, timeout=timeout)
            sender.join(HANG_GUARD_S)
        finally:
            handled.set()
            signal.signal(signal.SIGINT, previous)
        assert (e.stats.events_ingested, e.stats.snapshots_published) == (stats.events_ingested, stats.snapshots_published)
        assert e.snapshot() is snapshot
        assert len(e._waiters) == 0  # the interrupted call left no registration behind
        second = published(e, producer, 2)
        assert e.wait_for_newer(first.meta.sequence, timeout=0) is second


class _Interrupted(Exception):
    """Stands in for a KeyboardInterrupt that lands on the producer mid-publication."""


class _InterruptingRegistry(collections.deque):  # type: ignore[type-arg]
    """The Engine's waiter registry, raising once at the *at*-th ``popleft``, before popping."""

    def __init__(self, items: Any, at: int) -> None:
        super().__init__(items)
        self._calls, self._at = 0, at

    def popleft(self) -> Any:
        self._calls += 1
        if self._calls == self._at:
            raise _Interrupted
        return super().popleft()


class _DrainsWithoutSkipping(impl.Engine):
    """A deliberately broken Engine: its drain releases whatever it pops before its own marker,
    including a marker an interrupted publication left behind."""

    def _publish(self, now_ns: int) -> None:
        self._sequence += 1
        buffer = self._publisher.begin_write()
        self._accumulator._read_into(buffer)
        self._publisher.end_write(impl.SnapshotMeta(watermark=self._accumulator.watermark, sequence=self._sequence))
        marker = object()
        self._waiters.append(marker)
        while (waiter := self._waiters.popleft()) is not marker:
            waiter.release()
        self._accumulator._close_window()
        self._snapshots_published += 1
        self._pending = False
        self._last_published_ns = now_ns


def _after_an_interrupted_drain(engine_type: Any) -> tuple[BaseException | None, Any, Any]:
    """Interrupt a publication after it released one of two waiters, then publish again.
    Returns what the second publication raised, if anything, the waiter the interrupted one
    stranded, and the snapshot it should get."""
    e = engine_type(SENSOR, "event_count", snapshot_interval_ms=0)
    producer = ProducerThread()
    try:
        first = published(e, producer, 1)
        waiters = [Waiter(e, first.meta.sequence) for _ in range(2)]
        for waiter in waiters:
            waiter.start()
        until_registered(e, 2)
        e._waiters = _InterruptingRegistry(e._waiters, at=2)
        with pytest.raises(_Interrupted):
            producer(e.ingest, batch(2))
        e._waiters = collections.deque(e._waiters)
        deadline = time.monotonic() + HANG_GUARD_S
        while all(w.is_alive() for w in waiters):  # the waiter the interrupted publication released
            assert time.monotonic() < deadline, "the interrupted publication released no waiter"
            time.sleep(0)
        released = next(w for w in waiters if not w.is_alive())
        assert released.outcome().meta.watermark == 2
        (stranded,) = [w for w in waiters if w is not released]
        assert stranded.is_alive()
        error = None
        try:
            producer(e.ingest, batch(3))
        except Exception as raised:  # noqa: BLE001 - returned to the caller
            error = raised
        return error, stranded, e.snapshot()
    finally:
        producer.close()


class TestInterruptedPublication:
    def test_a_later_publication_works_and_releases_the_stranded_waiter(self) -> None:
        error, stranded, snapshot = _after_an_interrupted_drain(impl.Engine)
        assert error is None
        assert stranded.outcome() is snapshot and snapshot.meta.watermark == 3

    def test_the_check_catches_a_drain_that_releases_a_leftover_marker(self) -> None:
        error, stranded, _ = _after_an_interrupted_drain(_DrainsWithoutSkipping)
        assert isinstance(error, AttributeError)
        stranded.join(HANG_GUARD_S)


class TestNoHistory:
    def test_no_registration_outlives_its_call(self, producer: ProducerThread) -> None:
        # No unbounded history: timed-out, satisfied and interrupted waits leave nothing behind.
        e = engine()
        first = published(e, producer, 1)
        for _ in range(50):
            assert e.wait_for_newer(first.meta.sequence, timeout=0.001) is None
        waiters = [Waiter(e, first.meta.sequence) for _ in range(4)]
        for waiter in waiters:
            waiter.start()
        until_registered(e, 4)
        published(e, producer, 2)
        for waiter in waiters:
            waiter.outcome()
        assert len(e._waiters) == 0


class TestStress:
    @pytest.mark.parametrize("timeout", [None, 0.001], ids=["blocking", "timed"])
    @pytest.mark.parametrize("kernel", ["event_count", "time_surface"])
    def test_repeated_waits_see_only_publications_in_increasing_order(self, kernel: str, timeout: Any) -> None:
        # Waiters pass back the sequence they got and are re-registering throughout, so every
        # publication finds some registered; a lost wakeup leaves one blocked and the hang
        # guard fails the test.
        source = RecordingProducer(kernel, 0.0)
        e = source.engine
        publications, waiters = 300, 4
        last_before_final: list[int] = []
        reads: list[list[Record]] = [[] for _ in range(waiters)]
        errors: list[BaseException] = []

        def wait_loop(slot: list[Record]) -> None:
            try:
                sequence = None
                while True:
                    snapshot = e.wait_for_newer(sequence, timeout=timeout)
                    if snapshot is None:
                        continue
                    assert sequence is None or snapshot.meta.sequence > sequence
                    slot.append(record(snapshot))
                    sequence = snapshot.meta.sequence
                    if last_before_final and sequence > last_before_final[0]:
                        return
            except BaseException as error:  # noqa: BLE001 - re-raised below
                errors.append(error)

        threads = [threading.Thread(target=wait_loop, args=(slot,), daemon=True) for slot in reads]
        for thread in threads:
            thread.start()

        def produce() -> None:
            for _ in range(publications - 1):
                source.ingest()
            last_before_final.append(max(source.published))
            source.ingest()

        ProducerThread()(produce)
        for thread in threads:
            thread.join(HANG_GUARD_S)
        assert not any(thread.is_alive() for thread in threads), "a waiter was left blocked"
        if errors:
            raise errors[0]
        assert len(source.published) == publications
        for slot in reads:
            assert problems(slot, source.published) == []
            assert slot[-1][0] == max(source.published)


# ------------------------------------------------------------------ the producer never waits for a waiter


class _PausedWaiter(threading.Thread):
    """Calls ``wait_for_newer`` and pauses at the *pause_at*-th line it executes inside the
    call, in any frame, until released. Stands in for a waiter descheduled at that point."""

    def __init__(self, engine: Any, sequence: int, pause_at: int) -> None:
        super().__init__(daemon=True)
        self._engine, self._sequence, self._pause_at = engine, sequence, pause_at
        self.paused, self.resume = threading.Event(), threading.Event()
        self.result: Any = None
        self.done = False
        self.lines = 0

    def _trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if event == "line":
            self.lines += 1
            if self.lines == self._pause_at:
                self.paused.set()
                self.resume.wait(HANG_GUARD_S)
        return self._trace

    def run(self) -> None:
        sys.settrace(self._trace)
        try:
            self.result = self._engine.wait_for_newer(self._sequence)
        finally:
            sys.settrace(None)
            self.done = True
            self.paused.set()


REACH_S = 1.0
"""How long a waiter gets to reach its pause line before it is taken to have blocked first.
Only ends the scan of lines; it proves nothing."""


def _publications_complete_while_a_waiter_is_paused(engine_type: Any, pause_at: int) -> bool | None:
    """Pause a waiter at its *pause_at*-th line, before or after the publication that wakes
    it, then publish from the producer. Returns whether those publications finished while it
    stayed paused, or ``None`` if the call has fewer lines than that."""
    e = engine_type(SENSOR, "event_count", snapshot_interval_ms=0)
    producer = ProducerThread()
    try:
        first = published(e, producer, 1)
        waiter = _PausedWaiter(e, first.meta.sequence, pause_at)
        waiter.start()
        if not waiter.paused.wait(REACH_S):  # it blocked before that line: wake it
            producer(e.ingest, batch(2))
            assert waiter.paused.wait(HANG_GUARD_S)
        if waiter.done:
            waiter.join(HANG_GUARD_S)
            return None
        job = producer._pool.submit(lambda: [e.ingest(batch(k)) for k in range(3, 7)])
        try:
            job.result(timeout=10.0)
            finished = True
        except concurrent.futures.TimeoutError:
            finished = False
        waiter.resume.set()
        job.result(timeout=HANG_GUARD_S)
        waiter.join(HANG_GUARD_S)
        assert not waiter.is_alive() and waiter.result is not None
        return finished
    finally:
        producer.close()


def _paused_at_every_line(engine_type: Any, first_only: bool = False) -> list[int]:
    """The lines of a wait call at which pausing the waiter stopped the producer (the first
    such line only, with *first_only*)."""
    blocked = []
    for pause_at in range(1, 400):
        finished = _publications_complete_while_a_waiter_is_paused(engine_type, pause_at)
        if finished is None:
            assert pause_at > 1, "the call ran no line"
            return blocked
        if not finished:
            blocked.append(pause_at)
            if first_only:
                return blocked
    raise AssertionError("the wait call ran more than 400 lines")


@pytest.mark.timeout(600)
def test_publications_complete_while_a_waiter_is_paused_at_any_line_of_its_call() -> None:
    assert _paused_at_every_line(impl.Engine) == []


class _ConditionEngine(impl.Engine):
    """A deliberately broken Engine: waits on a ``threading.Condition`` that each publication
    notifies, so a waiter holds the condition's lock while it runs Python."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cond = threading.Condition()

    def wait_for_newer(self, sequence: Any, *, timeout: Any = None) -> Any:
        with self._cond:
            while True:
                snapshot = self._publisher.read()
                if snapshot is not None and snapshot.meta.sequence > sequence:
                    return snapshot
                self._cond.wait()

    def _publish(self, now_ns: int) -> None:
        super()._publish(now_ns)
        with self._cond:
            self._cond.notify_all()


@pytest.mark.timeout(600)
def test_the_pause_test_catches_a_condition_variable_wake() -> None:
    assert _paused_at_every_line(_ConditionEngine, first_only=True) != []
