"""Producer ownership, lifecycle calls from other threads, and the supported runtimes.

Each call runs to completion on its own thread before the next starts, so these tests fix
the order of calls; the orderings of calls that overlap are covered by
``test_interleavings.py``.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.contract.api import impl
from tests.contract.helpers import KERNELS, SENSOR, SUPPORTED_RUNTIME, events

requires_supported_runtime = pytest.mark.skipif(
    not SUPPORTED_RUNTIME, reason="the Engine refuses this free-threaded runtime"
)


def elsewhere(call: Callable[[], Any]) -> Any:
    """Run *call* on a new thread to completion; return its result or raised exception."""
    box: list[Any] = []

    def run() -> None:
        try:
            box.append(call())
        except BaseException as error:  # noqa: BLE001 - returned to the caller
            box.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return box[0]


def observe(engine: Any) -> tuple[Any, ...]:
    stats = engine.stats
    snap = engine.snapshot()
    published = None if snap is None else (snap.frame.tobytes(), snap.meta)
    return stats.events_ingested, stats.events_out_of_bounds, stats.snapshots_published, published


@requires_supported_runtime
class TestProducerOwnership:
    def test_the_first_ingest_makes_its_thread_the_producer(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.ingest(events((2, 2, 2, 0)))
        assert engine.stats.events_ingested == 2
        before = observe(engine)
        error = elsewhere(lambda: engine.ingest(events((3, 3, 3, 0))))
        assert isinstance(error, RuntimeError)
        assert observe(engine) == before

    def test_the_producer_can_be_another_thread_than_the_constructor(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        assert elsewhere(lambda: engine.ingest(events((1, 1, 1, 0)))) is None
        with pytest.raises(RuntimeError):
            engine.ingest(events((2, 2, 2, 0)))
        assert engine.stats.events_ingested == 1

    def test_a_rejected_first_call_does_not_claim_the_engine(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        assert isinstance(elsewhere(lambda: engine.ingest(np.zeros(3))), TypeError)
        assert isinstance(elsewhere(lambda: engine.ingest(events((2**63, 1, 1, 0)))), ValueError)
        engine.ingest(events((1, 1, 1, 0)))
        assert engine.stats.events_ingested == 1

    def test_ownership_survives_reset_from_another_thread(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        assert elsewhere(engine.reset) is None
        assert engine.snapshot() is None and engine.stats.events_ingested == 0
        assert isinstance(elsewhere(lambda: engine.ingest(events((2, 2, 2, 0)))), RuntimeError)
        engine.ingest(events((3, 3, 3, 0)))
        snap = engine.snapshot()
        assert int(snap.frame[3, 3]) == 1 and snap.meta.watermark == 3
        assert engine.stats.events_ingested == 1

    def test_a_non_producer_ingest_while_stopped_still_raises(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.stop()
        assert isinstance(elsewhere(lambda: engine.ingest(events((2, 2, 2, 0)))), RuntimeError)


@requires_supported_runtime
class TestLifecycleFromAnotherThread:
    def test_stop_from_another_thread_publishes_the_pending_window(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=3_600_000.0)
        engine.ingest(events((1, 1, 1, 0)))
        engine.ingest(events((2, 2, 2, 0)))
        assert engine.snapshot().meta.sequence == 1
        assert elsewhere(engine.stop) is None
        snap = engine.snapshot()
        assert snap.meta.sequence == 2 and int(snap.frame[2, 2]) == 1
        before = observe(engine)
        engine.ingest(events((3, 3, 3, 0)))
        assert observe(engine) == before

    def test_start_from_another_thread_resumes_the_producer(self) -> None:
        engine = KERNELS["event_count"].engine(interval_ms=0.0)
        engine.ingest(events((1, 1, 1, 0)))
        elsewhere(engine.stop)
        elsewhere(engine.start)
        engine.ingest(events((2, 2, 2, 0)))
        assert engine.snapshot().meta.watermark == 2

    def test_snapshot_and_stats_from_other_threads(self) -> None:
        engine = KERNELS["polarity"].engine(interval_ms=0.0)
        engine.ingest(events((5, 1, 1, 1)))
        snap = elsewhere(engine.snapshot)
        stats = elsewhere(lambda: engine.stats)
        assert snap is engine.snapshot()
        assert stats.events_ingested == 1


class TestSupportedRuntimes:
    def test_this_runtime(self) -> None:
        if SUPPORTED_RUNTIME:
            impl.Engine(SENSOR, "event_count")
        else:
            with pytest.raises(RuntimeError):
                impl.Engine(SENSOR, "event_count")

    @pytest.mark.parametrize(
        ("version", "gil_enabled", "refused"),
        [
            ((3, 13, 11, "final", 0), False, True),
            ((3, 15, 0, "final", 0), False, True),
            ((3, 14, 2, "final", 0), False, False),
            ((3, 13, 11, "final", 0), True, False),
            ((3, 15, 0, "final", 0), True, False),
        ],
        ids=["3.13t", "3.15t", "3.14t", "3.13 with GIL", "3.15 with GIL"],
    )
    def test_free_threaded_versions_are_refused_unless_verified(
        self, monkeypatch: pytest.MonkeyPatch, version: tuple[Any, ...], gil_enabled: bool, refused: bool
    ) -> None:
        monkeypatch.setattr(sys, "_is_gil_enabled", lambda: gil_enabled, raising=False)
        monkeypatch.setattr(sys, "version_info", version)
        if refused:
            with pytest.raises(RuntimeError):
                impl.Engine(SENSOR, "event_count")
        else:
            impl.Engine(SENSOR, "event_count")
        impl.Accumulator(SENSOR, "event_count").accumulate(events((1, 1, 1, 0)))
