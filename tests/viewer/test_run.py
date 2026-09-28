"""``frames2py.viewer.run``: cadence (with an injected clock, no sleeping), rendering only new
publications, black for ``None``, the main-thread rule, argument errors and the extra."""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

import numpy as np
import pytest

from frames2py import Engine, EVENT_DTYPE, SnapshotMeta
from frames2py.publish import Snapshot
from frames2py.viewer import _run, run


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000_000
        self.sleeps: list[float] = []

    def __call__(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += round(seconds * 1e9)


class FakeWindow:
    """Records what the loop shows; closes itself after *frames* presents."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.shown: list[Any] = []
        self.presents = 0

    def dispatch(self) -> bool:
        return self.presents < self.frames

    def show(self, image: Any) -> None:
        self.shown.append(image)

    def present(self) -> None:
        self.presents += 1


def snapshot(sequence: int) -> Snapshot:
    frame = np.full((2, 2), sequence, dtype=np.uint32)
    frame.flags.writeable = False
    return Snapshot(frame, SnapshotMeta(watermark=sequence, sequence=sequence))


def run_loop(sources: list[Snapshot | None], clock: FakeClock, interval_ns: int = 16_000_000,
             work_ns: int = 0) -> tuple[FakeWindow, list[int], list[int]]:
    """The loop over *sources*, one per tick; *work_ns* of clock time passes in each source call."""
    window, calls, drawn = FakeWindow(len(sources)), [], []
    items = iter(sources)

    def source() -> Snapshot | None:
        calls.append(clock.now)
        clock.now += work_ns
        return next(items)

    def draw(s: Snapshot) -> str:
        drawn.append(s.meta.sequence)
        return f"image {s.meta.sequence}"

    _run.loop(source, window, interval_ns=interval_ns, draw=draw, clock=clock, sleep=clock.sleep)  # type: ignore[arg-type]
    return window, calls, drawn


class TestCadence:
    def test_one_read_per_interval(self) -> None:
        clock = FakeClock()
        start = clock.now
        _, calls, _ = run_loop([None] * 6, clock, interval_ns=16_000_000)
        assert [c - start for c in calls] == [i * 16_000_000 for i in range(6)]

    def test_time_spent_in_a_tick_comes_out_of_the_sleep(self) -> None:
        clock = FakeClock()
        _, calls, _ = run_loop([None] * 4, clock, interval_ns=10_000_000, work_ns=3_000_000)
        assert [b - a for a, b in zip(calls, calls[1:])] == [10_000_000] * 3
        assert clock.sleeps == [0.007] * 4

    def test_a_late_tick_is_not_made_up(self) -> None:
        clock = FakeClock()
        _, calls, _ = run_loop([None] * 4, clock, interval_ns=10_000_000, work_ns=25_000_000)
        assert [b - a for a, b in zip(calls, calls[1:])] == [25_000_000] * 3
        assert clock.sleeps == []

    def test_a_real_engine_source(self) -> None:
        engine = Engine((4, 3), "event_count", snapshot_interval_ms=0)
        engine.ingest(np.array([(5, 1, 1, 1)], dtype=EVENT_DTYPE))
        clock = FakeClock()
        window = FakeWindow(3)
        _run.loop(engine.snapshot, window, interval_ns=1, draw=lambda s: s.frame.sum(), clock=clock, sleep=clock.sleep)
        assert window.shown == [1] and window.presents == 3


class TestWhatIsShown:
    def test_only_new_publications_are_rendered(self) -> None:
        a, b = snapshot(1), snapshot(2)
        window, _, drawn = run_loop([a, a, b, b, b, a], FakeClock())
        assert drawn == [1, 2, 1]
        assert window.shown == ["image 1", "image 2", "image 1"]
        assert window.presents == 6

    def test_none_is_black_and_shown_once_per_change(self) -> None:
        a = snapshot(1)
        window, _, drawn = run_loop([None, None, a, None, None, a], FakeClock())
        assert window.shown == [None, "image 1", None, "image 1"]
        assert drawn == [1, 1]

    def test_the_loop_ends_when_the_window_closes(self) -> None:
        window, calls, _ = run_loop([], FakeClock())
        assert (calls, window.shown, window.presents) == ([], [], 0)


class TestRunArguments:
    def test_off_the_main_thread_is_refused_before_any_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list[Any] = []
        monkeypatch.setattr(_run, "_PygletWindow", lambda *a: opened.append(a))
        errors: list[BaseException] = []

        def call() -> None:
            try:
                run(lambda: None)
            except BaseException as exc:  # noqa: BLE001 - reported to the main thread
                errors.append(exc)

        worker = threading.Thread(target=call)
        worker.start()
        worker.join()
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        assert "main thread" in str(errors[0]) and opened == []

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"interval_ms": 0}, ValueError),
            ({"interval_ms": -1}, ValueError),
            ({"interval_ms": float("inf")}, ValueError),
            ({"interval_ms": True}, ValueError),
            ({"scale": 0}, ValueError),
            ({"window_us": float("nan")}, ValueError),
        ],
    )
    def test_invalid_arguments_open_no_window(self, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any],
                                               error: type) -> None:
        opened: list[Any] = []
        monkeypatch.setattr(_run, "_PygletWindow", lambda *a: opened.append(a))
        with pytest.raises(error):
            run(lambda: None, **kwargs)
        assert opened == []

    def test_source_must_be_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            run(None)  # type: ignore[arg-type]

    def test_a_missing_pyglet_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pyglet", None)
        with pytest.raises(ImportError, match=r"frames2py\[viewer\]") as info:
            run(lambda: None)
        assert isinstance(info.value.__cause__, ImportError)


def test_importing_the_viewer_imports_no_gui_and_starts_no_thread() -> None:
    code = (
        "import sys, threading; before = threading.active_count(); import frames2py.viewer; "
        "assert 'pyglet' not in sys.modules; assert threading.active_count() == before"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
