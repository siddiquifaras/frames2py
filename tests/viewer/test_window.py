"""Opt-in (``pytest --display``): the pyglet window itself. Needs a display and ``frames2py[viewer]``.

These open real windows for a moment and run on the main thread, as ``run()`` requires.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from frames2py import Engine, EVENT_DTYPE
from frames2py.viewer import _run, run
from tests.adapters.backends import require_backend

pytestmark = pytest.mark.display


@pytest.fixture
def pyglet() -> Any:
    return require_backend("pyglet")[0]


def framebuffer(pyglet: Any) -> np.ndarray:
    """The window's colour buffer, rows top to bottom, as (height, width, 3)."""
    data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    raw = data.get_bytes("RGB", -data.width * 3)
    return np.frombuffer(raw, dtype=np.uint8).reshape(data.height, data.width, 3)


class Stop(Exception):
    pass


def test_the_window_shows_the_image_the_right_way_up(pyglet: Any) -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:60, :80] = (255, 0, 0)    # top left red
    image[:60, 80:] = (0, 255, 0)    # top right green
    image[60:, :80] = (0, 0, 255)    # bottom left blue
    image[60:, 80:] = (255, 255, 255)
    window = _run._PygletWindow(pyglet, "frames2py test")
    try:
        for _ in range(3):
            window.dispatch()
            window.show(image)
            window.present()
        window._window.flip = lambda: None  # draw once more without swapping, and read the back buffer
        window.present()
        shot = framebuffer(pyglet)
    finally:
        window.close()
    h, w = shot.shape[:2]  # the buffer may be larger than the window on a high-DPI display
    corners = {(h // 4, w // 4): (255, 0, 0), (h // 4, 3 * w // 4): (0, 255, 0),
               (3 * h // 4, w // 4): (0, 0, 255), (3 * h // 4, 3 * w // 4): (255, 255, 255)}
    for (row, col), colour in corners.items():
        assert tuple(shot[row, col]) == colour, (row, col)
    assert h * 160 == w * 120


def test_run_shows_an_engine_fed_by_another_thread_and_closes_its_window(pyglet: Any) -> None:
    engine = Engine((320, 240), "polarity", snapshot_interval_ms=16)
    done = threading.Event()
    errors: list[BaseException] = []

    def produce() -> None:
        rng = np.random.default_rng(0)
        t = 0
        try:
            while not done.is_set():
                events = np.zeros(5_000, dtype=EVENT_DTYPE)
                events["t"] = t + np.arange(5_000)
                events["x"], events["y"] = rng.integers(0, 320, 5_000), rng.integers(0, 240, 5_000)
                events["p"] = rng.integers(0, 2, 5_000)
                engine.ingest(events)
                t += 5_000
        except BaseException as exc:  # noqa: BLE001 - checked below
            errors.append(exc)

    producer = threading.Thread(target=produce)
    producer.start()
    reads: list[Any] = []

    def source() -> Any:
        reads.append(engine.snapshot())
        if len(reads) == 40:
            raise Stop
        return reads[-1]

    try:
        with pytest.raises(Stop):
            run(source, interval_ms=10, title="frames2py test")
    finally:
        done.set()
        producer.join()
    assert errors == []
    assert len({s.meta.sequence for s in reads if s is not None}) > 1
    assert list(pyglet.app.windows) == []
