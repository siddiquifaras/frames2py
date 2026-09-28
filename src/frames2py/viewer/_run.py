"""``run()``: show a snapshot source in a window, on the main thread."""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any, Final, Protocol

from frames2py.publish import Snapshot
from frames2py.viewer._render import RGB, check_options, render

INITIAL_SIZE: Final = (640, 480)
"""(width, height) of the window before the first snapshot sets it."""

Source = Callable[[], Snapshot | None]


def run(
    source: Source,
    *,
    interval_ms: float = 16.0,
    title: str = "Frames2Py",
    scale: float | None = None,
    window_us: float = 50_000,
) -> None:
    """Show the snapshots *source* returns in a window until the window is closed.

    Once per *interval_ms* the viewer calls *source*, typically ``engine.snapshot``, and,
    when the publication is not the one it last showed, renders it with ``render()``; a
    ``None`` shows black. The window is as large as the frame. The viewer runs entirely on
    the calling thread, which must be the main thread; it starts no thread. Run the producer
    on a thread of its own.

    Args:
        source: Called once per interval; returns a ``Snapshot`` or ``None``.
        interval_ms: How often to read *source*, in milliseconds.
        title: The window title.
        scale, window_us: Passed to ``render()``.

    Raises:
        RuntimeError: called from a thread other than the main thread.
        ImportError: pyglet is not installed (``frames2py[viewer]``).
        ValueError: *interval_ms*, *scale* or *window_us* is not a finite positive number.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "frames2py.viewer.run() must be called on the main thread; run the producer on another thread"
        )
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, (int, float)) or not (
        math.isfinite(interval_ms) and interval_ms > 0
    ):
        raise ValueError(f"interval_ms must be a finite positive number, got {interval_ms!r}")
    check_options(scale, window_us)
    if not callable(source):
        raise TypeError(f"source must be callable, like engine.snapshot, got {type(source).__name__}")
    pyglet = _require("pyglet")
    window = _PygletWindow(pyglet, str(title))
    try:
        loop(
            source,
            window,
            interval_ns=round(interval_ms * 1e6),
            draw=lambda snapshot: render(snapshot, scale=scale, window_us=window_us),
            clock=time.monotonic_ns,
            sleep=time.sleep,
        )
    finally:
        window.close()


class Window(Protocol):
    def dispatch(self) -> bool:
        """Handle pending window events; ``False`` once the window has been closed."""
        ...

    def show(self, image: RGB | None) -> None:
        """Use *image* from now on; ``None`` is black at the current size."""
        ...

    def present(self) -> None:
        """Draw the current image."""
        ...


def loop(
    source: Source,
    window: Window,
    *,
    interval_ns: int,
    draw: Callable[[Snapshot], RGB],
    clock: Callable[[], int],
    sleep: Callable[[float], object],
) -> None:
    """The viewer's cadence: one ``source()`` per interval, a render only for a new
    publication, until the window closes. Ticks that fall behind are skipped, not repeated."""
    shown: object = _NOTHING
    deadline = clock()
    while window.dispatch():
        snapshot = source()
        key = None if snapshot is None else snapshot.meta.sequence
        if key != shown:
            window.show(None if snapshot is None else draw(snapshot))
            shown = key
        window.present()
        deadline += interval_ns
        now = clock()
        if deadline > now:
            sleep((deadline - now) / 1e9)
        else:
            deadline = now


_NOTHING: Final = object()


class _PygletWindow:
    """A pyglet window showing one RGB image, driven one frame at a time."""

    def __init__(self, pyglet: ModuleType, title: str) -> None:
        self._pyglet = pyglet
        self._size = INITIAL_SIZE
        self._window: Any = pyglet.window.Window(*INITIAL_SIZE, caption=title, vsync=False)
        self._image: Any = None

    def dispatch(self) -> bool:
        self._window.dispatch_events()
        return not self._window.has_exit

    def show(self, image: RGB | None) -> None:
        if image is None:
            self._image = None  # present() clears to black
            return
        height, width = image.shape[:2]
        if (width, height) != self._size:
            self._window.set_size(width, height)
            self._size = (width, height)
        # A negative pitch: rows are stored top to bottom, as the frame's are.
        self._image = self._pyglet.image.ImageData(width, height, "RGB", image.tobytes(), pitch=-width * 3)

    def present(self) -> None:
        self._window.switch_to()
        self._window.clear()
        if self._image is not None:
            # Fill the drawable area: on a high-DPI display it is larger than the frame.
            self._image.blit(0, 0, width=self._window.width, height=self._window.height)
        self._window.flip()

    def close(self) -> None:
        self._window.close()


def _require(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"frames2py.viewer.run needs {module}; install it with: pip install 'frames2py[viewer]'") from exc
