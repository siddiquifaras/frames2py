"""Asynchronous viewer consumer.

The viewer runs in a **daemon thread** that polls
:meth:`Engine.latest_snapshot` at a configurable target FPS.  It is
completely decoupled from the ingest path -- if it falls behind, it
skips frames.  It never blocks the engine.

Supported backends:
    - ``"opencv"`` -- ``cv2.imshow`` (requires ``frames2py[viewer-opencv]``)
    - ``"headless"`` -- no-op (for testing / recording-only pipelines)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from frames2py.display.base import Overlay, Renderer
from frames2py.display.cv_renderer import HeadlessRenderer, OpenCVRenderer

if TYPE_CHECKING:
    from frames2py.core.engine import Engine


def _normalize_frame(frame: NDArray[np.floating], colormap: int | None) -> NDArray[np.uint8]:
    """Convert a float frame to a ``(H, W, 3)`` BGR uint8 image.

    Parameters:
        frame: 2-D or 3-D float array from the kernel snapshot.
        colormap: OpenCV colormap constant, or ``None`` for grayscale.

    Returns:
        ``(H, W, 3)`` uint8 BGR image ready for display.
    """
    if frame.ndim == 3 and frame.shape[2] == 2:
        # Polarity kernel: channel 0 = ON (blue), channel 1 = OFF (red)
        h, w = frame.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        fmax = frame.max()
        if fmax > 0:
            on = (frame[:, :, 0] / fmax * 255).clip(0, 255).astype(np.uint8)
            off = (frame[:, :, 1] / fmax * 255).clip(0, 255).astype(np.uint8)
            out[:, :, 0] = on   # Blue channel = ON
            out[:, :, 2] = off  # Red channel = OFF
        return out

    # Scalar frame (event_count, time_surface, exp_decay).
    fmin, fmax = float(frame.min()), float(frame.max())
    if fmax - fmin < 1e-8:
        gray = np.zeros(frame.shape[:2], dtype=np.uint8)
    else:
        gray = ((frame - fmin) / (fmax - fmin) * 255).clip(0, 255).astype(np.uint8)

    if colormap is not None:
        try:
            import cv2

            return cv2.applyColorMap(gray, colormap)  # type: ignore[return-value]
        except ImportError:
            pass

    # Grayscale → BGR
    return np.stack([gray, gray, gray], axis=-1)


def _resolve_colormap(name: str | None) -> int | None:
    """Convert a colormap name to an OpenCV constant."""
    if name is None:
        return None
    try:
        import cv2

        cmap_attr = f"COLORMAP_{name.upper()}"
        return getattr(cv2, cmap_attr, cv2.COLORMAP_VIRIDIS)
    except ImportError:
        return None


class Viewer:
    """Asynchronous snapshot viewer running in a daemon thread.

    Parameters:
        engine: The :class:`~frames2py.core.engine.Engine` to poll.
        backend: ``"opencv"`` or ``"headless"``.
        fps: Target polling rate in frames per second.
        window_name: Window title.
        colormap: OpenCV colormap name (e.g. ``"viridis"``, ``"hot"``,
            ``"inferno"``), or ``None`` for grayscale.

    Example::

        viewer = Viewer(engine, backend="opencv", fps=30).start()
        # ... run pipeline ...
        viewer.stop()
    """

    def __init__(
        self,
        engine: Engine,
        backend: str = "opencv",
        fps: float = 30.0,
        window_name: str = "frames2py",
        colormap: str | None = None,
    ) -> None:
        self._engine = engine
        self._fps = fps
        self._window_name = window_name
        self._colormap_id = _resolve_colormap(colormap)
        self._overlays: list[Overlay] = []

        self._renderer: Renderer
        if backend == "opencv":
            self._renderer = OpenCVRenderer()
        elif backend == "headless":
            self._renderer = HeadlessRenderer()
        else:
            raise ValueError(f"Unknown viewer backend: {backend!r}")

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames_shown = 0
        self._frames_dropped = 0

    def start(self) -> Viewer:
        """Start the viewer daemon thread.  Returns ``self`` for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="frames2py-viewer",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the viewer thread to exit and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def add_overlay(self, overlay: Overlay) -> None:
        """Register a best-effort overlay."""
        self._overlays.append(overlay)

    @property
    def frames_shown(self) -> int:
        """Total frames rendered since start."""
        return self._frames_shown

    @property
    def frames_dropped(self) -> int:
        """Snapshots skipped because seq was unchanged."""
        return self._frames_dropped

    # ------------------------------------------------------------------
    # Main viewer loop (runs in daemon thread)
    # ------------------------------------------------------------------
    def _run(self) -> None:
        w, h = self._engine.sensor_size
        self._renderer.init(w, h, self._window_name)

        interval = 1.0 / self._fps
        last_seq = -1

        try:
            while not self._stop_event.is_set() and self._engine.running:
                result = self._engine.latest_snapshot()
                if result is None:
                    time.sleep(interval)
                    continue

                frame, meta = result
                if meta.seq == last_seq:
                    self._frames_dropped += 1
                    time.sleep(interval)
                    continue

                colorised = _normalize_frame(frame, self._colormap_id)
                for overlay in self._overlays:
                    try:
                        overlay.draw(colorised, meta)
                    except Exception:
                        pass

                alive = self._renderer.show(colorised)
                if not alive:
                    break

                last_seq = meta.seq
                self._frames_shown += 1
                time.sleep(interval)
        finally:
            self._renderer.destroy()
