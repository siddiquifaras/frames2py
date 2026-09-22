"""Built-in frame overlays for the viewer.

All overlays implement the :class:`~frames2py.display.base.Overlay`
protocol: they modify the colorised ``(H, W, 3)`` uint8 frame in-place
before it reaches the display backend.

Overlays are best-effort -- if one fails (e.g. missing font), the frame
is displayed without it.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import SnapshotMeta

if TYPE_CHECKING:
    from frames2py.core.engine import Engine


def _put_text(
    frame: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Draw text onto *frame* using OpenCV.  No-op if cv2 is unavailable."""
    try:
        import cv2

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    except ImportError:
        pass


class FPSOverlay:
    """Displays a rolling FPS counter in the top-left corner.

    The FPS value is computed from wall-clock time between successive
    ``draw`` calls (viewer-side FPS, not camera FPS).
    """

    __slots__ = ("_last_time", "_fps", "_alpha")

    def __init__(self, smoothing: float = 0.9) -> None:
        self._last_time: float = 0.0
        self._fps: float = 0.0
        self._alpha = smoothing

    def draw(self, frame: NDArray[np.uint8], meta: SnapshotMeta) -> None:
        now = time.monotonic()
        if self._last_time > 0:
            dt = now - self._last_time
            if dt > 0:
                instant = 1.0 / dt
                self._fps = self._alpha * self._fps + (1 - self._alpha) * instant
        self._last_time = now
        _put_text(frame, f"FPS: {self._fps:.1f}", 10, 25, color=(0, 255, 0))


class TimestampOverlay:
    """Displays the latest event timestamp in the top-right corner.

    Timestamps are shown in seconds with microsecond precision.
    """

    def draw(self, frame: NDArray[np.uint8], meta: SnapshotMeta) -> None:
        t_sec = meta.timestamp / 1_000_000.0
        text = f"t={t_sec:.6f}s"
        h, w = frame.shape[:2]
        # Approximate text width: ~10px per character at scale 0.5
        x = max(w - len(text) * 10 - 10, 10)
        _put_text(frame, text, x, 25, color=(255, 255, 0))


class StatsOverlay:
    """Displays engine statistics: event rate, drop rate, buffer fill.

    Parameters:
        engine: Reference to the :class:`~frames2py.core.engine.Engine`
            to read stats from.
    """

    __slots__ = ("_engine", "_last_ingested", "_last_time", "_rate")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._last_ingested: int = 0
        self._last_time: float = 0.0
        self._rate: float = 0.0

    def draw(self, frame: NDArray[np.uint8], meta: SnapshotMeta) -> None:
        stats = self._engine.stats
        now = time.monotonic()
        if self._last_time > 0:
            dt = now - self._last_time
            if dt > 0:
                delta = stats.events_ingested - self._last_ingested
                self._rate = delta / dt
        self._last_ingested = stats.events_ingested
        self._last_time = now

        h = frame.shape[0]
        y_base = h - 60
        rate_str = f"{self._rate / 1e6:.2f}M" if self._rate > 1e6 else f"{self._rate:.0f}"
        lines = [
            f"Rate: {rate_str} ev/s",
            f"Drops: {stats.events_dropped}  Fill: {stats.buffer_fill_ratio:.0%}",
            f"Snaps: {stats.snapshots_published}",
        ]
        for i, line in enumerate(lines):
            _put_text(frame, line, 10, y_base + i * 20, color=(200, 200, 200))


@dataclasses.dataclass(slots=True)
class BBox:
    """A single bounding box for :class:`BBoxOverlay`.

    Attributes:
        x1: Left edge (pixels).
        y1: Top edge (pixels).
        x2: Right edge (pixels).
        y2: Bottom edge (pixels).
        label: Optional text label.
        color: BGR color tuple.
        confidence: Optional confidence score in ``[0, 1]``.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    color: tuple[int, int, int] = (0, 255, 0)
    confidence: float = 1.0


class BBoxOverlay:
    """Draws bounding boxes from external detections.

    Boxes are pushed via :meth:`update` (thread-safe) and drawn on the
    next viewer frame.  Stale boxes (one frame old) are acceptable --
    this is a best-effort overlay for human eyes.

    Example::

        bbox_overlay = BBoxOverlay()
        viewer.add_overlay(bbox_overlay)
        # From inference thread:
        bbox_overlay.update([BBox(100, 50, 300, 200, label="car", confidence=0.92)])
    """

    __slots__ = ("_boxes", "_lock")

    def __init__(self) -> None:
        self._boxes: list[BBox] = []
        self._lock = threading.Lock()

    def update(self, boxes: list[BBox]) -> None:
        """Replace the current set of boxes (thread-safe)."""
        with self._lock:
            self._boxes = list(boxes)

    def draw(self, frame: NDArray[np.uint8], meta: SnapshotMeta) -> None:
        with self._lock:
            boxes = list(self._boxes)

        try:
            import cv2

            for b in boxes:
                cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), b.color, 2)
                if b.label:
                    text = b.label
                    if b.confidence < 1.0:
                        text += f" {b.confidence:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (b.x1, b.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        b.color,
                        1,
                        cv2.LINE_AA,
                    )
        except ImportError:
            pass
