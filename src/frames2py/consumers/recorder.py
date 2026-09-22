"""Snapshot recorder consumer.

Writes published snapshots to an MP4 video file using OpenCV's
``VideoWriter``.  Runs in a daemon thread with the same polling pattern
as the :class:`~frames2py.consumers.viewer.Viewer`.

If the recorder falls behind (disk I/O slower than snapshot rate), it
**skips** frames -- it never asks the engine to slow down.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from frames2py.core.engine import Engine


class Recorder:
    """Records engine snapshots to an MP4 file.

    Parameters:
        engine: The :class:`~frames2py.core.engine.Engine` to poll.
        output_path: Filesystem path for the output video
            (e.g. ``"recording.mp4"``).
        fps: Recording frame rate.  Determines both the polling interval
            and the MP4 metadata frame rate.
        codec: FourCC codec string (default ``"mp4v"``).

    Example::

        rec = Recorder(engine, "out.mp4", fps=30).start()
        # ... run pipeline ...
        rec.stop()  # file is finalised
    """

    def __init__(
        self,
        engine: Engine,
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        self._engine = engine
        self._output_path = output_path
        self._fps = fps
        self._codec = codec

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames_written = 0

    def start(self) -> Recorder:
        """Start the recorder daemon thread.  Returns ``self`` for chaining."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="frames2py-recorder",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the recorder to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    @property
    def frames_written(self) -> int:
        """Number of frames written to disk."""
        return self._frames_written

    # ------------------------------------------------------------------
    # Main recorder loop (runs in daemon thread)
    # ------------------------------------------------------------------
    def _run(self) -> None:
        import cv2

        w, h = self._engine.sensor_size
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (w, h))

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
                    time.sleep(interval)
                    continue

                # Normalise float frame → uint8 BGR for VideoWriter.
                fmin, fmax = float(frame.min()), float(frame.max())
                if fmax - fmin < 1e-8:
                    gray = np.zeros(frame.shape[:2], dtype=np.uint8)
                else:
                    gray = ((frame - fmin) / (fmax - fmin) * 255).clip(0, 255).astype(
                        np.uint8
                    )
                bgr = np.stack([gray, gray, gray], axis=-1)

                writer.write(bgr)
                last_seq = meta.seq
                self._frames_written += 1
                time.sleep(interval)
        finally:
            writer.release()
