"""OpenCV display backend.

Renders frames via ``cv2.imshow`` in the **viewer thread** (never the
user's ingest thread).  The ``opencv-python`` dependency is lazily
imported so the core library remains installable without it.

Install: ``pip install frames2py[viewer-opencv]``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class OpenCVRenderer:
    """Display backend using OpenCV's ``imshow``/``waitKey`` loop.

    This class is instantiated by the :class:`~frames2py.consumers.viewer.Viewer`
    and runs entirely within the viewer's daemon thread.
    """

    __slots__ = ("_window_name",)

    def __init__(self) -> None:
        self._window_name: str = ""

    def init(self, width: int, height: int, window_name: str) -> None:
        """Create a named OpenCV window.

        Parameters:
            width: Frame width in pixels.
            height: Frame height in pixels.
            window_name: Window title.
        """
        import cv2

        self._window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    def show(self, frame: NDArray[np.uint8]) -> bool:
        """Display *frame* via ``cv2.imshow``.

        Returns ``False`` if the user pressed 'q' or closed the window.
        """
        import cv2

        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def destroy(self) -> None:
        """Destroy the OpenCV window."""
        try:
            import cv2

            cv2.destroyWindow(self._window_name)
        except Exception:
            pass


class HeadlessRenderer:
    """No-op display backend for testing and recording-only pipelines.

    Satisfies the :class:`~frames2py.display.base.Renderer` protocol
    without any GUI dependency.
    """

    def init(self, width: int, height: int, window_name: str) -> None:
        pass

    def show(self, frame: NDArray[np.uint8]) -> bool:
        return True

    def destroy(self) -> None:
        pass
