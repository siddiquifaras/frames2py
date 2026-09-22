"""Display backend and overlay protocols.

Defines the abstract interfaces that viewer backends and frame overlays
must satisfy.  Concrete implementations live in sibling modules
(:mod:`cv_renderer`, :mod:`overlays`).

All display code is **optional** -- the core engine has zero imports
from this package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import SnapshotMeta


@runtime_checkable
class Renderer(Protocol):
    """Protocol for display backends (OpenCV, headless, etc.)."""

    def init(self, width: int, height: int, window_name: str) -> None:
        """Create the display window / context.

        Parameters:
            width: Frame width in pixels.
            height: Frame height in pixels.
            window_name: Title for the window.
        """
        ...

    def show(self, frame: NDArray[np.uint8]) -> bool:
        """Display a single colorised frame.

        Parameters:
            frame: ``(H, W, 3)`` BGR uint8 image.

        Returns:
            ``False`` if the window was closed by the user, ``True``
            otherwise.
        """
        ...

    def destroy(self) -> None:
        """Release all display resources."""
        ...


@runtime_checkable
class Overlay(Protocol):
    """Protocol for best-effort frame overlays (FPS counter, bboxes, etc.).

    Overlays modify the colorised frame **in-place** before it is
    displayed.
    """

    def draw(
        self,
        frame: NDArray[np.uint8],
        meta: SnapshotMeta,
    ) -> None:
        """Draw the overlay onto *frame*.

        Parameters:
            frame: ``(H, W, 3)`` BGR uint8 image.  Modify in-place.
            meta: Snapshot metadata for the current frame.
        """
        ...
