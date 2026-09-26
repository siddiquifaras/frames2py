"""Snapshot publication: the seam between the Engine and its consumers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from frames2py._types import Snapshot, SnapshotMeta

__all__ = ["SnapshotPublisher", "ImmutablePublisher", "Snapshot"]


@runtime_checkable
class SnapshotPublisher(Protocol):
    """Carries published frames from one writer to any number of readers.

    The writer calls ``begin_write()``, fills the returned buffer, then calls
    ``end_write(meta)``. Readers call ``read()``, which returns the latest complete
    publication, shared rather than copied, or ``None``.
    """

    def begin_write(self) -> NDArray[Any]:
        """Start a publication; return the buffer to fill."""
        ...

    def end_write(self, meta: SnapshotMeta) -> None:
        """Complete the publication started by ``begin_write()``."""
        ...

    def read(self) -> Snapshot | None:
        """The latest complete publication, or ``None``."""
        ...

    def reset(self) -> None:
        """Forget the published snapshot; ``read()`` returns ``None`` until the next."""
        ...


class ImmutablePublisher:
    """Publishes each snapshot as a new buffer that is never written again, in pure Python.

    ``begin_write()`` allocates a fresh buffer for the writer to fill. ``end_write(meta)``
    marks it read-only and stores it, with its metadata, as one ``Snapshot`` in a
    one-element list. ``read()`` loads that item and returns it: it never copies, never
    retries and never sees a buffer being written, and the frame and metadata it returns
    always belong together. One writer only.

    The handoff between threads is the list item's store and load. CPython documents single
    list-item reads and writes as atomic. That a reader which loads the new item also sees
    the frame written before it is CPython implementation behaviour: on free-threaded 3.14
    the store is a release store and the load a sequentially consistent one; with the GIL,
    the GIL orders them. It is not a Python language guarantee, and CPython itself locks the
    list during the store.

    Args:
        shape: Frame shape.
        dtype: Frame dtype.
    """

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._writing: NDArray[Any] | None = None
        self._slot: list[Snapshot | None] = [None]

    def begin_write(self) -> NDArray[Any]:
        self._writing = np.empty(self._shape, dtype=self._dtype)
        return self._writing

    def end_write(self, meta: SnapshotMeta) -> None:
        frame = self._writing
        if frame is None:
            raise RuntimeError("end_write() without begin_write()")
        self._writing = None
        frame.flags.writeable = False
        self._slot[0] = Snapshot(frame.view(), meta)

    def read(self) -> Snapshot | None:
        return self._slot[0]

    def reset(self) -> None:
        self._slot[0] = None
