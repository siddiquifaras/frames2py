"""Snapshot publication: the seam between the Engine and its consumers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from frames2py._types import SnapshotMeta

__all__ = ["SnapshotPublisher", "SeqlockPublisher"]


@runtime_checkable
class SnapshotPublisher(Protocol):
    """Carries published frames from one writer to any number of readers.

    The writer calls ``begin_write()``, fills the returned buffer, then calls
    ``end_write(meta)``. Readers call ``read()``, which returns a copy of the latest
    complete snapshot or ``None``.
    """

    def begin_write(self) -> NDArray[Any]:
        """Start a publication; return the buffer to fill."""
        ...

    def end_write(self, meta: SnapshotMeta) -> None:
        """Complete the publication started by ``begin_write()``."""
        ...

    def read(self) -> tuple[NDArray[Any], SnapshotMeta] | None:
        """A copy of the latest complete snapshot and its metadata, or ``None``."""
        ...

    def reset(self) -> None:
        """Forget the published snapshot; ``read()`` returns ``None`` until the next."""
        ...


class SeqlockPublisher:
    """Two buffers and a sequence counter, in pure Python.

    One writer fills the buffer readers aren't using and then switches them over.
    A reader copies the latest buffer and retries until no write started or finished
    during its copy, so it never returns a torn frame. That reasoning relies on the
    CPython GIL: this publisher makes no guarantee on free-threaded builds. Retries are
    unbounded, so a reader can be delayed while writes keep arriving.

    Args:
        shape: Frame shape.
        dtype: Frame dtype.
    """

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
        self._buffers = (np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype))
        self._metas: list[SnapshotMeta | None] = [None, None]
        self._latest: int | None = None
        self._writing = 0
        self._version = 0  # odd while a write is in progress

    def begin_write(self) -> NDArray[Any]:
        self._writing = 0 if self._latest is None else 1 - self._latest
        self._version += 1
        return self._buffers[self._writing]

    def end_write(self, meta: SnapshotMeta) -> None:
        self._metas[self._writing] = meta
        self._latest = self._writing
        self._version += 1

    def read(self) -> tuple[NDArray[Any], SnapshotMeta] | None:
        while True:
            version = self._version
            latest = self._latest
            if latest is None:
                return None
            frame = self._buffers[latest].copy()
            meta = self._metas[latest]
            if self._version == version and meta is not None:
                return frame, meta

    def reset(self) -> None:
        self._latest = None
        self._metas = [None, None]
        self._version += 2
