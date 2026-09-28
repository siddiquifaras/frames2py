"""The reader every file adapter's ``open()`` returns, and the checks the adapters share.

A reader is single-pass: it yields ``EVENT_DTYPE`` arrays once, in source order, and
then it is spent. It owns the open file or decoder and releases it on ``close()``, which
leaving a ``with`` block calls. It starts no threads and holds no Engine.
"""

from __future__ import annotations

import builtins
import importlib
import operator
import os
from collections.abc import Callable, Iterator
from types import ModuleType, TracebackType
from typing import BinaryIO

import numpy as np

from frames2py._events import EventArray

PathArg = str | os.PathLike[str]
Size = tuple[int, int]


class Reader:
    """Iterate once over a recording's events.

    Iteration yields 1-D, C-contiguous ``EVENT_DTYPE`` arrays in source order. With
    ``batch_size=None`` each array follows the decoder's own boundaries; with
    ``batch_size=N`` every array has N events except the last. No array is empty.

    Each array is new or a view of a new array; no two share memory, and none shares
    memory with the file or the backend.

    A second ``iter()`` raises ``RuntimeError``. Iterating a closed reader, or calling
    ``next()`` on its iterator after ``close()``, raises ``ValueError``.
    """

    def __init__(
        self,
        batches: Callable[[], Iterator[EventArray]],
        close: Callable[[], None],
        sensor_size: Size | None,
        batch_size: int | None,
    ) -> None:
        self._batches = batches
        self._close = close
        self._sensor_size = sensor_size
        self._batch_size = batch_size
        self._closed = False
        self._iterated = False
        self._active: Iterator[EventArray] | None = None

    @property
    def sensor_size(self) -> Size | None:
        """``(width, height)``, from the source or the explicit ``sensor_size``."""
        return self._sensor_size

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release the file or decoder. Closing twice is a no-op."""
        if self._closed:
            return
        self._closed = True
        active, self._active = self._active, None
        try:
            if active is not None:
                close_generator = getattr(active, "close", None)
                if close_generator is not None:
                    close_generator()
        finally:
            self._close()

    def __enter__(self) -> Reader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[EventArray]:
        if self._closed:
            raise ValueError("the reader is closed")
        if self._iterated:
            raise RuntimeError("a reader is single-pass; open the file again to read it again")
        self._iterated = True
        self._active = _rebatch(self._batches(), self._batch_size)
        return _Iteration(self)

    def _next(self) -> EventArray:
        if self._closed or self._active is None:
            raise ValueError("the reader is closed")
        return next(self._active)


class _Iteration:
    """The reader's one iterator; it refuses to continue once the reader is closed."""

    def __init__(self, reader: Reader) -> None:
        self._reader = reader

    def __iter__(self) -> _Iteration:
        return self

    def __next__(self) -> EventArray:
        return self._reader._next()


def _rebatch(batches: Iterator[EventArray], size: int | None) -> Iterator[EventArray]:
    if size is None:
        for batch in batches:
            if len(batch):
                yield batch
        return
    pending: list[EventArray] = []
    count = 0
    for batch in batches:
        start, n = 0, len(batch)
        if count:
            need = size - count
            if n < need:
                pending.append(batch)
                count += n
                continue
            pending.append(batch[:need])
            yield np.concatenate(pending)
            pending, count, start = [], 0, need
        while n - start >= size:
            yield batch[start : start + size]
            start += size
        if start < n:
            pending, count = [batch[start:]], n - start
    if count:
        yield pending[0] if len(pending) == 1 else np.concatenate(pending)


def check_batch_size(batch_size: int | None) -> int | None:
    """``None``, or a positive int; ``TypeError`` for a non-integer, ``ValueError`` below 1."""
    if batch_size is None:
        return None
    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be an int or None, got a bool")
    size = operator.index(batch_size)
    if size < 1:
        raise ValueError(f"batch_size must be at least 1, got {size}")
    return size


def check_sensor_size(sensor_size: tuple[int, int] | None) -> Size | None:
    """``None``, or ``(width, height)`` as two positive ints."""
    if sensor_size is None:
        return None
    try:
        width, height = sensor_size
    except (TypeError, ValueError):
        raise ValueError(f"sensor_size must be (width, height), got {sensor_size!r}") from None
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("sensor_size must hold ints, got a bool")
    size = (operator.index(width), operator.index(height))
    if size[0] < 1 or size[1] < 1:
        raise ValueError(f"sensor_size must be positive, got {size}")
    return size


def merge_sensor_size(explicit: Size | None, source: Size | None, what: str) -> Size | None:
    """The source's geometry, which an explicit value must equal, else the explicit value."""
    if source is None:
        return explicit
    if explicit is not None and explicit != source:
        raise ValueError(f"sensor_size {explicit} conflicts with the {what} geometry {source}")
    return source


def open_binary(path: PathArg) -> BinaryIO:
    """Open *path* for reading: ``FileNotFoundError`` if it doesn't exist, ``OSError`` if
    it is a directory or can't be read."""
    return builtins.open(os.fspath(path), "rb")  # IsADirectoryError for a directory


def check_readable(path: PathArg) -> str:
    """The path as a string, after checking it is a readable file (as ``open_binary``)."""
    open_binary(path).close()
    return os.fspath(path)


def require(module: str, extra: str) -> ModuleType:
    """Import an adapter's optional dependency, or raise ``ImportError`` naming its extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"frames2py.adapters.{extra} needs {module}; install it with: pip install 'frames2py[{extra}]'"
        ) from exc
