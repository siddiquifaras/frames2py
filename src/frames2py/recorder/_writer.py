"""The HDF5 event recorder: ``open()`` and the ``Recorder`` it returns."""

from __future__ import annotations

import contextlib
import errno
import importlib
import operator
import os
import secrets
from types import ModuleType, TracebackType
from typing import Any, Final

import numpy as np

from frames2py import EVENT_DTYPE
from frames2py._events import TIMESTAMP_LIMIT, validate
from frames2py.adapters.hdf5 import FORMAT_VERSION, FORMAT_VERSION_ATTRIBUTE

CHUNK_EVENTS: Final = 65_536
"""Events per HDF5 chunk. Each chunk is written to the file once, whole."""

COMPRESSIONS: Final = (None, "gzip", "blosc")
FIELDS: Final = ("t", "x", "y", "p")
GZIP_LEVEL: Final = 4


def open(
    path: str | os.PathLike[str],
    *,
    sensor_size: tuple[int, int],
    group: str = "events",
    compression: str | None = "blosc",
    overwrite: bool = False,
) -> Recorder:
    """Start a recording at *path*; events go in with ``write()``, and ``close()`` finishes it.

    The recording is written to a temporary file next to *path* and moved onto *path* only
    once it is complete.

    Args:
        path: The recording to create.
        sensor_size: ``(width, height)``, stored as the ``sensor_width`` and ``sensor_height``
            attributes.
        group: The HDF5 group that holds the datasets.
        compression: ``"blosc"`` (LZ4 through hdf5plugin), ``"gzip"`` or ``None``.
        overwrite: Replace an existing file at *path* when the recording is complete.
            Otherwise an existing file is an error.

    Raises:
        ImportError: h5py or hdf5plugin is not installed (``frames2py[recorder]``).
        TypeError: *path* is not a str or path, *group* not a str, *overwrite* not a bool,
            or *sensor_size* doesn't hold ints.
        ValueError: *sensor_size* is not a pair of positive ints, *compression* is not one
            of the three, or *group* is not a valid group name.
        FileExistsError: *path* exists and *overwrite* is false.
        IsADirectoryError: *path* is a directory.
        FileNotFoundError: *path*'s directory doesn't exist.
        PermissionError: the directory can't be written.
    """
    width, height = _check_sensor_size(sensor_size)
    if not isinstance(group, str):
        raise TypeError(f"group must be a str, got {type(group).__name__}")
    if group == "":
        raise ValueError("group must name a group; use '/' for the root")
    if compression is not None and (not isinstance(compression, str) or compression not in COMPRESSIONS):
        raise ValueError(f"compression must be one of {COMPRESSIONS}, got {compression!r}")
    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be a bool, got {type(overwrite).__name__}")
    h5py = _require("h5py")
    hdf5plugin = _require("hdf5plugin")
    target = _target(path, overwrite)
    directory, name = os.path.split(target)
    temporary = os.path.join(directory, f".{name}.{secrets.token_hex(4)}.partial")
    handle = h5py.File(temporary, "x")
    try:
        try:
            node = handle["/"] if group.strip("/") == "" else handle.create_group(group)
        except ValueError as exc:
            raise ValueError(f"cannot create group {group!r}: {exc}") from exc
        filters = _filters(compression, hdf5plugin)
        datasets = [
            node.create_dataset(
                field,
                shape=(0,),
                maxshape=(None,),
                dtype=EVENT_DTYPE.fields[field][0],  # type: ignore[index]
                chunks=(CHUNK_EVENTS,),
                track_times=False,
                **filters,
            )
            for field in FIELDS
        ]
        node.attrs["sensor_width"] = np.int64(width)
        node.attrs["sensor_height"] = np.int64(height)
        node.attrs[FORMAT_VERSION_ATTRIBUTE] = np.int64(FORMAT_VERSION)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return Recorder(handle, datasets, temporary, target, overwrite)


class Recorder:
    """An open recording. ``write()`` adds events; ``close()`` or leaving the ``with`` block
    finishes it.

    ``write()`` runs on the caller's thread, compression and file writes included; the
    recorder starts no thread. Events are kept in a buffer of at most one chunk until a
    chunk is full, so a file's bytes depend only on the events written, not on how they were
    split across calls.

    ``close()`` writes the buffered events, closes the file and moves it onto the target
    path. Leaving the ``with`` block through an exception, ``KeyboardInterrupt`` included,
    closes it the same way: the recording then holds every event of every completed
    ``write()`` call, plus a prefix, possibly empty, of the events of a call the exception
    interrupted. A process that is killed or loses power leaves only the temporary file,
    which is not a valid recording.
    """

    def __init__(self, handle: Any, datasets: list[Any], temporary: str, target: str, overwrite: bool) -> None:
        self._handle = handle
        self._datasets = datasets
        self._temporary = temporary
        self._target = target
        self._overwrite = overwrite
        self._buffer = np.empty(CHUNK_EVENTS, dtype=EVENT_DTYPE)
        self._counts = (0, 0)
        """(events in the file, events in the buffer), assigned as one value so that an
        exception between two statements never leaves the two out of step."""
        self._closed = False

    def write(self, events: object) -> None:
        """Record one call's events, in order, exactly as given.

        Raises ``TypeError`` for a malformed array and ``ValueError`` if any event has
        ``t >= 2**63`` (nothing of the call is recorded) or if the recorder is closed.
        """
        if self._closed:
            raise ValueError("the recorder is closed")
        checked = validate(events)
        total = len(checked)
        if total == 0:
            return
        if int(checked["t"].max()) >= TIMESTAMP_LIMIT:
            raise ValueError("an event has t >= 2**63; nothing of this call was recorded")
        start = 0
        buffered = self._counts[1]
        if buffered:
            start = min(CHUNK_EVENTS - buffered, total)
            self._fill(checked[:start])
            if self._counts[1] == CHUNK_EVENTS:
                self._commit(self._buffer)
        while total - start >= CHUNK_EVENTS:
            self._commit(checked[start : start + CHUNK_EVENTS])
            start += CHUNK_EVENTS
        if start < total:
            self._fill(checked[start:])

    def close(self) -> None:
        """Finish the recording and move it onto the target path. Closing twice is a no-op.

        Raises ``FileExistsError`` if ``overwrite`` is false and a file appeared at the
        target after ``open()``; the finished recording is then left at the temporary path
        the message names. If finishing the file fails, the temporary file is removed and
        the error raised.
        """
        if self._closed:
            return
        self._closed = True
        try:
            written, buffered = self._counts
            for dataset in self._datasets:
                if dataset.shape[0] != written:  # a commit an exception interrupted
                    dataset.resize((written,))
            if buffered:
                self._commit(self._buffer[:buffered])
            self._handle.close()
        except BaseException:
            with contextlib.suppress(Exception):
                self._handle.close()
            with contextlib.suppress(OSError):
                os.unlink(self._temporary)
            raise
        if not self._overwrite and os.path.lexists(self._target):
            raise FileExistsError(
                errno.EEXIST,
                f"a file appeared at the target while recording; the recording is at {self._temporary}",
                self._target,
            )
        os.replace(self._temporary, self._target)

    def __enter__(self) -> Recorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _fill(self, part: np.ndarray) -> None:
        written, buffered = self._counts
        end = buffered + len(part)
        for field in FIELDS:
            self._buffer[field][buffered:end] = part[field]
        self._counts = (written, end)

    def _commit(self, chunk: np.ndarray) -> None:
        """Append *chunk* to the file; it is recorded once the counts say so."""
        written = self._counts[0]
        end = written + len(chunk)
        for dataset in self._datasets:
            dataset.resize((end,))
        for field, dataset in zip(FIELDS, self._datasets):
            dataset[written:end] = chunk[field]
        self._counts = (end, 0)


def _check_sensor_size(sensor_size: Any) -> tuple[int, int]:
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


def _target(path: object, overwrite: bool) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be a str or os.PathLike, got {type(path).__name__}")
    target = os.fspath(path)
    if not isinstance(target, str):
        raise TypeError("path must be a str path, got bytes")
    target = os.path.abspath(target)
    if os.path.isdir(target):
        raise IsADirectoryError(errno.EISDIR, "the path is a directory", target)
    if os.path.lexists(target) and not overwrite:
        raise FileExistsError(errno.EEXIST, "the file exists; pass overwrite=True to replace it", target)
    directory = os.path.dirname(target)
    if not os.path.isdir(directory):
        raise FileNotFoundError(errno.ENOENT, "the directory doesn't exist", directory)
    return target


def _filters(compression: str | None, hdf5plugin: ModuleType) -> dict[str, Any]:
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": GZIP_LEVEL, "shuffle": True}
    if compression == "blosc":
        return dict(hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
    return {}


def _require(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"frames2py.recorder needs {module}; install it with: pip install 'frames2py[recorder]'"
        ) from exc

