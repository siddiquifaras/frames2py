"""Read events stored as four 1-D HDF5 datasets, through h5py and hdf5plugin.

::

    from frames2py.adapters import hdf5

    # DSEC: events/{t,x,y,p}, with t relative to the scalar dataset /t_offset
    with hdf5.open("events.h5", group="events", t_offset="/t_offset", sensor_size=(640, 480)) as reader:
        for events in reader:
            engine.ingest(events)

The schema: ``group`` holds datasets ``t``, ``x``, ``y`` and ``p``, each 1-D with the same
length, one element per event, in event order. ``t`` is in microseconds. ``t_offset``, an
int or the path of a scalar integer dataset, is added to every ``t`` when given, and only
then. Nothing is autodetected: other layouts, such as one compound dataset, are rejected.

Values are carried over exactly or refused, never clamped or reinterpreted:
- ``t`` and ``t_offset`` are integers, and ``t + t_offset`` must lie in ``[0, 2**63)``;
- ``x`` and ``y`` are integers in ``[0, 65535]``;
- ``p`` is a bool (``True`` becomes 1) or an integer in ``[0, 255]``, copied as it is. The
  core reads 0 as OFF and anything else as ON, so a ``-1``/``+1`` polarity convention is
  refused rather than guessed at.
A value outside its range raises ``ValueError`` when iteration reaches its batch.

The file has no geometry field: ``sensor_size`` is the explicit value, or ``None``.

Compressed datasets are read through the filters h5py and hdf5plugin provide. A dataset
whose mandatory filter neither has raises ``ValueError`` on ``open()``; one whose optional
filter is missing (Blosc is usually stored as optional) raises ``ValueError`` when iteration
reads a chunk that needs it. Needs the ``frames2py[hdf5]`` extra.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator
from typing import Any, Final

import numpy as np

from frames2py._events import EVENT_DTYPE, TIMESTAMP_LIMIT, EventArray
from frames2py.adapters._reader import (
    PathArg,
    Reader,
    check_batch_size,
    check_readable,
    check_sensor_size,
    require,
)

__all__ = ["open"]

READ_EVENTS: Final = 1 << 20
"""Events read from each dataset per step."""

FIELDS: Final = ("t", "x", "y", "p")
_RANGES: Final = {"x": (0, 0xFFFF), "y": (0, 0xFFFF), "p": (0, 0xFF)}


def open(
    path: PathArg,
    *,
    group: str,
    t_offset: int | str | None = None,
    sensor_size: tuple[int, int] | None = None,
    batch_size: int | None = None,
) -> Reader:
    """Open an HDF5 file for one pass over the events in *group*.

    Args:
        path: The HDF5 file.
        group: Path of the group holding the ``t``, ``x``, ``y`` and ``p`` datasets.
        t_offset: Microseconds added to every ``t``: an int, or the path of a scalar
            integer dataset in the file. ``None`` adds nothing.
        sensor_size: ``(width, height)``, reported as ``reader.sensor_size``.
        batch_size: Events per yielded array (the last may have fewer), or ``None`` for
            steps of ``READ_EVENTS`` events.

    Raises:
        ImportError: h5py or hdf5plugin is not installed (``frames2py[hdf5]``).
        FileNotFoundError: *path* doesn't exist.
        OSError: *path* is a directory or can't be read.
        TypeError: *group* is not a str, or *t_offset* not an int, str or ``None``.
        ValueError: the file isn't HDF5, the group or its datasets don't follow the
            schema, the ``t_offset`` dataset isn't a scalar integer, or a dataset's
            mandatory filter isn't available.
    """
    explicit = check_sensor_size(sensor_size)
    size = check_batch_size(batch_size)
    if not isinstance(group, str):
        raise TypeError(f"group must be a str, got {type(group).__name__}")
    if t_offset is not None and not isinstance(t_offset, str):
        if isinstance(t_offset, bool):
            raise TypeError("t_offset must be an int, a dataset path or None, got a bool")
        t_offset = operator.index(t_offset)
    h5py = require("h5py", "hdf5")
    require("hdf5plugin", "hdf5")
    name = check_readable(path)
    try:
        handle = h5py.File(name, "r")
    except OSError as exc:
        raise ValueError(f"h5py could not open {name!r} as HDF5: {exc}") from exc
    try:
        datasets = _datasets(h5py, handle, group)
        offset = _offset(h5py, handle, t_offset)
    except BaseException:
        handle.close()
        raise
    return Reader(lambda: _batches(datasets, offset), handle.close, explicit, size)


def _datasets(h5py: Any, handle: Any, group: str) -> dict[str, Any]:
    node = handle.get(group)
    if not isinstance(node, h5py.Group):
        raise ValueError(f"{group!r} is not a group in the file" if node is None else f"{group!r} is not a group")
    datasets = {}
    for field in FIELDS:
        dataset = node.get(field)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"group {group!r} has no dataset {field!r}")
        if dataset.ndim != 1:
            raise ValueError(f"dataset {dataset.name!r} must be 1-D, got shape {dataset.shape}")
        kinds = "iub" if field == "p" else "iu"
        if dataset.dtype.kind not in kinds:
            raise ValueError(f"dataset {dataset.name!r} has dtype {dataset.dtype}; expected {'a bool or ' if field == 'p' else ''}an integer")
        _check_filters(h5py, dataset)
        datasets[field] = dataset
    lengths = {field: len(ds) for field, ds in datasets.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"datasets in {group!r} have different lengths: {lengths}")
    return datasets


def _check_filters(h5py: Any, dataset: Any) -> None:
    plist = dataset.id.get_create_plist()
    for i in range(plist.get_nfilters()):
        filter_id, flags = plist.get_filter(i)[:2]
        if not flags & h5py.h5z.FLAG_OPTIONAL and not h5py.h5z.filter_avail(filter_id):
            raise ValueError(
                f"dataset {dataset.name!r} needs HDF5 filter {filter_id}, which neither h5py nor hdf5plugin provides"
            )


def _offset(h5py: Any, handle: Any, t_offset: int | str | None) -> int:
    if t_offset is None:
        return 0
    if isinstance(t_offset, int):
        return t_offset
    dataset = handle.get(t_offset)
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != () or dataset.dtype.kind not in "iu":
        raise ValueError(f"t_offset {t_offset!r} is not a scalar integer dataset in the file")
    return int(dataset[()])


def _batches(datasets: dict[str, Any], offset: int) -> Iterator[EventArray]:
    total = len(datasets["t"])
    for start in range(0, total, READ_EVENTS):
        stop = min(start + READ_EVENTS, total)
        try:
            columns = {field: ds[start:stop] for field, ds in datasets.items()}
        except OSError as exc:
            raise ValueError(f"h5py failed reading events {start}-{stop} (a filter may be missing): {exc}") from exc
        yield _to_events(columns, offset)


def _to_events(columns: dict[str, np.ndarray], offset: int) -> EventArray:
    t = columns["t"]
    lowest, highest = _bounds(t)
    if lowest + offset < 0 or highest + offset >= TIMESTAMP_LIMIT:
        low, high = int(t.min()) + offset, int(t.max()) + offset
        if low < 0 or high >= TIMESTAMP_LIMIT:
            raise ValueError(f"t + t_offset spans [{low}, {high}], outside [0, 2**63); timestamps are never clamped")
    for field, (lo, hi) in _RANGES.items():
        values = columns[field]
        if values.dtype.kind == "b":
            continue
        dtype_lo, dtype_hi = _bounds(values)
        if dtype_lo < lo or dtype_hi > hi:
            if int(values.min()) < lo or int(values.max()) > hi:
                raise ValueError(f"{field} has values outside [{lo}, {hi}]")
    events = np.empty(len(t), dtype=EVENT_DTYPE)
    events["t"] = t  # a negative t becomes t + 2**64 here, and the offset brings it back
    if offset:
        stamps = events["t"]
        stamps += np.uint64(offset % (1 << 64))  # modulo 2**64; the true sums are in [0, 2**63)
    events["x"] = columns["x"]
    events["y"] = columns["y"]
    events["p"] = columns["p"]
    return events


def _bounds(values: np.ndarray) -> tuple[int, int]:
    """The range the array's integer dtype can hold; its values need no check within it."""
    info = np.iinfo(values.dtype)
    return int(info.min), int(info.max)
