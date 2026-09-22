"""HDF5 event file adapter.

Reads event data from HDF5 files in either the *compound-dataset*
layout (single dataset with structured dtype) or the *separate-dataset*
layout (``/events/t``, ``/events/x``, ``/events/y``, ``/events/p``).

Requires ``h5py``: ``pip install frames2py[adapter-h5]``
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from frames2py.core.types import EVENT_DTYPE, BatchMeta, EventBatch


def from_h5(
    path: str,
    chunk_size: int = 50_000,
    dataset: str = "events",
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Yield event batches from an HDF5 file.

    Supports two common layouts:

    1. **Compound dataset** -- a single dataset at *dataset* (e.g.
       ``"/events"``) with a structured dtype containing fields
       ``t``, ``x``, ``y``, ``p``.
    2. **Separate datasets** -- the group at *dataset* contains child
       datasets ``t``, ``x``, ``y``, ``p`` of equal length.

    Parameters:
        path: Filesystem path to the ``.h5`` file.
        chunk_size: Number of events per yielded batch.
        dataset: HDF5 path to the dataset or group.

    Yields:
        ``(EventBatch, BatchMeta)`` tuples.  ``BatchMeta.monotonic``
        is set to ``True`` if timestamps are verified non-decreasing.
    """
    import h5py

    meta_template = BatchMeta(source="h5")

    with h5py.File(path, "r") as f:
        node = f[dataset]

        if isinstance(node, h5py.Dataset):
            yield from _read_compound(node, chunk_size, meta_template)
        elif isinstance(node, h5py.Group):
            yield from _read_separate(node, chunk_size, meta_template)
        else:
            raise ValueError(
                f"HDF5 node at {dataset!r} is neither a Dataset nor a Group"
            )


def _read_compound(
    ds: object,
    chunk_size: int,
    meta_template: BatchMeta,
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Read from a single compound dataset."""
    import h5py

    assert isinstance(ds, h5py.Dataset)
    n = ds.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        raw = ds[start:end]

        batch = np.empty(end - start, dtype=EVENT_DTYPE)
        batch["t"] = raw["t"]
        batch["x"] = raw["x"]
        batch["y"] = raw["y"]
        batch["p"] = raw["p"]

        t = batch["t"]
        monotonic = len(t) <= 1 or bool(np.all(t[1:] >= t[:-1]))
        meta = BatchMeta(
            monotonic=monotonic,
            source=meta_template.source,
            sensor_size=meta_template.sensor_size,
        )
        yield batch, meta


def _read_separate(
    group: object,
    chunk_size: int,
    meta_template: BatchMeta,
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Read from separate t/x/y/p datasets under a group."""
    import h5py

    assert isinstance(group, h5py.Group)
    t_ds = group["t"]
    x_ds = group["x"]
    y_ds = group["y"]
    p_ds = group["p"]
    n = t_ds.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = np.empty(end - start, dtype=EVENT_DTYPE)
        batch["t"] = t_ds[start:end]
        batch["x"] = x_ds[start:end]
        batch["y"] = y_ds[start:end]
        batch["p"] = p_ds[start:end]

        t = batch["t"]
        monotonic = len(t) <= 1 or bool(np.all(t[1:] >= t[:-1]))
        meta = BatchMeta(
            monotonic=monotonic,
            source=meta_template.source,
            sensor_size=meta_template.sensor_size,
        )
        yield batch, meta


def to_h5(
    path: str,
    events: EventBatch,
    dataset: str = "events",
    compression: str | None = "gzip",
) -> None:
    """Write an event array to an HDF5 file.

    Creates a single compound dataset at *dataset*.

    Parameters:
        path: Output ``.h5`` file path.
        events: Structured array with dtype :data:`EVENT_DTYPE`.
        dataset: HDF5 dataset path.
        compression: HDF5 compression filter (``"gzip"``, ``"lzf"``,
            or ``None``).
    """
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset(
            dataset,
            data=events,
            compression=compression,
            chunks=True,
        )
