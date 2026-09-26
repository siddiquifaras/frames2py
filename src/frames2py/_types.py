"""Result types: snapshots, their metadata, and Engine statistics."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotMeta:
    """Metadata published with each snapshot.

    Attributes:
        watermark: The largest timestamp among accumulated in-bounds events at
            publication, or ``None`` if none have been accumulated since construction
            or the last ``reset()``.
        sequence: Publication number. It increases with every publication for the
            Engine's lifetime, across ``reset()``.
    """

    watermark: int | None
    sequence: int


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """One publication: a frame and its metadata, shared by every consumer that reads it.

    ``frame`` is the published array itself, not a copy, and is marked read-only. Frames2Py
    never writes a published frame again. The read-only flag is NumPy's: it stops accidental
    writes through ``frame``, not code that sets it back on the owning array, and libraries
    that ignore it (``torch.from_numpy``, for one) share the memory writably. A consumer that
    needs to modify the data, or hands it to such a library, uses ``copy()``.

    Attributes:
        frame: The published frame, read-only, ``(height, width[, channels])``.
        meta: The publication's metadata.
    """

    frame: NDArray[Any]
    meta: SnapshotMeta

    def copy(self, out: NDArray[Any] | None = None) -> NDArray[Any]:
        """A writable copy of ``frame``: a new array, or *out* filled and returned.

        *out* must be a writable, C-contiguous ``ndarray`` with exactly ``frame``'s shape and
        dtype. Otherwise this raises before writing anything: ``TypeError`` for a non-array
        or a different dtype, ``ValueError`` for a different shape, a non-C-contiguous array
        or a read-only one.
        """
        if out is None:
            return self.frame.copy()
        if not isinstance(out, np.ndarray):
            raise TypeError(f"out must be a numpy.ndarray, got {type(out).__name__}")
        if out.dtype != self.frame.dtype:
            raise TypeError(f"out has dtype {out.dtype}; the frame's is {self.frame.dtype}")
        if out.shape != self.frame.shape:
            raise ValueError(f"out has shape {out.shape}; the frame's is {self.frame.shape}")
        if not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
        if not out.flags.writeable:
            raise ValueError("out is read-only")
        np.copyto(out, self.frame)
        return out


@dataclasses.dataclass(frozen=True, slots=True)
class EngineStats:
    """A frozen diagnostic view of an Engine's counters.

    The fields are read one after another, not as one instant-consistent snapshot.

    Attributes:
        events_ingested: Events in accepted ``ingest()`` calls, out-of-bounds ones
            included.
        events_out_of_bounds: The subset of those rejected by the bounds check.
        snapshots_published: Publications since construction or the last ``reset()``.
        uptime_ns: Nanoseconds since the Engine was constructed. ``reset()`` doesn't
            change it.
    """

    events_ingested: int
    events_out_of_bounds: int
    snapshots_published: int
    uptime_ns: int
