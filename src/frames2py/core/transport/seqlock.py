"""Sequence-lock protected double-buffered snapshot bridge.

Implements the publication protocol described in ``docs/spec/snapshot.md``.

The seqlock allows a single writer (the engine) to publish frame snapshots
that zero or more readers (viewer, recorder, telemetry) can poll without
any mutex on the hot path.

Protocol:
    1. Writer calls :meth:`begin_write` -- ``seq`` incremented to *odd*
       (write in progress).
    2. Writer fills the returned buffer and calls :meth:`end_write` --
       ``seq`` incremented to *even* (published, safe to read).
    3. Readers call :meth:`try_read` -- check ``seq`` is even, copy frame
       + metadata, check ``seq`` unchanged.  If ``seq`` changed during the
       copy (torn read), return ``None`` and let the consumer retry on its
       next poll cycle.

Guarantees:
    - If a reader observes ``seq`` as even and stable across its copy, the
      frame and metadata are mutually consistent.
    - ``seq`` is monotonically non-decreasing.  Readers may skip values but
      never observe regression.
    - The writer and readers **never** access the same underlying buffer
      simultaneously (double buffering).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import SnapshotMeta


class Seqlock:
    """Double-buffered seqlock for lock-free snapshot publication.

    Parameters:
        frame_shape: Shape of each frame buffer (e.g. ``(720, 1280)``
            or ``(720, 1280, 2)`` for multi-channel kernels).
        frame_dtype: NumPy dtype of the frame buffer (default ``float32``).
    """

    __slots__ = (
        "_buffers",
        "_meta",
        "_seq",
        "_write_buf_idx",
    )

    def __init__(
        self,
        frame_shape: tuple[int, ...],
        frame_dtype: np.dtype = np.dtype(np.float32),
    ) -> None:
        # Two pre-allocated frame buffers -- ping-pong style.
        self._buffers: tuple[NDArray[np.floating], NDArray[np.floating]] = (
            np.zeros(frame_shape, dtype=frame_dtype),
            np.zeros(frame_shape, dtype=frame_dtype),
        )
        self._meta: SnapshotMeta = SnapshotMeta()
        self._seq: int = 0  # even = published / idle
        self._write_buf_idx: int = 0  # toggles 0 ↔ 1

    # ------------------------------------------------------------------
    # Writer interface (single producer only)
    # ------------------------------------------------------------------
    def begin_write(self) -> NDArray[np.floating]:
        """Begin a write transaction.

        Increments ``seq`` to odd (signalling "write in progress") and
        returns the **inactive** buffer for the writer to fill.

        The caller must fill the returned buffer and then call
        :meth:`end_write`.
        """
        self._seq += 1  # odd → write in progress
        return self._buffers[self._write_buf_idx]

    def end_write(self, meta: SnapshotMeta) -> None:
        """Finish a write transaction.

        Stores *meta*, increments ``seq`` to even (signalling "published"),
        and toggles the active buffer so the next write uses the other
        slot.
        """
        self._meta = meta
        self._seq += 1  # even → published
        self._write_buf_idx ^= 1  # toggle 0 ↔ 1

    # ------------------------------------------------------------------
    # Reader interface (multiple concurrent consumers)
    # ------------------------------------------------------------------
    def try_read(self) -> tuple[NDArray[np.floating], SnapshotMeta] | None:
        """Attempt a consistent snapshot read.

        Returns:
            ``(frame_copy, meta)`` if a consistent read was achieved, or
            ``None`` if a write was in progress (torn read detected).

        The returned frame is a **copy** -- the caller owns it and may
        modify it freely (e.g. applying colormaps, overlays).
        """
        seq1 = self._seq

        # Seq must be even (no write in progress).
        if seq1 & 1:
            return None

        # Nothing published yet.
        if seq1 == 0 and self._meta.seq == 0 and self._meta.events_accumulated == 0:
            return None

        # Read from the buffer that the writer is NOT currently targeting.
        read_idx = self._write_buf_idx ^ 1
        frame_copy = self._buffers[read_idx].copy()
        meta = self._meta

        seq2 = self._seq
        if seq2 != seq1:
            # Writer mutated during our copy -- torn read.
            return None

        return frame_copy, meta

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def seq(self) -> int:
        """Current sequence number (even = idle, odd = write in progress)."""
        return self._seq

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Shape of each frame buffer."""
        return self._buffers[0].shape

    @property
    def frame_dtype(self) -> np.dtype:
        """Dtype of each frame buffer."""
        return self._buffers[0].dtype

    def reset(self) -> None:
        """Zero both buffers, reset seq to 0, clear metadata."""
        self._buffers[0][:] = 0
        self._buffers[1][:] = 0
        self._meta = SnapshotMeta()
        self._seq = 0
        self._write_buf_idx = 0
