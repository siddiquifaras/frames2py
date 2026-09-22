"""Chunk-based ring buffer with deterministic overflow accounting.

Implements the transport layer described in ``docs/spec/overflow.md``.

Design constraints (from invariants.md):
- All chunk arrays are **pre-allocated** at ``__init__`` time.
  No heap allocation on the write path.
- Overflow evicts **whole chunks** atomically (never individual events).
- Counters (``events_dropped``, ``chunks_dropped``) are exact -- not
  approximations.
- Single-producer, single-consumer.  No locks.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from frames2py.core.types import EVENT_DTYPE, EventBatch, OverflowPolicy


class ChunkedRingBuffer:
    """Fixed-capacity ring buffer that stores events in pre-allocated chunk slots.

    Parameters:
        capacity: Number of chunk slots.
        chunk_size: Maximum events per chunk slot.
        dtype: Structured dtype for events (default :data:`EVENT_DTYPE`).
        overflow_policy: Strategy when the buffer is full.
            ``DROP_OLDEST`` evicts the oldest unread chunk.
            ``DROP_NEWEST`` silently discards the incoming write.

    The buffer is a circular array of *capacity* slots, each capable of
    holding up to *chunk_size* events.  A parallel ``_counts`` array
    records how many events are actually stored in each slot (the last
    chunk in a batch may be partial).

    Complexity:
        - ``write``: *O(n)* where *n* = ``len(events)`` (one ``memcpy``
          per chunk-sized segment).
        - ``read_new``: *O(1)* -- returns a view/slice, no copy.
    """

    __slots__ = (
        "_slots",
        "_counts",
        "_capacity",
        "_chunk_size",
        "_policy",
        "_write_idx",
        "_read_idx",
        "_size",
        "_events_dropped",
        "_chunks_dropped",
    )

    def __init__(
        self,
        capacity: int = 64,
        chunk_size: int = 65_536,
        dtype: np.dtype = EVENT_DTYPE,  # type: ignore[assignment]
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        self._capacity = capacity
        self._chunk_size = chunk_size
        self._policy = overflow_policy

        # Pre-allocate all chunk slots up front -- no further allocation.
        self._slots: list[NDArray[np.void]] = [
            np.empty(chunk_size, dtype=dtype) for _ in range(capacity)
        ]
        self._counts = np.zeros(capacity, dtype=np.int64)

        self._write_idx = 0
        self._read_idx = 0
        self._size = 0  # occupied slots

        self._events_dropped = 0
        self._chunks_dropped = 0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def write(self, events: EventBatch) -> int:
        """Write *events* into the ring buffer.

        If *events* exceeds ``chunk_size``, the batch is split across
        multiple consecutive slots.

        Returns:
            Number of events **dropped** due to overflow (0 when no
            overflow occurs).
        """
        if len(events) == 0:
            return 0

        total_dropped = 0
        offset = 0
        n = len(events)

        while offset < n:
            end = min(offset + self._chunk_size, n)
            segment = events[offset:end]
            seg_len = len(segment)

            dropped = self._write_chunk(segment, seg_len)
            total_dropped += dropped
            offset = end

        return total_dropped

    def _write_chunk(self, segment: EventBatch, seg_len: int) -> int:
        """Write a single chunk-sized segment into the next slot.

        Returns the number of events dropped (from the evicted chunk, if
        any).
        """
        dropped = 0

        if self._policy is OverflowPolicy.DROP_OLDEST:
            if self._size == self._capacity:
                # Evict the oldest unread chunk.
                evicted_count = int(self._counts[self._read_idx])
                dropped = evicted_count
                self._events_dropped += evicted_count
                self._chunks_dropped += 1
                self._read_idx = (self._read_idx + 1) % self._capacity
                self._size -= 1

            slot = self._slots[self._write_idx]
            slot[:seg_len] = segment
            self._counts[self._write_idx] = seg_len
            self._write_idx = (self._write_idx + 1) % self._capacity
            self._size += 1

        else:
            # DROP_NEWEST: discard incoming if buffer is full.
            if self._size == self._capacity:
                dropped = seg_len
                self._events_dropped += seg_len
                self._chunks_dropped += 1
            else:
                slot = self._slots[self._write_idx]
                slot[:seg_len] = segment
                self._counts[self._write_idx] = seg_len
                self._write_idx = (self._write_idx + 1) % self._capacity
                self._size += 1

        return dropped

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def read_new(self) -> EventBatch | None:
        """Read and consume the next unread chunk.

        Returns:
            A **slice** (not copy) of the pre-allocated chunk array
            containing the stored events, or ``None`` if no new data is
            available.

        The returned array is valid until the slot is overwritten by a
        future ``write``.  Callers that need the data to persist must
        copy it.
        """
        if self._size == 0:
            return None

        idx = self._read_idx
        count = int(self._counts[idx])
        data = self._slots[idx][:count]
        self._read_idx = (self._read_idx + 1) % self._capacity
        self._size -= 1
        return data

    def iter_chunks(self) -> Iterator[EventBatch]:
        """Yield and consume all pending chunks as zero-copy views.

        Each yielded array is a **slice** of a pre-allocated slot -- no
        copy, no concatenation.  The view is valid until the slot is
        overwritten by a future ``write()``.  Since the engine calls
        ``iter_chunks`` synchronously inside ``ingest()``, the view
        remains valid for the entire kernel accumulate call.

        Complexity: O(k) where *k* is the number of pending chunks.
        """
        while self._size > 0:
            idx = self._read_idx
            count = int(self._counts[idx])
            self._read_idx = (self._read_idx + 1) % self._capacity
            self._size -= 1
            if count > 0:
                yield self._slots[idx][:count]

    def drain(self) -> EventBatch | None:
        """Read and concatenate **all** unread chunks into a single array.

        Returns:
            A single contiguous array of all pending events, or ``None``
            if the buffer is empty.

        Unlike :meth:`iter_chunks`, this performs a copy (via
        ``np.concatenate``) so the caller owns the result.

        Prefer :meth:`iter_chunks` on the hot path to avoid the
        concatenation allocation.
        """
        if self._size == 0:
            return None

        parts: list[NDArray[np.void]] = []
        for chunk in self.iter_chunks():
            parts.append(chunk)

        if not parts:
            return None
        if len(parts) == 1:
            return np.array(parts[0])
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fill_ratio(self) -> float:
        """Current buffer occupancy in ``[0.0, 1.0]``."""
        return self._size / self._capacity

    @property
    def total_dropped(self) -> int:
        """Total events dropped since construction or last reset."""
        return self._events_dropped

    @property
    def chunks_dropped(self) -> int:
        """Total chunk slots evicted since construction or last reset."""
        return self._chunks_dropped

    @property
    def capacity(self) -> int:
        """Number of chunk slots."""
        return self._capacity

    @property
    def chunk_size(self) -> int:
        """Maximum events per chunk slot."""
        return self._chunk_size

    @property
    def size(self) -> int:
        """Number of occupied chunk slots."""
        return self._size

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all buffered data and reset counters.

        Chunk arrays remain allocated (no re-allocation).
        """
        self._write_idx = 0
        self._read_idx = 0
        self._size = 0
        self._events_dropped = 0
        self._chunks_dropped = 0
        self._counts[:] = 0
