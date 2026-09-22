"""Overflow accounting for the chunked ring buffer.

Tracks monotonically non-decreasing counters for events ingested, events
dropped, and chunks dropped.  These counters are always maintained --
even when no :class:`~frames2py.consumers.telemetry.Telemetry` consumer is
attached -- so that any caller can inspect system health at any time.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class OverflowCounters:
    """Monotonic counters for ring-buffer overflow accounting.

    All fields are ``int`` and only ever increase (never reset to zero
    except on an explicit :meth:`Engine.reset` call).

    Attributes:
        events_ingested: Total events successfully written into the ring
            buffer since construction or last reset.
        events_dropped: Total events lost because the ring buffer was full
            and the overflow policy evicted chunks.  This is the **exact**
            sum of event counts in each evicted chunk (not
            ``chunks_dropped * chunk_size``, because the trailing chunk in
            a batch may be partial).
        chunks_dropped: Number of whole chunk slots evicted from the ring
            buffer due to overflow.
    """

    events_ingested: int = 0
    events_dropped: int = 0
    chunks_dropped: int = 0

    def record_ingest(self, n_events: int) -> None:
        """Record *n_events* successfully written to the ring buffer."""
        self.events_ingested += n_events

    def record_drop(self, n_events: int, n_chunks: int = 1) -> None:
        """Record a chunk eviction of *n_events* across *n_chunks* slots."""
        self.events_dropped += n_events
        self.chunks_dropped += n_chunks

    def reset(self) -> None:
        """Zero all counters.  Called by :meth:`Engine.reset`."""
        self.events_ingested = 0
        self.events_dropped = 0
        self.chunks_dropped = 0
