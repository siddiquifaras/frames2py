"""Result types: snapshot metadata and Engine statistics."""

from __future__ import annotations

import dataclasses


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
