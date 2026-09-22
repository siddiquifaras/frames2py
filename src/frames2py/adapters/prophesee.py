"""Prophesee Metavision SDK adapter.

Wraps ``metavision_core.event_io.EventsIterator`` to yield event batches
in the canonical :data:`EVENT_DTYPE` format.

Requires ``metavision-core``:
``pip install frames2py[adapter-prophesee]``

If Metavision is not installed, :func:`from_prophesee` raises an
``ImportError`` with a helpful message.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from frames2py.core.types import EVENT_DTYPE, BatchMeta, EventBatch


def from_prophesee(
    source: str,
    chunk_size: int = 10_000,
    delta_t: int | None = None,
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Yield event batches from a Prophesee camera or RAW file.

    Parameters:
        source: Device path (e.g. ``"/dev/prophesee0"``) or path to a
            Prophesee RAW/EVT2/EVT3 recording file.
        chunk_size: Target number of events per batch (used when
            *delta_t* is ``None``).
        delta_t: If set, yield events in fixed-duration windows of
            *delta_t* microseconds instead of fixed-count batches.

    Yields:
        ``(EventBatch, BatchMeta)`` tuples.  Prophesee events are
        guaranteed monotonic by the SDK, so ``meta.monotonic`` is
        ``True``.
    """
    try:
        from metavision_core.event_io import EventsIterator
    except ImportError as exc:
        raise ImportError(
            "Prophesee adapter requires metavision_core. "
            "Install via: pip install frames2py[adapter-prophesee]"
        ) from exc

    iterator_kwargs: dict = {}
    if delta_t is not None:
        iterator_kwargs["delta_t"] = delta_t
    else:
        iterator_kwargs["max_duration"] = None

    ev_it = EventsIterator(source, **iterator_kwargs)

    for raw_events in ev_it:
        if len(raw_events) == 0:
            continue

        batch = np.empty(len(raw_events), dtype=EVENT_DTYPE)
        batch["t"] = raw_events["t"]
        batch["x"] = raw_events["x"]
        batch["y"] = raw_events["y"]
        batch["p"] = raw_events["p"]

        meta = BatchMeta(
            monotonic=True,
            source="prophesee",
        )
        yield batch, meta
