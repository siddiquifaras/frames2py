"""The event layout and structural validation of event arrays."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

EVENT_DTYPE: Final = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])
"""One event: ``t`` µs, ``x`` column, ``y`` row, ``p`` polarity (0 is OFF, anything else ON).
13 bytes, little-endian, no padding."""

TIMESTAMP_LIMIT: Final = 2**63
"""Calls containing any event with ``t`` at or above this are rejected whole."""

REQUIRED_FIELDS: Final = ("t", "x", "y", "p")

EventArray = NDArray[np.void]


def validate(events: object) -> EventArray:
    """Check the structure of an event array without looking at any value.

    Accepts a 1-D, C-contiguous structured array whose fields ``t``, ``x``, ``y``
    and ``p`` have exactly their ``EVENT_DTYPE`` dtypes, byte order included. Other
    fields are ignored. Anything else raises ``TypeError``.
    """
    if not isinstance(events, np.ndarray):
        raise TypeError(f"events must be a numpy structured array, got {type(events).__name__}")
    fields: Any = events.dtype.fields
    if fields is None:
        raise TypeError(f"events must be a structured array, got dtype {events.dtype}")
    for name in REQUIRED_FIELDS:
        if name not in fields:
            raise TypeError(f"events has no {name!r} field")
        expected = EVENT_DTYPE.fields[name][0]  # type: ignore[index]
        if fields[name][0] != expected:
            raise TypeError(f"field {name!r} must be {expected.str}, got {fields[name][0].str}")
    if events.ndim != 1:
        raise TypeError(f"events must be 1-D, got shape {events.shape}")
    if not events.flags.c_contiguous:
        raise TypeError("events must be C-contiguous")
    return events
