"""Frames2Py: live, decoupled observation of event-camera state.

A producer feeds events to ``Engine.ingest()``, which accumulates them through a
kernel and publishes snapshots. Any number of consumers read them with
``Engine.snapshot()`` at their own pace; the producer never waits on a consumer.

``Accumulator`` is the same accumulation without publication, for synchronous use.

Quick start::

    import frames2py

    engine = frames2py.Engine((1280, 720), "event_count")
    engine.ingest(events)  # 1-D structured array of frames2py.EVENT_DTYPE
    snapshot = engine.snapshot()  # shared and read-only; snapshot.copy() for your own
    if snapshot is not None:
        frame, meta = snapshot.frame, snapshot.meta
"""

from frames2py._accumulator import Accumulator
from frames2py._engine import Engine
from frames2py._events import EVENT_DTYPE
from frames2py._types import EngineStats, SnapshotMeta
from frames2py.kernels import EventCount, ExpDecay, Polarity, TimeSurface, TimestampDecay

__all__ = [
    "Accumulator",
    "Engine",
    "EVENT_DTYPE",
    "EventCount",
    "Polarity",
    "TimeSurface",
    "ExpDecay",
    "TimestampDecay",
    "EngineStats",
    "SnapshotMeta",
]

__version__ = "1.0.0rc1"
