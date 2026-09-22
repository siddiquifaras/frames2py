"""frames2py -- Non-blocking event-to-frame accumulation engine.

A formalized, lock-free engine for real-time event camera visualization
that **never blocks the processing pipeline**.  Designed as the "PyTorch
of event systems" -- composable, backend-agnostic, and open-source.

Quick start::

    import frames2py

    engine = frames2py.Engine(sensor_size=(1280, 720), kernel="event_count")
    engine.ingest(events)  # non-blocking
    result = engine.latest_snapshot()
    if result is not None:
        frame, meta = result

Architecture:
    - **Plane A** -- Ingest + Accumulate (``Engine``)
    - **Plane B** -- Snapshot Bridge (seqlock, double-buffered)
    - **Plane C** -- Async Consumers (``Viewer``, ``Recorder``, ``Telemetry``)
"""

from frames2py.consumers.recorder import Recorder
from frames2py.consumers.telemetry import Telemetry, TelemetrySample
from frames2py.consumers.viewer import Viewer
from frames2py.core.engine import Engine
from frames2py.core.types import (
    EVENT_DTYPE,
    BatchMeta,
    EngineStats,
    EventBatch,
    FrameView,
    KernelState,
    OverflowPolicy,
    SnapshotMeta,
)
from frames2py.kernels.base import Kernel

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "Viewer",
    "Recorder",
    "Telemetry",
    "TelemetrySample",
    "EVENT_DTYPE",
    "BatchMeta",
    "EngineStats",
    "EventBatch",
    "FrameView",
    "KernelState",
    "OverflowPolicy",
    "SnapshotMeta",
    "Kernel",
    "__version__",
]
