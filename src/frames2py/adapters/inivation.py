"""iniVation dv-processing adapter.

Wraps ``dv_processing`` to yield event batches from iniVation cameras
(DAVIS346, DVXplorer, etc.) or AEDAT4 recordings in the canonical
:data:`EVENT_DTYPE` format.

Requires ``dv-processing``:
``pip install frames2py[adapter-inivation]``
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from frames2py.core.types import EVENT_DTYPE, BatchMeta, EventBatch


def from_inivation(
    source: str,
    chunk_size: int = 10_000,
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Yield event batches from an iniVation camera or AEDAT4 file.

    Parameters:
        source: Camera name (e.g. ``"DVXplorer_DXA00000"``) or path
            to an AEDAT4 file recorded with dv-processing.
        chunk_size: Maximum events per yielded batch.

    Yields:
        ``(EventBatch, BatchMeta)`` tuples.
    """
    try:
        import dv_processing as dv
    except ImportError as exc:
        raise ImportError(
            "iniVation adapter requires dv-processing. "
            "Install via: pip install frames2py[adapter-inivation]"
        ) from exc

    # Determine if source is a file or live camera
    if source.endswith(".aedat4") or source.endswith(".aedat"):
        reader = dv.io.MonoCameraRecording(source)
        sensor_size = (reader.getEventResolution().width,
                       reader.getEventResolution().height)

        while reader.isRunning():
            event_store = reader.getNextEventBatch()
            if event_store is None:
                break

            events_np = event_store.numpy()
            if len(events_np) == 0:
                continue

            for start in range(0, len(events_np), chunk_size):
                end = min(start + chunk_size, len(events_np))
                segment = events_np[start:end]

                batch = np.empty(len(segment), dtype=EVENT_DTYPE)
                batch["t"] = segment["timestamp"]
                batch["x"] = segment["x"]
                batch["y"] = segment["y"]
                batch["p"] = segment["polarity"]

                meta = BatchMeta(
                    monotonic=True,
                    source="inivation",
                    sensor_size=sensor_size,
                )
                yield batch, meta
    else:
        # Live camera
        camera = dv.io.CameraCapture(source)
        sensor_size = (camera.getEventResolution().width,
                       camera.getEventResolution().height)

        while camera.isRunning():
            event_store = camera.getNextEventBatch()
            if event_store is None:
                continue

            events_np = event_store.numpy()
            if len(events_np) == 0:
                continue

            for start in range(0, len(events_np), chunk_size):
                end = min(start + chunk_size, len(events_np))
                segment = events_np[start:end]

                batch = np.empty(len(segment), dtype=EVENT_DTYPE)
                batch["t"] = segment["timestamp"]
                batch["x"] = segment["x"]
                batch["y"] = segment["y"]
                batch["p"] = segment["polarity"]

                meta = BatchMeta(
                    monotonic=True,
                    source="inivation",
                    sensor_size=sensor_size,
                )
                yield batch, meta
