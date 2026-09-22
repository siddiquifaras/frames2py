"""Live camera or file playback with Prophesee or iniVation cameras.

Demonstrates:
- Using Prophesee (Metavision) or iniVation (dv-processing) adapters
- --source: device path or index for live camera, or file path for playback
- --backend: "prophesee" or "inivation"
- Graceful error handling when SDK is not installed
- OpenCV viewer for live display (falls back to headless if no cv2)

Requires (one of):
  pip install frames2py[adapter-prophesee]   # for Prophesee cameras/RAW files
  pip install frames2py[adapter-inivation]    # for iniVation cameras/AEDAT4 files

Run with:
  python examples/03_live_camera.py --backend prophesee --source /path/to/file.raw
  python examples/03_live_camera.py --backend inivation --source /path/to/file.aedat4
  python examples/03_live_camera.py --backend prophesee --source 0  # live camera
"""

from __future__ import annotations

import argparse
import sys
import time

from frames2py import Engine, Viewer


def get_event_stream(backend: str, source: str):
    """Get event stream iterator from the specified backend. Raises ImportError if SDK missing."""
    if backend == "prophesee":
        try:
            from frames2py.adapters.prophesee import from_prophesee
            return from_prophesee(source, chunk_size=10_000)
        except ImportError as e:
            raise ImportError(
                "Prophesee adapter requires metavision-core. "
                "Install: pip install frames2py[adapter-prophesee]"
            ) from e
    elif backend == "inivation":
        try:
            from frames2py.adapters.inivation import from_inivation
            return from_inivation(source, chunk_size=10_000)
        except ImportError as e:
            raise ImportError(
                "iniVation adapter requires dv-processing. "
                "Install: pip install frames2py[adapter-inivation]"
            ) from e
    else:
        raise ValueError(f"Unknown backend: {backend}")


def infer_sensor_size(batch, meta) -> tuple[int, int]:
    """Infer sensor size from batch and meta."""
    if meta.sensor_size is not None:
        return meta.sensor_size
    if len(batch) > 0:
        w = int(batch["x"].max()) + 1
        h = int(batch["y"].max()) + 1
        return (max(w, 64), max(h, 48))
    return (640, 480)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play events from Prophesee or iniVation camera/file."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera device (e.g. 0, /dev/video0) or file path (default: 0)",
    )
    parser.add_argument(
        "--backend",
        choices=["prophesee", "inivation"],
        required=True,
        help="Camera SDK: prophesee or inivation",
    )
    parser.add_argument(
        "--kernel",
        choices=["event_count", "polarity", "time_surface", "exp_decay"],
        default="event_count",
        help="Accumulation kernel (default: event_count)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use headless viewer (no display window)",
    )
    args = parser.parse_args()

    try:
        stream = get_event_stream(args.backend, args.source)
    except ImportError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error opening source {args.source!r}: {e}")
        return 1

    # Read first batch to infer sensor size
    print("Opening source...")
    first_batch, first_meta = None, None
    try:
        for batch, meta in stream:
            first_batch, first_meta = batch, meta
            break
    except Exception as e:
        print(f"Error: Could not read from source: {e}")
        return 1

    if first_batch is None or len(first_batch) == 0:
        print("Error: No events from source")
        return 1

    sensor_size = infer_sensor_size(first_batch, first_meta)
    print(f"Sensor size: {sensor_size[0]}x{sensor_size[1]}")
    print("Press Ctrl+C to stop")
    print()

    engine = Engine(sensor_size=sensor_size, kernel=args.kernel)
    viewer_backend = "headless" if args.headless else "opencv"
    try:
        viewer = Viewer(engine, backend=viewer_backend, fps=30, colormap="viridis")
    except ValueError:
        # OpenCV not available, fall back to headless
        viewer = Viewer(engine, backend="headless", fps=30)
    viewer.start()

    total_events = 0
    start = time.monotonic()

    try:
        engine.ingest(first_batch, first_meta)
        total_events += len(first_batch)
        for batch, meta in stream:
            engine.ingest(batch, meta)
            total_events += len(batch)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    elapsed = time.monotonic() - start
    engine.stop()
    viewer.stop()

    stats = engine.stats
    print()
    print("=== Session complete ===")
    print(f"  Duration:      {elapsed:.2f}s")
    print(f"  Events:        {total_events:,}")
    print(f"  Ingested:      {stats.events_ingested:,}")
    print(f"  Dropped:       {stats.events_dropped:,}")
    print(f"  Frames shown:  {viewer.frames_shown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
