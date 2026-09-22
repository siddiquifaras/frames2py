"""H5 file playback through the Engine.

Demonstrates:
- Reading event data from HDF5 files via from_h5 adapter
- Playing back events through the Engine with paced timing
- Creating a temporary H5 file with synthetic data when no file is provided
- Displaying playback stats at completion

Requires: pip install frames2py[adapter-h5]

Run with: python examples/02_h5_playback.py [path/to/events.h5]
          python examples/02_h5_playback.py  (creates temp file with synthetic data)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time

from frames2py import Engine, Viewer
from frames2py.bench.synthetic import generate_batch, PROFILES

try:
    from frames2py.adapters.h5 import from_h5, to_h5
    HAS_H5 = True
except ImportError:
    HAS_H5 = False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play back events from an H5 file through the Engine."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to .h5 file (if omitted, creates temp file with synthetic data)",
    )
    parser.add_argument(
        "--kernel",
        choices=["event_count", "polarity", "time_surface", "exp_decay"],
        default="event_count",
        help="Accumulation kernel (default: event_count)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help="Events per batch when reading H5 (default: 20000)",
    )
    parser.add_argument(
        "--paced",
        action="store_true",
        default=True,
        help="Pace playback at real-time rate (default: True)",
    )
    parser.add_argument(
        "--no-paced",
        action="store_false",
        dest="paced",
        help="Play back as fast as possible",
    )
    args = parser.parse_args()

    if not HAS_H5:
        print("Error: H5 adapter requires h5py. Install with: pip install frames2py[adapter-h5]")
        return 1

    h5_path = args.path
    temp_file = None
    sensor_size: tuple[int, int]

    if h5_path is None:
        # Create temp H5 with synthetic data
        profile = PROFILES["medium"]
        sensor_size = profile.sensor_size
        n_events = int(profile.rate * 2)  # ~2 seconds worth
        events = generate_batch(
            n_events=n_events,
            sensor_size=sensor_size,
            seed=123,
        )
        temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        temp_file.close()
        h5_path = temp_file.name
        to_h5(h5_path, events)
        print(f"Created temp H5 file with {n_events:,} synthetic events: {h5_path}")
    else:
        # Infer sensor size from event coordinates (H5 doesn't store it)
        max_x, max_y = 0, 0
        for batch, _ in from_h5(h5_path, chunk_size=args.chunk_size):
            if len(batch) > 0:
                max_x = max(max_x, int(batch["x"].max()))
                max_y = max(max_y, int(batch["y"].max()))
        if max_x == 0 and max_y == 0:
            print("Error: H5 file is empty or has no valid events")
            return 1
        sensor_size = (max(max_x + 1, 64), max(max_y + 1, 48))

    engine = Engine(sensor_size=sensor_size, kernel=args.kernel)
    viewer = Viewer(engine, backend="headless", fps=30)
    viewer.start()

    total_events = 0
    batch_count = 0
    start = time.monotonic()
    t_prev = start

    try:
        for batch, meta in from_h5(h5_path, chunk_size=args.chunk_size):
            engine.ingest(batch, meta)
            total_events += len(batch)
            batch_count += 1

            if args.paced and len(batch) > 0:
                # Simple pacing: sleep proportional to batch
                t_last = batch["t"][-1] / 1e6
                t_now = time.monotonic()
                target = t_last - (batch["t"][0] / 1e6) if batch_count > 1 else 0
                if target > 0 and (t_now - t_prev) < target:
                    time.sleep(target - (t_now - t_prev))
                t_prev = t_now
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    elapsed = time.monotonic() - start
    engine.stop()
    viewer.stop()

    stats = engine.stats
    print()
    print("=== Playback complete ===")
    print(f"  Source:        {h5_path}")
    print(f"  Wall time:     {elapsed:.2f}s")
    print(f"  Batches:       {batch_count}")
    print(f"  Events:        {total_events:,}")
    print(f"  Ingested:      {stats.events_ingested:,}")
    print(f"  Dropped:       {stats.events_dropped:,}")
    print(f"  Snapshots:     {stats.snapshots_published}")
    print(f"  Viewer shown:  {viewer.frames_shown}")

    if temp_file is not None:
        import os
        os.unlink(h5_path)
        print(f"  (temp file removed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
