"""Synthetic event stream with Engine and headless Viewer.

Demonstrates:
- Engine with all four kernels: event_count, polarity, time_surface, exp_decay
- Headless Viewer (zero external deps, no OpenCV required)
- Synthetic event generation via profile_stream
- Snapshot metadata inspection
- Engine stats at completion

Run with: python examples/01_synthetic.py [--kernel event_count] [--duration 2] [--profile medium]
"""

from __future__ import annotations

import argparse
import sys
import time

from frames2py import Engine, Viewer
from frames2py.bench.synthetic import profile_stream, PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic event stream through Engine with headless Viewer."
    )
    parser.add_argument(
        "--kernel",
        choices=["event_count", "polarity", "time_surface", "exp_decay"],
        default="event_count",
        help="Accumulation kernel (default: event_count)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Stream duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="medium",
        help="Workload profile: low, medium, high, stress (default: medium)",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    sensor_size = profile.sensor_size

    print(f"Profile: {args.profile} ({profile.description})")
    print(f"Kernel: {args.kernel}")
    print(f"Duration: {args.duration}s")
    print(f"Sensor: {sensor_size[0]}x{sensor_size[1]}")
    print()

    engine = Engine(sensor_size=sensor_size, kernel=args.kernel)
    viewer = Viewer(engine, backend="headless", fps=30, colormap="viridis")
    viewer.start()

    snapshot_count = 0
    start = time.monotonic()

    try:
        for batch in profile_stream(
            args.profile,
            duration_sec=args.duration,
            paced=True,
        ):
            engine.ingest(batch)

            result = engine.latest_snapshot()
            if result is not None:
                frame, snap_meta = result
                snapshot_count += 1
                if snapshot_count <= 3 or snapshot_count % 50 == 0:
                    print(
                        f"  Snapshot #{snap_meta.seq}: "
                        f"ts={snap_meta.timestamp / 1e6:.4f}s, "
                        f"events_acc={snap_meta.events_accumulated}, "
                        f"shape={frame.shape}"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    elapsed = time.monotonic() - start
    engine.stop()
    viewer.stop()

    stats = engine.stats
    print()
    print("=== Playback complete ===")
    print(f"  Wall time:     {elapsed:.2f}s")
    print(f"  Events ingested: {stats.events_ingested:,}")
    print(f"  Events dropped:  {stats.events_dropped:,}")
    print(f"  Snapshots:      {stats.snapshots_published}")
    print(f"  Viewer shown:   {viewer.frames_shown}")
    print(f"  Viewer dropped: {viewer.frames_dropped}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
