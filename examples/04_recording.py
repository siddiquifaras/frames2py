"""Record synthetic event stream to MP4 video.

Demonstrates:
- Engine + Recorder consumer for saving frames to MP4
- Synthetic event generation via profile_stream
- Graceful fallback when OpenCV (cv2) is not available: prints stats only
- Uses cv2.VideoWriter for MP4 encoding

Requires: pip install frames2py[viewer-opencv] or opencv-python for MP4 output

Run with: python examples/04_recording.py [--output recording.mp4] [--duration 3]
"""

from __future__ import annotations

import argparse
import sys
import time

from frames2py import Engine, Recorder
from frames2py.bench.synthetic import profile_stream, PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record synthetic event stream to MP4."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="recording.mp4",
        help="Output MP4 path (default: recording.mp4)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Stream duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="medium",
        help="Workload profile (default: medium)",
    )
    parser.add_argument(
        "--kernel",
        choices=["event_count", "polarity", "time_surface", "exp_decay"],
        default="event_count",
        help="Accumulation kernel (default: event_count)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Recording FPS (default: 30)",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    sensor_size = profile.sensor_size

    # Check if OpenCV is available for Recorder
    try:
        import cv2  # noqa: F401
        has_opencv = True
    except ImportError:
        has_opencv = False

    if not has_opencv:
        print("Note: OpenCV not installed. Running without recording (stats only).")
        print("Install with: pip install frames2py[viewer-opencv] or opencv-python")
        print()

    engine = Engine(sensor_size=sensor_size, kernel=args.kernel)
    recorder = None
    if has_opencv:
        recorder = Recorder(engine, args.output, fps=args.fps)
        recorder.start()
        print(f"Recording to {args.output} at {args.fps} FPS")

    print(f"Profile: {args.profile}, Duration: {args.duration}s")
    print()

    start = time.monotonic()

    try:
        for batch in profile_stream(
            args.profile,
            duration_sec=args.duration,
            paced=True,
        ):
            engine.ingest(batch)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    elapsed = time.monotonic() - start
    engine.stop()
    if recorder is not None:
        recorder.stop()

    stats = engine.stats
    print()
    print("=== Recording complete ===")
    print(f"  Wall time:       {elapsed:.2f}s")
    print(f"  Events ingested: {stats.events_ingested:,}")
    print(f"  Events dropped:   {stats.events_dropped:,}")
    print(f"  Snapshots:        {stats.snapshots_published}")
    if recorder is not None:
        print(f"  Frames written:   {recorder.frames_written}")
        print(f"  Output file:      {args.output}")
    else:
        print("  (No recording - OpenCV not available)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
