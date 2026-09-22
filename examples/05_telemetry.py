"""Engine + Telemetry consumer with formatted table output.

Demonstrates:
- Telemetry consumer for periodic engine health monitoring
- Accumulate latency percentiles (p50, p95, p99)
- TelemetrySample fields: events, drops, buffer fill, viewer stats
- Formatted table output of collected samples

Run with: python examples/05_telemetry.py [--duration 5] [--interval 500]
"""

from __future__ import annotations

import argparse
import sys
import time

from frames2py import Engine, Telemetry, TelemetrySample
from frames2py.bench.synthetic import profile_stream, PROFILES


def format_sample(s: TelemetrySample, index: int, start_ns: int) -> str:
    """Format a TelemetrySample as a table row."""
    elapsed_ms = (s.wall_time_ns - start_ns) // 1_000_000
    ingest = f"{s.events_ingested:,}"
    drops = f"{s.events_dropped:,}"
    fill = f"{s.buffer_fill_ratio:.0%}"
    snaps = str(s.snapshots_published)
    p50 = f"{s.accumulate_ms_p50:.2f}"
    p95 = f"{s.accumulate_ms_p95:.2f}"
    p99 = f"{s.accumulate_ms_p99:.2f}"
    viewer = "-"
    if s.viewer_frames_shown is not None:
        viewer = f"{s.viewer_frames_shown}"
    return (
        f"  {index:3d} | {elapsed_ms:6d}ms | {ingest:>10} | {drops:>8} | "
        f"{fill:>5} | {snaps:>6} | {p50:>6} | {p95:>6} | {p99:>6} | {viewer:>6}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic stream with Telemetry consumer and print stats table."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Stream duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=500.0,
        help="Telemetry poll interval in ms (default: 500)",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="high",
        help="Workload profile (default: high)",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    sensor_size = profile.sensor_size

    print("Telemetry consumer demo")
    print(f"  Profile: {args.profile}, Duration: {args.duration}s")
    print(f"  Poll interval: {args.interval}ms")
    print()

    engine = Engine(sensor_size=sensor_size, kernel="event_count")
    telemetry = Telemetry(engine, poll_interval_ms=args.interval)
    telemetry.start()

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
    telemetry.stop()

    history = telemetry.history
    print()
    print("=== Telemetry samples (TelemetrySample fields) ===")
    print()
    print(
        "  #   | Elapsed | Ingested   | Dropped  | Fill  | Snaps  | "
        "p50ms  | p95ms  | p99ms  | Viewer"
    )
    print("  " + "-" * 85)

    start_ns = history[0].wall_time_ns if history else 0
    for i, sample in enumerate(history):
        print(format_sample(sample, i + 1, start_ns))

    if history:
        latest = history[-1]
        print()
        print("Latest sample fields:")
        print(f"  wall_time_ns:         {latest.wall_time_ns}")
        print(f"  events_ingested:      {latest.events_ingested:,}")
        print(f"  events_dropped:       {latest.events_dropped:,}")
        print(f"  chunks_dropped:       {latest.chunks_dropped}")
        print(f"  buffer_fill_ratio:    {latest.buffer_fill_ratio:.2%}")
        print(f"  snapshots_published:  {latest.snapshots_published}")
        print(f"  accumulate_ms_p50:    {latest.accumulate_ms_p50:.3f}")
        print(f"  accumulate_ms_p95:    {latest.accumulate_ms_p95:.3f}")
        print(f"  accumulate_ms_p99:    {latest.accumulate_ms_p99:.3f}")

    print()
    print(f"Total samples: {len(history)}, Wall time: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
