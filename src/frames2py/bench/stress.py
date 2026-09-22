"""Throughput benchmark -- measures maximum sustained event ingestion rate.

Feeds synthetic events into the engine as fast as possible (no pacing)
and reports:

- Events ingested / dropped
- Sustained throughput (events/sec)
- Peak and average buffer fill ratio
- Chunks dropped

Usage::

    python -m frames2py.bench.stress --profile high --duration 10 --kernel event_count
    python -m frames2py.bench.stress --profile stress --duration 30 --kernel event_count --json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from frames2py.bench.synthetic import PROFILES, profile_stream
from frames2py.core.engine import Engine
from frames2py.core.types import OverflowPolicy


def run_throughput(
    profile_name: str = "high",
    duration_sec: float = 10.0,
    kernel_name: str = "event_count",
    buffer_capacity: int = 64,
    chunk_size: int = 65_536,
    seed: int = 42,
) -> dict:
    """Run the throughput benchmark and return results as a dict.

    Parameters:
        profile_name: One of ``"low"``, ``"medium"``, ``"high"``,
            ``"stress"``.
        duration_sec: Benchmark duration in seconds.
        kernel_name: Accumulation kernel to use.
        buffer_capacity: Ring buffer chunk slots.
        chunk_size: Events per chunk slot.
        seed: RNG seed.

    Returns:
        Dictionary with benchmark results.
    """
    profile = PROFILES[profile_name]

    engine = Engine(
        sensor_size=profile.sensor_size,
        kernel=kernel_name,
        buffer_capacity=buffer_capacity,
        chunk_size=chunk_size,
        overflow_policy=OverflowPolicy.DROP_OLDEST,
    )

    fill_samples: list[float] = []
    events_generated = 0

    t_start = time.perf_counter()
    for batch in profile_stream(profile_name, duration_sec=duration_sec, seed=seed, paced=False):
        engine.ingest(batch)
        events_generated += len(batch)
        fill_samples.append(engine.stats.buffer_fill_ratio)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    stats = engine.stats

    return {
        "benchmark": "throughput",
        "profile": profile_name,
        "profile_description": profile.description,
        "kernel": kernel_name,
        "duration_sec": round(elapsed, 3),
        "buffer_capacity": buffer_capacity,
        "chunk_size": chunk_size,
        "sensor_size": list(profile.sensor_size),
        "events_generated": events_generated,
        "events_ingested": stats.events_ingested,
        "events_dropped": stats.events_dropped,
        "throughput_evps": round(events_generated / elapsed) if elapsed > 0 else 0,
        "drop_rate_pct": round(stats.events_dropped / max(1, events_generated) * 100, 2),
        "peak_buffer_fill": round(max(fill_samples) if fill_samples else 0.0, 4),
        "avg_buffer_fill": round(float(np.mean(fill_samples)) if fill_samples else 0.0, 4),
        "chunks_dropped": stats.chunks_dropped,
        "snapshots_published": stats.snapshots_published,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": f"{sys.platform}-{platform.machine()}",
    }


def format_results(results: dict) -> str:
    """Format benchmark results as a human-readable report."""
    lines = [
        "frames2py throughput benchmark",
        "=" * 40,
        f"Profile:        {results['profile']} ({results['profile_description']})",
        f"Kernel:         {results['kernel']}",
        f"Duration:       {results['duration_sec']}s",
        f"Buffer:         {results['buffer_capacity']} chunks x {results['chunk_size']} events",
        f"Sensor:         {results['sensor_size'][0]}x{results['sensor_size'][1]}",
        "",
        "Results:",
        f"  Events generated:     {results['events_generated']:,}",
        f"  Events ingested:      {results['events_ingested']:,}",
        f"  Events dropped:       {results['events_dropped']:,}",
        f"  Throughput:           {results['throughput_evps']:,} ev/s",
        f"  Drop rate:            {results['drop_rate_pct']}%",
        f"  Peak buffer fill:     {results['peak_buffer_fill']}",
        f"  Avg buffer fill:      {results['avg_buffer_fill']}",
        f"  Chunks dropped:       {results['chunks_dropped']}",
        f"  Snapshots published:  {results['snapshots_published']:,}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="frames2py throughput benchmark"
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="high",
        help="Benchmark profile (default: high)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--kernel",
        default="event_count",
        help="Kernel name (default: event_count)",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=64,
        help="Ring buffer chunk slots (default: 64)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=65536,
        help="Events per chunk slot (default: 65536)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )

    args = parser.parse_args(argv)

    results = run_throughput(
        profile_name=args.profile,
        duration_sec=args.duration,
        kernel_name=args.kernel,
        buffer_capacity=args.buffer_capacity,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
