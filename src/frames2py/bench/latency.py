"""Latency benchmark -- measures per-call ``ingest()`` wall time.

Feeds synthetic events and records the wall-clock duration of each
:meth:`Engine.ingest` call.  Reports P50 / P95 / P99 / P99.9 / max
percentiles.

Usage::

    python -m frames2py.bench.latency --profile high --iterations 5000 --kernel event_count
    python -m frames2py.bench.latency --profile high --iterations 5000 --json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from frames2py.bench.synthetic import PROFILES, generate_batch
from frames2py.core.engine import Engine
from frames2py.core.types import OverflowPolicy


def run_latency(
    profile_name: str = "high",
    iterations: int = 5_000,
    kernel_name: str = "event_count",
    buffer_capacity: int = 64,
    chunk_size: int = 65_536,
    seed: int = 42,
) -> dict:
    """Run the latency benchmark and return results as a dict.

    Parameters:
        profile_name: One of ``"low"``, ``"medium"``, ``"high"``,
            ``"stress"``.
        iterations: Number of ingest calls to measure.
        kernel_name: Accumulation kernel to use.
        buffer_capacity: Ring buffer chunk slots.
        chunk_size: Events per chunk slot.
        seed: RNG seed.

    Returns:
        Dictionary with latency percentiles and metadata.
    """
    profile = PROFILES[profile_name]

    engine = Engine(
        sensor_size=profile.sensor_size,
        kernel=kernel_name,
        buffer_capacity=buffer_capacity,
        chunk_size=chunk_size,
        overflow_policy=OverflowPolicy.DROP_OLDEST,
    )

    # Pre-generate all batches to avoid generator overhead in the timed loop.
    batches = [
        generate_batch(
            n_events=profile.batch_size,
            sensor_size=profile.sensor_size,
            t_start=i * profile.batch_size,
            seed=seed + i,
        )
        for i in range(iterations)
    ]

    # Warmup: 10 iterations to stabilise caches / JIT
    warmup_n = min(10, iterations)
    for i in range(warmup_n):
        engine.ingest(batches[i])
    engine.reset()

    latencies_ns = np.empty(iterations, dtype=np.int64)

    for i in range(iterations):
        t0 = time.perf_counter_ns()
        engine.ingest(batches[i])
        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0

    latencies_ms = latencies_ns / 1_000_000.0

    return {
        "benchmark": "latency",
        "profile": profile_name,
        "profile_description": PROFILES[profile_name].description,
        "kernel": kernel_name,
        "batch_size": profile.batch_size,
        "iterations": iterations,
        "sensor_size": list(profile.sensor_size),
        "results": {
            "p50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
            "p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
            "p999_ms": round(float(np.percentile(latencies_ms, 99.9)), 4),
            "max_ms": round(float(np.max(latencies_ms)), 4),
            "mean_ms": round(float(np.mean(latencies_ms)), 4),
            "stddev_ms": round(float(np.std(latencies_ms)), 4),
            "min_ms": round(float(np.min(latencies_ms)), 4),
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": f"{sys.platform}-{platform.machine()}",
    }


def format_results(results: dict) -> str:
    """Format latency results as a human-readable report."""
    r = results["results"]
    lines = [
        "frames2py latency benchmark",
        "=" * 40,
        f"Profile:        {results['profile']} ({results['profile_description']})",
        f"Kernel:         {results['kernel']}",
        f"Batch size:     {results['batch_size']:,}",
        f"Iterations:     {results['iterations']:,}",
        f"Sensor:         {results['sensor_size'][0]}x{results['sensor_size'][1]}",
        "",
        "Results:",
        f"  P50:    {r['p50_ms']:.4f} ms",
        f"  P95:    {r['p95_ms']:.4f} ms",
        f"  P99:    {r['p99_ms']:.4f} ms",
        f"  P99.9:  {r['p999_ms']:.4f} ms",
        f"  Max:    {r['max_ms']:.4f} ms",
        f"  Mean:   {r['mean_ms']:.4f} ms",
        f"  Stddev: {r['stddev_ms']:.4f} ms",
        f"  Min:    {r['min_ms']:.4f} ms",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="frames2py latency benchmark"
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="high",
        help="Benchmark profile (default: high)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of ingest calls (default: 5000)",
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

    results = run_latency(
        profile_name=args.profile,
        iterations=args.iterations,
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
