# Benchmark Harness Documentation

**Version:** 1.0  
**Status:** Documentation for frames2py benchmark suite.

---

## 1. Overview

The benchmark suite provides two profiles:

| Benchmark | Purpose | Entry Point |
|-----------|---------|-------------|
| **Throughput (stress)** | Maximum sustained event ingestion rate | `python -m frames2py.bench.stress` |
| **Latency** | Per-call `ingest()` wall time percentiles | `python -m frames2py.bench.latency` |

Both use a deterministic synthetic event generator for reproducibility.

---

## 2. Workload Profiles

Defined in `frames2py.bench.synthetic.PROFILES`:

| Profile | Rate (ev/s) | Sensor Size | Batch Size | Description |
|---------|-------------|-------------|------------|-------------|
| `low` | 500K | 346×260 | 5,000 | DAVIS346, indoor static scene |
| `medium` | 2M | 640×480 | 20,000 | DVXplorer, moderate motion |
| `high` | 5M | 1280×720 | 50,000 | IMX636, fast drone flight |
| `stress` | 10M | 1280×720 | 100,000 | Synthetic worst case |

---

## 3. Throughput Benchmark (stress)

### 3.1 Purpose

Feeds synthetic events into the engine as fast as possible (no pacing) and measures:

- Events ingested vs. dropped
- Sustained throughput (events/sec)
- Peak and average buffer fill ratio
- Chunks dropped

### 3.2 Usage

```bash
python -m frames2py.bench.stress --profile high --duration 10
python -m frames2py.bench.stress --profile stress --duration 30 --kernel event_count --json
```

### 3.3 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile` | `high` | One of `low`, `medium`, `high`, `stress` |
| `--duration` | `10.0` | Benchmark duration in seconds |
| `--kernel` | `event_count` | Accumulation kernel name |
| `--buffer-capacity` | `64` | Ring buffer chunk slots |
| `--chunk-size` | `65536` | Events per chunk slot |
| `--seed` | `42` | RNG seed for reproducibility |
| `--json` | `false` | Output results as JSON |

### 3.4 Programmatic API

```python
from frames2py.bench.stress import run_throughput, format_results

results = run_throughput(
    profile_name="high",
    duration_sec=10.0,
    kernel_name="event_count",
    buffer_capacity=64,
    chunk_size=65_536,
    seed=42,
)

print(format_results(results))
```

### 3.5 Output Format (Human-Readable)

```
frames2py throughput benchmark
========================================
Profile:        high (IMX636, fast drone flight)
Kernel:         event_count
Duration:       10.0s
Buffer:         64 chunks x 65536 events
Sensor:         [1280, 720]

Results:
  Events generated:     50,000,000
  Events ingested:      50,000,000
  Events dropped:       0
  Throughput:           5,000,000 ev/s
  Drop rate:            0.0%
  Peak buffer fill:     0.25
  Avg buffer fill:      0.12
  Chunks dropped:       0
  Snapshots published:  1000
```

### 3.6 Output Format (JSON)

```json
{
  "benchmark": "throughput",
  "profile": "high",
  "profile_description": "IMX636, fast drone flight",
  "kernel": "event_count",
  "duration_sec": 10.0,
  "buffer_capacity": 64,
  "chunk_size": 65536,
  "sensor_size": [1280, 720],
  "events_generated": 50000000,
  "events_ingested": 50000000,
  "events_dropped": 0,
  "throughput_evps": 5000000,
  "drop_rate_pct": 0.0,
  "peak_buffer_fill": 0.25,
  "avg_buffer_fill": 0.12,
  "chunks_dropped": 0,
  "snapshots_published": 1000,
  "python_version": "3.11.0",
  "numpy_version": "1.24.0",
  "platform": "darwin-arm64"
}
```

### 3.7 Pass Criteria

- **NumPy (low/medium):** 0 drops expected.
- **NumPy (high/stress):** Best effort; some drops acceptable.
- **C++ backend:** 0 drops for low/medium/high; < 1% for stress.

---

## 4. Latency Benchmark

### 4.1 Purpose

Measures per-call `ingest()` wall time. Reports P50, P95, P99, P99.9, max, mean, stddev, min (milliseconds).

### 4.2 Usage

```bash
python -m frames2py.bench.latency --profile high --iterations 5000
python -m frames2py.bench.latency --profile high --iterations 5000 --json
```

### 4.3 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile` | `high` | Workload profile |
| `--iterations` | `5000` | Number of ingest calls to measure |
| `--kernel` | `event_count` | Accumulation kernel |
| `--buffer-capacity` | `64` | Ring buffer chunk slots |
| `--chunk-size` | `65536` | Events per chunk slot |
| `--seed` | `42` | RNG seed |
| `--json` | `false` | Output as JSON |

### 4.4 Programmatic API

```python
from frames2py.bench.latency import run_latency, format_results

results = run_latency(
    profile_name="high",
    iterations=5_000,
    kernel_name="event_count",
    buffer_capacity=64,
    chunk_size=65_536,
    seed=42,
)

print(format_results(results))
```

### 4.5 Output Format (Human-Readable)

```
frames2py latency benchmark
========================================
Profile:        high (IMX636, fast drone flight)
Kernel:         event_count
Batch size:     50,000
Iterations:     5,000
Sensor:         [1280, 720]

Results:
  P50:    1.2345 ms
  P95:    2.3456 ms
  P99:    3.4567 ms
  P99.9:  4.5678 ms
  Max:    5.6789 ms
  Mean:   1.3456 ms
  Stddev: 0.5678 ms
  Min:    0.8901 ms
```

### 4.6 Output Format (JSON)

```json
{
  "benchmark": "latency",
  "profile": "high",
  "profile_description": "IMX636, fast drone flight",
  "kernel": "event_count",
  "batch_size": 50000,
  "iterations": 5000,
  "sensor_size": [1280, 720],
  "results": {
    "p50_ms": 1.2345,
    "p95_ms": 2.3456,
    "p99_ms": 3.4567,
    "p999_ms": 4.5678,
    "max_ms": 5.6789,
    "mean_ms": 1.3456,
    "stddev_ms": 0.5678,
    "min_ms": 0.8901
  },
  "python_version": "3.11.0",
  "numpy_version": "1.24.0",
  "platform": "darwin-arm64"
}
```

### 4.7 Pass Criteria

- **P99 < 5 ms** for high profile (50K events/batch) on typical hardware.
- **P99.9** should not exceed 2× P99 for stable implementations.

---

## 5. Synthetic Event Generator

### 5.1 generate_batch

```python
from frames2py.bench.synthetic import generate_batch, EVENT_DTYPE

events = generate_batch(
    n_events=10_000,
    sensor_size=(1280, 720),
    t_start=0,
    t_step=1,
    seed=42,
)
# Returns 1-D structured array with EVENT_DTYPE
```

### 5.2 event_stream

```python
from frames2py.bench.synthetic import event_stream

for batch in event_stream(
    rate_events_per_sec=5_000_000,
    sensor_size=(1280, 720),
    batch_size=50_000,
    duration_sec=10.0,
    seed=42,
    paced=True,  # False for throughput benchmark
):
    engine.ingest(batch)
```

### 5.3 profile_stream

```python
from frames2py.bench.synthetic import profile_stream, PROFILES

for batch in profile_stream("high", duration_sec=10.0, seed=42, paced=False):
    engine.ingest(batch)
```

---

## 6. Reproducibility

All benchmarks use a **seeded** RNG. Same `--seed` and `--profile` produce identical event sequences across runs and machines.
