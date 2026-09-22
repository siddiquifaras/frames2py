# frames2py

**Non-blocking event-to-frame accumulation engine for event cameras.**

Event cameras produce millions of asynchronous events per second. If your visualization code is slow, it throttles your whole pipeline. frames2py fixes this by completely separating accumulation from display. Your processing loop never waits on rendering.

## How it works

frames2py has three layers that run independently:

```
1. Ingest + Accumulate  (never blocks)
   events -> ring buffer -> kernel -> snapshot

2. Snapshot Bridge  (lock-free)
   double-buffered publication via seqlock

3. Consumers  (best-effort, async)
   viewer / recorder / telemetry poll on their own schedule
```

The engine is a "tap" -- it copies events into its own buffer for visualization. Your processing path keeps running at full speed, even if the viewer falls behind.

## Quick start

```python
import frames2py

engine = frames2py.Engine(
    sensor_size=(1280, 720),
    kernel="event_count",
)

viewer = frames2py.Viewer(engine, backend="opencv").start()

for events in your_event_source():
    engine.ingest(events)      # non-blocking, < 2ms
    run_inference(events)      # your critical path, unblocked

viewer.stop()
```

## Installation

```bash
# Core (NumPy only, no GUI dependencies)
pip install frames2py

# Or with uv
uv add frames2py

# With OpenCV viewer
pip install "frames2py[viewer-opencv]"

# With HDF5 support
pip install "frames2py[adapter-h5]"

# Everything
pip install "frames2py[all]"
```

Prophesee (`metavision-core`) and iniVation (`dv-processing`) adapters are included but their SDKs must be installed separately from each vendor. The adapters will tell you what to install if the SDK is missing.

## Kernels

Kernels define how events are turned into frames.

| Kernel | What it does | Resets each snapshot |
|--------|-------------|---------------------|
| `event_count` | Counts events per pixel | Yes |
| `polarity` | Separates ON/OFF events into two channels | Yes |
| `time_surface` | Stores the latest timestamp per pixel | No |
| `exp_decay` | Exponential decay trails | No |

All kernels use vectorized NumPy. Optional C++ backends (pybind11) are available for higher throughput.

## Adapters

Read and write events in any format. No lock-in.

```python
from frames2py.adapters.h5 import from_h5
from frames2py.adapters.aedat4 import from_aedat4
from frames2py.adapters.prophesee import from_prophesee
from frames2py.adapters.inivation import from_inivation
from frames2py.adapters.udp import from_udp

# Convert between formats
from frames2py.adapters.convert import convert
convert("recording.h5", "output.aedat4")
convert("data.aedat4", "data.csv")
convert("events.npy", "events.h5")
```

Supported: **HDF5**, **AEDAT4**, **NumPy**, **CSV**, **UDP**, **Prophesee RAW**, **iniVation AEDAT**.

## Overlays

```python
from frames2py.display.overlays import FPSOverlay, BBoxOverlay, StatsOverlay

viewer.add_overlay(FPSOverlay())
viewer.add_overlay(StatsOverlay(engine))

bbox = BBoxOverlay()
viewer.add_overlay(bbox)
bbox.update([BBox(x1=100, y1=50, x2=300, y2=200, label="car")])
```

## Benchmarks

```bash
python -m frames2py.bench.stress --profile high --duration 10
python -m frames2py.bench.latency --profile high --iterations 5000
```

| Profile | Rate | Sensor | NumPy | C++ |
|---------|------|--------|-------|-----|
| low | 500K ev/s | 346x260 | 0 drops | 0 drops |
| medium | 2M ev/s | 640x480 | 0 drops | 0 drops |
| high | 5M ev/s | 1280x720 | best effort | 0 drops |
| stress | 10M ev/s | 1280x720 | best effort | < 1% drops |

## Key guarantees

1. `engine.ingest()` never blocks. No mutex, no backpressure.
2. Ring buffer drops whole chunks, not individual events.
3. Snapshot publication is lock-free (seqlock).
4. Consumers never hold locks needed by the engine.
5. Telemetry counters are always maintained, even with no consumers attached.

See [docs/invariants.md](docs/invariants.md) for the full list of 12 build-time invariants.

## Project structure

```
frames2py/
  src/frames2py/
    core/           # Engine, types, transport (ring buffer, seqlock)
    kernels/        # Kernel protocol + NumPy/C++ implementations
    consumers/      # Viewer, Recorder, Telemetry
    display/        # Renderer backends + overlays
    adapters/       # H5, AEDAT4, Prophesee, iniVation, UDP, converters
    bench/          # Synthetic event generator + benchmarks
  native/           # C++ pybind11 kernels
  tests/            # pytest suite (169 tests)
  docs/             # Specifications
  examples/         # Usage examples
```

## Development

```bash
# Clone and set up
git clone https://github.com/siddiquifaras/frames2py.git
cd frames2py
uv sync

# Run tests
uv run pytest

# Build for PyPI
uv build
```

## License

MIT. See [LICENSE](LICENSE).
