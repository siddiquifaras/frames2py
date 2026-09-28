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
   viewer / telemetry poll on their own schedule
   (the recorder is written to by your own loop, next to ingest)
```

The engine is a "tap" -- it copies events into its own buffer for visualization. Your processing path keeps running at full speed, even if the viewer falls behind.

## Quick start

```python
import threading
import frames2py
from frames2py import viewer

engine = frames2py.Engine((1280, 720), "event_count")

def produce():
    for events in your_event_source():   # 1-D arrays of frames2py.EVENT_DTYPE
        engine.ingest(events)            # never waits for a consumer
        run_inference(events)            # your critical path

threading.Thread(target=produce, daemon=True).start()
viewer.run(engine.snapshot)              # on the main thread; needs frames2py[viewer]
```

Any consumer can read `engine.snapshot()` the same way, at its own pace.

## Installation

```bash
# Core (NumPy only, no GUI dependencies)
pip install frames2py

# Or with uv
uv add frames2py

# Viewer (pyglet) and event recorder (HDF5)
pip install "frames2py[viewer]"
pip install "frames2py[recorder]"

# File adapters: EVT 2.0 / 3.0 (Prophesee RAW), AEDAT 4.0, HDF5
pip install "frames2py[evt]"
pip install "frames2py[aedat4]"
pip install "frames2py[hdf5]"
```

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

Read recordings as `EVENT_DTYPE` arrays:

```python
from frames2py.adapters import evt

with evt.open("recording.raw") as reader:
    engine = frames2py.Engine(reader.sensor_size, "event_count")
    for events in reader:
        engine.ingest(events)
```

Supported: **EVT 2.0 / 3.0** (`frames2py.adapters.evt`), **AEDAT 4.0** (`frames2py.adapters.aedat4`),
**HDF5** (`frames2py.adapters.hdf5`). See [docs/adapters.md](docs/adapters.md).

## Viewer, recorder and replay

- `frames2py.viewer`: `render()` turns a snapshot into an RGB image; `run()` shows an Engine's
  snapshots in a window, on the main thread. See [docs/viewer.md](docs/viewer.md).
- `frames2py.recorder`: records events to HDF5, read back by `frames2py.adapters.hdf5`. See
  [docs/recorder.md](docs/recorder.md).
- `frames2py.replay.paced()`: replays a recording's batches at their recorded pace. See
  [docs/adapters.md](docs/adapters.md#replaying-at-the-recorded-pace).

Examples: `examples/view_synthetic.py`, `examples/record_and_read_back.py`,
`examples/replay_recording.py`.

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
    consumers/      # Telemetry
    viewer/         # render() and the pyglet viewer
    recorder/       # HDF5 event recorder
    replay.py       # paced replay
    adapters/       # EVT 2.0 / 3.0, AEDAT4 and HDF5 file adapters
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
