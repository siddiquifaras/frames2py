# frames2py

**Live, decoupled observation of event-camera state.**

A producer feeds events to `Engine.ingest()`, which accumulates them through a kernel and
publishes snapshots. Any number of consumers read the latest snapshot at their own pace, and
the producer never waits for them.

## How it works

```
event stream -> Engine.ingest() -> kernel state -> published snapshot -> consumers
                (your producer thread)                (engine.snapshot(), any thread, any number)
```

`ingest()` does its CPU work on the caller's thread and never waits for a consumer or for
I/O. A consumer calls `engine.snapshot()` whenever it likes and gets the latest published
snapshot; one that falls behind simply sees a later snapshot. The recorder is not a consumer
of snapshots: your own loop writes events to it, next to `ingest()`.

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

A kernel defines the state the events accumulate into.

| Kernel | Output | State across snapshots |
|--------|--------|------------------------|
| `event_count` | `(H, W)` uint32, events per pixel | windowed: each publication starts a new window |
| `polarity` | `(H, W, 2)` uint32, OFF and ON counts | windowed |
| `time_surface` | `(H, W)` uint64, latest timestamp per pixel | running |
| `ExpDecay(decay)` | `(H, W)` float32, decays once per `ingest()` call | running |
| `TimestampDecay(tau_us)` | `(H, W)` float32, decays with event time | running |

The kernels are NumPy.

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

The benchmark suite is in `benchmarks/`, in the repository but not in the package
(`uv run python -m benchmarks --help`). Results with their conditions (hardware, Python,
NumPy, resolution, events per call, kernel, method) will be published with the v1
documentation; none are quoted here yet.

## Core behaviour

1. `ingest()` never waits on consumers. Consumer reads do not require the producer to wait for
   consumer-side frame copies. It is still CPU work on the caller's thread, and runtime
   effects (the GIL, CPython's own locks, garbage collection, scheduling) can delay it.
2. A snapshot's frame and metadata always come from the same publication. The frame is shared
   by every consumer and marked read-only, and Frames2Py never writes it again. The read-only
   flag is NumPy's, not a memory-safety boundary: to modify the data, or hand it to a library
   that ignores the flag, use `snapshot.copy()`.
3. Every event of an accepted `ingest()` call is either accumulated or counted as out of bounds.

## Project structure

```
frames2py/
  src/frames2py/
    _engine.py, _accumulator.py, publish.py   # the v1 core
    kernels/        # Kernel protocol and the five kernels
    viewer/         # render() and the pyglet viewer
    recorder/       # HDF5 event recorder
    replay.py       # paced replay
    adapters/       # EVT 2.0 / 3.0, AEDAT4 and HDF5 file adapters
  benchmarks/       # the benchmark suite
  tests/            # pytest suite
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
