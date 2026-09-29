<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/siddiquifaras/frames2py/main/docs/content/assets/frames2py-logo-dark.svg">
    <img alt="Frames2Py" src="https://raw.githubusercontent.com/siddiquifaras/frames2py/main/docs/content/assets/frames2py-logo.svg" width="360">
  </picture>
</p>

**Live, decoupled observation of event-camera state.**

**Documentation: <https://siddiquifaras.github.io/frames2py/>**

An event camera reports per-pixel brightness changes as a stream of events, often millions a
second. The loop that consumes that stream (tracking, inference, control) has to keep up,
while something else usually wants to look at what the sensor sees right now: a display, a
monitor, a second algorithm. Put the looking inside the loop and the loop slows to its speed.

Frames2Py separates the two. Your producer feeds events to `Engine.ingest()`, which
accumulates them through a kernel and publishes snapshots. Any number of consumers read the
latest snapshot at their own pace, and the producer never waits for them.

```text
event stream                 your producer: a camera SDK, a file adapter, your own code
    ↓  Engine.ingest()       EVENT_DTYPE arrays, on the producer's thread
Frames2Py
    ↓
accumulation / kernel        counts, polarity, time surface, decays
    ↓
Engine                       publishes at most once per snapshot_interval_ms
    ↓
immutable snapshot           a fresh frame + metadata, shared, never written again
    ↓
independent consumers        engine.snapshot(): any thread, any number
```

- **`Engine`**: the live runtime. One producer thread calls `ingest()`; consumers call
  `snapshot()`. `ingest()` does its CPU work on the caller's thread and never waits on a
  consumer or on I/O.
- **`Accumulator`**: the same accumulation without publication or threads, for offline
  processing, tests and loops you drive yourself.
- **Kernels**: `event_count` and `polarity` (windowed counts: each snapshot holds only the
  events since the previous publication), `time_surface` (latest timestamp per pixel),
  `ExpDecay(decay)` (decays once per call) and `TimestampDecay(tau_us)` (decays with event
  time, independent of how events are batched).
- **Snapshots**: a frame and its metadata (watermark, sequence) from one publication. The
  frame is shared by every consumer and read-only; `snapshot.copy()` gives you your own.

## Quickstart

```python
import numpy as np

import frames2py

# 10,000 synthetic events on a 640x480 sensor, one every microsecond.
rng = np.random.default_rng(seed=0)
events = np.zeros(10_000, dtype=frames2py.EVENT_DTYPE)
events["t"] = np.arange(10_000)              # timestamps, µs
events["x"] = rng.integers(0, 640, 10_000)   # column
events["y"] = rng.integers(0, 480, 10_000)   # row
events["p"] = rng.integers(0, 2, 10_000)     # polarity: 0 is OFF, anything else ON

engine = frames2py.Engine((640, 480), "event_count")  # sensor_size is (width, height)
engine.ingest(events)                                 # the first ingest() always publishes

snapshot = engine.snapshot()  # the latest publication: shared, read-only
print(snapshot.frame.shape, snapshot.frame.dtype)
print("events counted:", int(snapshot.frame.sum()))
print("watermark:", snapshot.meta.watermark, "sequence:", snapshot.meta.sequence)
print("ingested:", engine.stats.events_ingested, "out of bounds:", engine.stats.events_out_of_bounds)
```

```text title="Output"
(480, 640) uint32
events counted: 10000
watermark: 9999 sequence: 1
ingested: 10000 out of bounds: 0
```

In a live program, `ingest()` runs in the producer's own loop on its own thread, and consumers
such as the viewer (`frames2py.viewer.run(engine.snapshot)`) read `snapshot()` elsewhere.

## Installation

CPython 3.11+, NumPy 2.4+:

```sh
pip install frames2py
pip install "frames2py[hdf5,recorder,viewer]"   # with extras
```

The core needs NumPy only. The extras are `evt`, `aedat4`, `hdf5`, `recorder` and `viewer`.

The current release is 1.0.0rc1, a release candidate. pip and uv install a pre-release only
when no stable release exists: until 1.0.0 is published, `pip install frames2py` installs
1.0.0rc1, and after that 1.0.0. To ask for the release candidate explicitly:
`pip install frames2py==1.0.0rc1`, or `pip install --pre frames2py`.

What each extra pulls in, and installing the development version from Git:
[Installation](https://siddiquifaras.github.io/frames2py/getting-started/installation/).

## Adapters, recorder, viewer

- **File adapters** yield `EVENT_DTYPE` arrays from EVT 2.0 / 3.0 (Prophesee RAW), AEDAT 4.0
  and HDF5 recordings:

  ```python
  # Sketch (not runnable): needs a recording of your own.
  import frames2py
  from frames2py.adapters import evt

  with evt.open("recording.raw") as reader:   # pass sensor_size=(w, h) if the header has no geometry
      engine = frames2py.Engine(reader.sensor_size, "event_count")
      for events in reader:
          engine.ingest(events)
  ```

- **Recorder**: writes events to HDF5, next to `ingest()` in your loop; the Engine never calls
  it.
- **Replay**: `frames2py.replay.paced()` yields a recording's batches at their recorded pace.
- **Viewer**: `render()` turns a snapshot into an RGB image; `run()` shows an Engine in a
  window.

Frames2Py ships no vendor SDK adapters: convert your SDK's buffers to `EVENT_DTYPE` and call
`ingest()`.

## Performance

On one Apple M4 (16 GB), the v1 performance gate measured all 150 of its cells above 20M
events/s, on CPython 3.11 and free-threaded 3.14t. No other hardware has been measured.

The cells, the method, the live (paced) figures and the caveats:
[Performance](https://siddiquifaras.github.io/frames2py/reference/performance/).

## Python and platforms

CPython 3.11 to 3.14 and free-threaded 3.14t (GIL disabled), on Linux x86_64, Linux ARM64 and
macOS ARM64, tested in CI (3.12 and 3.13 on Linux x86_64 only). Details:
[Supported Python and platforms](https://siddiquifaras.github.io/frames2py/reference/support/).

## Documentation

The full documentation, with the event contract, kernel semantics, the snapshot and
lifecycle model, adapters, the API reference and the benchmark methodology, is at
**<https://siddiquifaras.github.io/frames2py/>**.

## Development

```sh
git clone https://github.com/siddiquifaras/frames2py.git
cd frames2py
uv sync --all-extras
uv run pytest
```

See [Testing](https://siddiquifaras.github.io/frames2py/development/testing/) and
[Contributing](https://siddiquifaras.github.io/frames2py/development/contributing/).

## License

MIT. See [LICENSE](https://github.com/siddiquifaras/frames2py/blob/main/LICENSE).
