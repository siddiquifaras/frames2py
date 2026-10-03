# Frames2Py

![Frames2Py](assets/frames2py-logo.svg#only-light){ width="360" }
![Frames2Py](assets/frames2py-logo-dark.svg#only-dark){ width="360" }

Frames2Py is a Python library for live, decoupled observation of event-camera state.

An event camera reports brightness changes per pixel, as a stream of events, often millions
a second. Most programs that consume such a stream need two things at once: a hot loop that
keeps up with the stream (tracking, inference, control), and some way to look at what the
sensor is seeing right now (a display, a monitor, a logger, a second algorithm). Put the
second inside the first and the hot loop slows to the speed of the display.

Frames2Py separates them. Your producer feeds events to `Engine.ingest()`, which
accumulates them through a kernel into a per-pixel representation and publishes snapshots
of it. Any number of consumers read the latest snapshot at their own pace. The producer
never waits for them.

```text
event stream
    ↓  Engine.ingest()          on your producer thread
kernel state
    ↓  publication              at most once per snapshot_interval_ms
immutable snapshot
    ↓  engine.snapshot()        any thread, any number of consumers
consumers
```

## What is in the box

- **[`Engine`](core/engine.md)**: the live runtime. One producer thread calls `ingest()`;
  consumers call `snapshot()` and read `stats`.
- **[`Accumulator`](core/accumulator.md)**: the same accumulation without publication, for
  synchronous use: offline processing, tests, your own loop.
- **[Seven kernels](core/kernels.md)**: event counts, per-polarity counts, a time surface,
  a per-call exponential decay, an event-time exponential decay, and two temporal kernels
  for models: a per-polarity histogram over time bins and a voxel grid.
- **[Snapshots](core/snapshots.md)**: each publication is a frame and its metadata, shared
  by every consumer and never written again.
- **[File adapters](data/adapters.md)** for EVT 2.0 / 3.0 (Prophesee RAW), AEDAT 4.0 and
  HDF5; a [recorder](data/recorder.md) that writes events to HDF5; [paced
  replay](data/replay.md) of recordings; and a small [viewer](consumers/viewer.md).

The core needs NumPy and nothing else. Adapters, the recorder and the viewer are optional
extras.

## Where to start

1. [Install](getting-started/installation.md) it: `pip install frames2py`.
2. Run the [quickstart](getting-started/quickstart.md): synthetic events in, a snapshot
   out, in about 20 lines.
3. Read [Concepts](getting-started/concepts.md) for the vocabulary, then the
   [event contract](core/event-contract.md) before feeding real data.

## What Frames2Py is not

It is not a camera SDK or driver, a file-conversion toolkit, an ML framework, a general
stream processor or a visualisation package. It reads event arrays at one boundary
([`EVENT_DTYPE`](core/event-contract.md)) and leaves decoding of most formats, hardware access and offline
conversion to the tools that already do them well.

## Status

Version 1.0.0 (see the [changelog](changelog.md)). The public API described here is stable:
changing it incompatibly needs a 2.0. Performance figures are measurements on one machine,
with their conditions: see [Performance](reference/performance.md).
