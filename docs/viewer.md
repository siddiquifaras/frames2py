# Looking at snapshots

`frames2py.viewer` is a small consumer for humans: `render()` turns a published snapshot into
an RGB image, and `run()` shows a snapshot source in a window. It is one consumer among any
number; it reads what the Engine publishes, at its own pace, and the Engine never waits for it.
It is deliberately minimal (no overlays, colormaps, FPS counters or history): Frames2Py is
about observing live state, and the viewer is one way to look at it, not the point.

```sh
pip install "frames2py[viewer]"   # pyglet; only run() needs it
```

`render()` needs nothing beyond NumPy. `import frames2py` and `import frames2py.viewer` import
no GUI library and start no thread; `run()` without pyglet raises `ImportError` naming the
extra.

## Watch an Engine

```python
import threading
import frames2py
from frames2py import viewer

engine = frames2py.Engine((1280, 720), "polarity")

def produce():                      # your camera or file loop
    for events in source:
        engine.ingest(events)
    engine.stop()

threading.Thread(target=produce, daemon=True).start()
viewer.run(engine.snapshot)         # main thread; returns when the window is closed (or Esc)
```

```python
frames2py.viewer.run(source, *, interval_ms=16.0, title="Frames2Py", scale=None, window_us=50_000)
frames2py.viewer.render(snapshot, *, scale=None, window_us=50_000)   # -> (H, W, 3) uint8 RGB
```

`run()` calls `source()` once every `interval_ms`, typically `engine.snapshot`. When the
publication is not the one it last showed (a new `SnapshotMeta.sequence`), it renders it; when
it is the same, it only redraws. `None` (before the first publication, or after `reset()`)
shows black. The window takes the frame's size. An exception raised by `source` closes the
window and propagates out of `run()`.

The examples: `examples/view_synthetic.py` (a producer thread with synthetic events) and
`examples/replay_recording.py` (a recording replayed at its own pace, below).

## The main thread

`run()` must be called on the main thread, and raises `RuntimeError` on any other, on every
platform. macOS only lets the main thread create and drive windows: in the research for the
viewer, every GUI toolkit tried (pyglet, pygame, OpenCV, matplotlib, tkinter, glfw, Qt; macOS 15.7
on an M4) failed when a window was opened on another thread: the process aborted, hung, or
the call raised. Rather than depend on
that, the viewer refuses up front, everywhere, so code that works on Linux works on macOS too.

So the pattern is always: **the viewer on the main thread, the producer on a thread you
start.** The Engine allows this: any one thread can be its producer (the first to call
`ingest()`), and `snapshot()` is safe to call from any thread. The viewer itself starts no
thread.

## Cadence

Read snapshots at the publication cadence, not in a tight loop: the default 16 ms matches the
Engine's default `snapshot_interval_ms`. A tick that runs late (a slow render, a slow
`source`) is not made up: the next read happens one interval after the late one. Waiting is
done with `time.sleep()` until the next tick's deadline, so sleep overshoot does not add up
from tick to tick; in the measurement below the viewer presented 61-62 frames a second at the
16 ms default.

## What you see

`render()` chooses the mapping from the frame's dtype and shape, which are fixed per kernel.
It only reads the snapshot: the published frame is shared by every consumer and read-only,
and `render()` never writes it; the image it returns is a new array.

| frame | kernel | image |
|---|---|---|
| `(H, W)` uint32 | `event_count` | grey: `floor(255 * min(v, s) / s)` |
| `(H, W)` float32 | `exp_decay`, `timestamp_decay` | grey, the same formula |
| `(H, W, 2)` uint32 | `polarity` | OFF (channel 0) in blue, ON (channel 1) in yellow (red + green), both white; each channel `floor(255 * min(v, s) / s)` |
| `(H, W)` uint64 | `time_surface` | grey: `floor(255 * max(0, 1 - (T - v) / window_us))`, `T` the snapshot's watermark; `v == 0` black |

Anything else (a custom kernel's frame of another dtype or shape) raises `TypeError`; the
viewer doesn't guess.

`s` is the value shown at full brightness. With `scale=` it is that number. With
`scale=None` (the default) it comes from the frame, from its nonzero values only (for
`polarity`, both channels together):

- no nonzero value: the image is black;
- fewer than 100 nonzero values: `s` is their maximum;
- otherwise: `s` is their 99th percentile (NumPy's default, linear interpolation, in float64),
  so a few hot pixels don't dim everything else. Values above `s` show at full brightness.

This is recomputed for every frame; there is no smoothing between frames and no other state.
For a steady picture, pass a fixed `scale`.

For `time_surface`, a pixel is bright if its last event is recent relative to the snapshot's
watermark (the latest accumulated timestamp) and fades linearly to black over `window_us`
microseconds (default 50,000, 50 ms). A pixel holding `t = 0` looks like a pixel with no event,
the kernel's documented limitation. Because `T` is the watermark, not the wall clock, a paused
stream keeps its last image.

## What rendering costs

Measured with `python -m benchmarks viewer` (`benchmarks/consumers.py`) on an Apple M4, macOS
15.7.7, CPython 3.11.14 and NumPy 2.4.6; 5 runs, medians. CPython 3.14.2t with the GIL disabled
(NumPy 2.5.3) was 3-13% slower in the same cells.

`render()` at 1280x720, on an Engine snapshot after one 16 ms window of a 20M events/s stream
(the running kernels after 80 ms of it):

| kernel | uniform events | clustered events |
|---|---|---|
| `event_count` | 6.8 ms (2.8 of them for the automatic scale) | 4.4 ms (0.6) |
| `polarity` | 9.5 ms (3.4) | 7.2 ms (1.1) |
| `time_surface` | 7.6 ms | 6.4 ms |
| `exp_decay` | 12.1 ms (8.3) | 4.5 ms (0.8) |
| `timestamp_decay` | 12.0 ms (8.3) | 4.6 ms (0.8) |

The automatic scale sorts out the 99th percentile of the nonzero values, so it costs in
proportion to how many there are: 0.6-0.8 million on the uniform decay frames, where it is
two-thirds of the render. Passing `scale=` skips it. At 640x480 a render took 1.4-4.1 ms,
at 346x260 0.4-1.8 ms. Each render allocates temporaries of up to 28 MiB at 1280x720.

With the viewer's loop rendering every publication at 16 ms on the main thread, and a producer
thread feeding the Engine at 20M events/s (1280x720, `event_count`, `polarity`,
`time_surface`, `timestamp_decay`, 10,000 and 100,000 events per call), the producer kept its
20M events/s in every run, and the viewer presented 61-62 frames a second. The time the
producer spent inside `ingest()` did not go up: its throughput while busy was higher with the
viewer running than without it, 1.2-1.9x on 3.11.14 and 1.5-3.6x on 3.14.2t. That is most
likely the machine keeping a busier process on faster cores, not something the viewer does for
the producer, and it was not measured directly; what the measurement shows is that the viewer
didn't slow the producer down in these runs. The full figures, conditions and the p99 latencies
are in the benchmark's output.
