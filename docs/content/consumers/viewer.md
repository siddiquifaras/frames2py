# Viewer

`frames2py.viewer` is a small consumer for people: `render()` turns a published snapshot into
an RGB image, and `run()` shows a snapshot source in a window. It reads what the Engine
publishes, at its own pace, and the Engine never waits for it. It is deliberately minimal:
no overlays, colour maps, FPS counters or history. Frames2Py is about observing live state;
the viewer is one way to look at it, not the point.

`render()` needs NumPy only. `run()` needs `frames2py[viewer]` (pyglet >= 2.1.16).
`import frames2py.viewer` imports no GUI library and starts no thread; `run()` without
pyglet raises `ImportError` naming the extra.

## Watch an Engine

```python
# Sketch (not runnable): opens a window and needs an event source of your own.
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

`run(source, *, interval_ms=16.0, title="Frames2Py", scale=None, window_us=50_000)` calls
`source()` once every `interval_ms`, typically `engine.snapshot`. When the publication is
not the one it last showed (a new `SnapshotMeta.sequence`), it renders it; when it is the
same, it only redraws. `None` (before the first publication, or after `reset()`) shows
black. The window takes the frame's size. An exception raised by `source` closes the window
and propagates out of `run()`.

The repository's `examples/view_synthetic.py` (a producer thread with synthetic events) and
`examples/replay_recording.py` (a recording [replayed](../data/replay.md) at its own pace)
are complete programs.

## The main thread

`run()` must be called on the main thread, and raises `RuntimeError` on any other, on every
platform. macOS only lets the main thread create and drive windows: every GUI toolkit tried
while building the viewer (pyglet, pygame, OpenCV, matplotlib, tkinter, glfw, Qt; macOS 15.7
on an Apple M4) failed when a window was opened on another thread: the process aborted, hung,
or the call raised. Rather than depend on that, the viewer refuses up front everywhere, so
code that works on Linux works on macOS too.

So the pattern is always: **the viewer on the main thread, the producer on a thread you
start.** The Engine allows this: any one thread can be its producer, and `snapshot()` is
safe from any thread. The viewer itself starts no thread.

## Cadence

Read snapshots at the publication cadence, not in a tight loop: the default 16 ms matches
the Engine's default `snapshot_interval_ms`. A tick that runs late (a slow render, a slow
`source`) is not made up: the next read happens one interval after the late one. Waiting is
done with `time.sleep()` until the next tick's deadline, so sleep overshoot does not add up
from tick to tick.

## What you see

`render(snapshot, *, scale=None, window_us=50_000)` returns a new `(H, W, 3)` uint8 RGB array.
It only reads the snapshot and never writes it. The mapping follows the frame's dtype and
shape, which are fixed per kernel:

| frame | kernel | image |
|---|---|---|
| `(H, W)` uint32 | `event_count` | grey: `floor(255 * min(v, s) / s)` |
| `(H, W)` float32 | `exp_decay`, `timestamp_decay` | grey, the same formula |
| `(H, W, 2)` uint32 | `polarity` | OFF (channel 0) in blue, ON (channel 1) in yellow (red + green), both white; each channel `floor(255 * min(v, s) / s)` |
| `(H, W)` uint64 | `time_surface` | grey: `floor(255 * max(0, 1 - (T - v) / window_us))`, `T` the snapshot's watermark; `v == 0` black |

Anything else raises `TypeError`; the viewer doesn't guess. That includes the
[temporal kernels'](../core/kernels.md#temporal-kernels) time-first frames, which are not
single images, and a custom kernel's frame of another dtype or shape.

```python title="render.py"
--8<-- "render.py"
```

```text title="Output"
--8<-- "render.out"
```

`s` is the value shown at full brightness. With `scale=` it is that number. With
`scale=None` (the default) it comes from the frame's nonzero values only (for `polarity`,
both channels together):

- no nonzero value: the image is black;
- fewer than 100 nonzero values: `s` is their maximum;
- otherwise: `s` is their 99th percentile (NumPy's default linear interpolation, in float64),
  so a few hot pixels don't dim everything else. Values above `s` show at full brightness.

This is recomputed for every frame, with no smoothing between frames. For a steady picture,
pass a fixed `scale`.

For `time_surface`, a pixel is bright if its last event is recent relative to the snapshot's
watermark and fades linearly to black over `window_us` microseconds (default 50,000, 50 ms).
A pixel holding `t = 0` looks like a pixel with no event, the kernel's documented limitation.
Because `T` is the watermark, not the wall clock, a paused stream keeps its last image.

`scale` and `window_us` must be finite and positive (`ValueError`), and so must
`interval_ms`.

## Platforms

Window tests run in CI on Linux x86_64 under Xvfb, with CPython 3.11 and 3.14t. On macOS, CI
tests the renderer, which needs no window, but opens no window. Measured rendering cost and
the viewer's effect on a live producer are on the [Performance](../reference/performance.md#viewer)
page.
