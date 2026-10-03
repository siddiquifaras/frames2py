# Replay

`frames2py.replay` has two helpers: `paced()` feeds an Engine at the rate the events were
recorded, and [`windows()`](#frames-in-event-time) turns a recording into frames every N µs of
event time.

A reader yields batches as fast as it can decode them. To feed an Engine at the rate the
events were recorded (to watch a recording, or to test a consumer against a realistic
stream), wrap the reader in `frames2py.replay.paced`:

```python
# Sketch (not runnable): needs a recording of your own.
import frames2py
from frames2py.adapters import evt
from frames2py.replay import paced

with evt.open("recording.raw", batch_size=10_000) as reader:
    engine = frames2py.Engine(reader.sensor_size, "event_count")
    for events in paced(reader, speed=1.0):   # 2.0 is twice as fast, 0.5 half
        engine.ingest(events)
    engine.stop()
```

`paced(batches, *, speed=1.0, clock=time.monotonic_ns, sleep=time.sleep)` works on any
iterable of `EVENT_DTYPE` arrays. It is a generator that runs on the caller's thread and
yields each batch unchanged (the same array object), sleeping between batches. It starts no
thread, keeps no queue and never drops, reorders or copies a batch.

## When a batch is due

The first nonempty batch fixes the start: its smallest timestamp `t0`, and the clock's
reading at that moment. `M` is the largest timestamp seen so far, including the batch about
to be yielded. The batch is yielded once the clock has advanced `(M - t0) / speed`
microseconds past the start, rounded up to a whole nanosecond so that no batch is early. So
the first batch waits for its own span, as it would have from a live sensor, and each later
batch for its newest event. Empty batches are yielded at once.

```python title="replay_fake_clock.py"
--8<-- "replay_fake_clock.py"
```

```text title="Output"
--8<-- "replay_fake_clock.out"
```

`clock` (nanoseconds, monotonic) and `sleep` (seconds) are replaceable, as here, for a
simulated clock in tests or a more precise sleep.

## Discontinuities are taken as they come

Nothing is repaired and no reset is inferred:

- a batch whose timestamps go back, such as a source clock that restarted, leaves `M` where
  it was, so it is already due and is yielded at once (the third batch above), and so are the
  ones after it until the timestamps pass `M` again;
- a batch with a timestamp far ahead advances `M`, so the replay waits as long as the jump
  says, and the batches after it wait behind it.

If a recording's clock restarts, call `engine.reset()` yourself, as for a live source.

## Falling behind

If the consumer is slower than the recording, each batch is yielded as soon as it is asked
for; the replay doesn't skip ahead to catch up. The schedule is always measured from the
start, so lateness never accumulates into drift.

## Accuracy and errors

The wait is `sleep`, by default `time.sleep()`, so each batch is late by the operating
system's sleep overshoot (measured figures on the [Performance](../reference/performance.md#paced-replay)
page). `speed` must be a finite number above 0, otherwise `ValueError` when `paced()` is
called; a batch that is not an `EVENT_DTYPE` array raises `TypeError` when it is reached.

## Frames in event time

To turn a recording into frames at regular intervals of *event* time, for training or
evaluating a model, use `frames2py.replay.windows`. It has no clock: the same events always
give the same frames, however fast it runs.

```python
# Sketch (not runnable): needs a recording of your own.
import frames2py
from frames2py.adapters import evt
from frames2py.replay import windows

with evt.open("recording.raw") as reader:
    kernel = frames2py.VoxelGrid(bins=5, bin_us=12_500)
    for t_us, frame in windows(reader, reader.sensor_size, kernel, every_us=50_000):
        ...  # frame: (5, H, W) float32, the 50 ms of events before t_us
```

`windows(batches, sensor_size, kernel, *, every_us)` yields `(t_us, frame)` pairs.
`sensor_size` and `kernel` are as for [`Accumulator`](../core/accumulator.md), which it
accumulates through, in arrival order; `every_us` is an int `>= 1`.

- **Boundaries** are the multiples of `every_us`: absolute event times, like the temporal
  kernels' bins. `t_us` is the frame's boundary, an exclusive end.
- **When a frame is taken.** The frame at `t_us` is read just before the first in-bounds
  event that brings the watermark to `t_us` or past it is accumulated. For in-order events it
  therefore holds exactly the events before `t_us`; an event at `t_us` belongs to the next
  frame. An event that arrives after the watermark has passed a boundary later than its own
  timestamp shows only in later frames: a windowed kernel counts it in the window open when
  it arrives, and a running kernel places it at its own timestamp, so a temporal kernel shows
  it only while its bin is still in the frame.
- **Read at the boundary.** Each frame is evaluated at `t_us`, not at the latest event: a
  `StackedHistogram` or `VoxelGrid` frame shows the bins completed by `t_us`, even when the
  last of them are empty, and `TimestampDecay` is decayed to `t_us`.
- **The first frame** is at the first boundary after the first in-bounds event. Out-of-bounds
  events cross no boundary.
- **Windowed kernels** (`event_count`, `polarity`) start a new window after each frame, as
  at an Engine publication, so with in-order events a frame counts `[t_us - every_us, t_us)`.
- **Gaps get frames.** Every boundary the watermark passes gets one, so frame `i` is always
  `every_us` after frame `i - 1`. A forward timestamp spike therefore yields a frame for
  every boundary up to the spike, which can be a very large number of frames: handle
  [timestamp discontinuities](../core/event-contract.md#timestamp-discontinuities) before
  calling.
- **The tail is dropped.** The window after the last boundary the watermark reached is
  incomplete and isn't yielded.
- **Each frame is a new writable array**, yours to keep or modify.
- **Batches don't matter.** The frames don't depend on how the events are split into
  batches; batches are split at the boundaries as needed.

```python title="windows.py"
--8<-- "windows.py"
```

```text title="Output"
--8<-- "windows.out"
```

**`ExpDecay` is refused** with `TypeError` when `windows()` is called: its decay is per call,
and `windows()` splits batches at the boundaries, so its frames would depend on those split
points. Use `TimestampDecay`, which decays in event time. A
[custom kernel](../core/kernels.md#custom-kernels) must evaluate at the time `read()` is
given; one whose result depends on how events are split into calls isn't detected, and its
frames depend on the split points.

**Errors.** `every_us` that isn't an int, a bool included, raises `TypeError`, and one below
1 `ValueError`, when `windows()` is called. Each batch is checked whole when it is reached,
before any of its events is accumulated: a batch that isn't an `EVENT_DTYPE` array raises
`TypeError`, and one with an event at `t >= 2**63` raises `ValueError` before any frame it
would complete is yielded.
