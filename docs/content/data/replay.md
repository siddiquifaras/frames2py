# Replay

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
