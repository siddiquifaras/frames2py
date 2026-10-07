# Quickstart

Synthetic events in, a snapshot out. No camera, file or GUI needed.

```python title="quickstart.py"
--8<-- "quickstart.py"
```

```text title="Output"
--8<-- "quickstart.out"
```

What happened:

- **Events** are a 1-D NumPy structured array of [`EVENT_DTYPE`](../core/event-contract.md):
  `t` in microseconds, `x` the column, `y` the row, `p` the polarity. Any array with those
  four fields at exactly those dtypes works; extra fields are ignored.
- **`sensor_size` is `(width, height)`**, and the frame it produces is `(height, width)`, as
  NumPy images are.
- **`ingest()`** validated the array, accumulated it into the `event_count` kernel's state
  and, because it was the first call, published a snapshot. Later calls publish at most
  once per `snapshot_interval_ms` (16 ms by default).
- **`snapshot()`** returned that publication: the frame, shared and read-only, plus its
  metadata. The watermark is the largest timestamp accumulated so far; the sequence counts
  publications.

## Next

- **A producer thread and a consumer.** This is what the Engine is for: the producer calls
  `ingest()` in its own loop, and consumers read `snapshot()` whenever they like, or wait
  for the next publication with `wait_for_newer()`. See
  [Writing a consumer](../consumers/writing-a-consumer.md) for a runnable example.
- **Watch it.** The [viewer](../consumers/viewer.md) shows an Engine's snapshots in a
  window: `viewer.run(engine.snapshot)` on the main thread, with the producer on another.
- **Real data.** Read a recording with a [file adapter](../data/adapters.md), or feed your
  camera SDK's buffers converted to `EVENT_DTYPE`.
- **Other representations.** Swap `"event_count"` for another [kernel](../core/kernels.md):
  `"polarity"`, `"time_surface"`, `frames2py.ExpDecay(0.9)`,
  `frames2py.TimestampDecay(10_000.0)`, or a temporal kernel such as
  `frames2py.VoxelGrid(bins=5, bin_us=1_000)`.
- **Frames for a model.** [Handing snapshots to PyTorch](../consumers/pytorch.md) shows the
  copy, dtype and device steps; [`replay.windows()`](../data/replay.md#frames-in-event-time)
  turns a recording into frames at fixed steps of event time.
