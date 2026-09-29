# Snapshots and consumers

```text
event stream
    ↓
accumulation through a kernel      (Engine.ingest(), on the producer's thread)
    ↓
Engine publication                 (at most once per snapshot_interval_ms)
    ↓
immutable published snapshot       (a fresh frame + its SnapshotMeta, one object)
    ↓
independent consumers              (engine.snapshot(), any thread, any number)
```

A consumer never interacts with the producer. It asks the Engine for the latest published
snapshot whenever it wants one, and gets it without waiting and without making the producer
wait. A slow consumer doesn't slow the producer down; it just sees a later snapshot next
time, and skips the publications in between. A fast consumer doesn't get more publications
than there are; it sees the same one again (same `sequence`).

## What a snapshot is

`engine.snapshot()` returns a `frames2py.publish.Snapshot`, or `None`:

- **`frame`**: the published array, `(height, width)` or `(height, width, 2)`, with the
  kernel's output dtype. It is the published array itself, not a copy, shared by every
  consumer that reads this publication, and marked read-only. Frames2Py never writes it
  again: the next publication goes into a new array.
- **`meta`**: its `SnapshotMeta`, with `watermark` (the largest accumulated in-bounds
  timestamp at publication, or `None`) and `sequence` (the publication number).
- **`copy(out=None)`**: an independent, writable copy of the frame.

Frame and metadata always come from the same publication. A snapshot never shows a frame
that is still being written.

```python title="snapshots.py"
--8<-- "snapshots.py"
```

```text title="Output"
--8<-- "snapshots.out"
```

## Read-only is NumPy's flag, not memory protection

The frame's `writeable` flag stops accidental writes through `snapshot.frame`. It is not a
memory-safety boundary: code can set the flag back on the owning array, and libraries that
ignore it (`torch.from_numpy`, for one) share the memory writably. Frames2Py never writes a
published frame; if your consumer modifies the data, or hands it to such a library, give it
`snapshot.copy()` or `snapshot.copy(out=...)` instead.

`copy(out=...)` fills an array you own, for consumers that want to avoid an allocation per
read. `out` must be an `ndarray` with exactly the frame's shape and dtype (byte order
included), writable and C-contiguous. It is checked before anything is written, and a
rejected `out` is left unchanged: a non-array or a different dtype raises `TypeError`; a
different shape, a non-C-contiguous or a read-only array raises `ValueError`. On success it
returns `out`.

## `read()` in its three places

- `Accumulator.read()` returns a **new array** with the current representation. It changes
  nothing and can be called any number of times.
- `Engine.snapshot()` returns the **latest publication**, shared, not copied. Reading
  consumes nothing: every consumer, and every repeated read, sees the same snapshot until
  the next publication.
- `SnapshotPublisher.read()` (`frames2py.publish`) is what `Engine.snapshot()` calls: the
  latest complete publication, or `None`.

## Publication, windows and `reset()`

- `snapshot()` is `None` before the first publication, and again after `reset()` until the
  next one. The first `ingest()`, and the first after `reset()`, always publishes.
- For windowed kernels (`event_count`, `polarity`), each snapshot holds only the events of
  its window, between the previous publication and this one. For running kernels, each
  snapshot holds everything since construction or `reset()`.
- `stop()` publishes the pending window, if any in-bounds events arrived since the last
  publication, and the last snapshot stays readable while stopped.
- `sequence` keeps increasing across `reset()`: the first snapshot after a reset has a
  higher number than any before it.
- A read that overlaps a `reset()` on another thread returns either the snapshot from before
  the reset or `None`. Every read that starts after `reset()` has returned gets `None` until
  the next publication.

## How often to read

Read at about the publication cadence (16 ms by default), not in a tight loop: a tight loop
spends a core re-reading the same snapshot. Compare `snapshot.meta.sequence` with the last
one you handled to tell a new publication from a repeat. [Writing a
consumer](../consumers/writing-a-consumer.md) has a complete producer and consumer.

Holding on to a snapshot keeps its frame alive. The Engine only keeps the latest; a consumer
that stores snapshots stores their memory.
