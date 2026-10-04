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

- **`frame`**: the published array, `(height, width)` or `(height, width, 2)`, or
  time-first for the [temporal kernels](kernels.md#temporal-kernels), with the kernel's
  output dtype. It is the published array itself, not a copy, shared by every
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
[Handing snapshots to PyTorch](../consumers/pytorch.md) shows the copy-first recipe.

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
  snapshot holds everything since construction or `reset()`; the temporal kernels show the
  part of it in their most recent completed bins.
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
one you handled to tell a new publication from a repeat, or block in `wait_for_newer()`
(below) until there is one. [Writing a consumer](../consumers/writing-a-consumer.md) has a
complete producer and consumer.

## Waiting for a newer snapshot

`engine.wait_for_newer(sequence, *, timeout=None)` blocks until a snapshot newer than
`sequence` is published, then returns the latest one:

- **Newer** means `snapshot.meta.sequence > sequence`. `None` accepts any publication.
  `sequence` must be an integer (an `int`, or a NumPy integer) or `None`: a negative one
  raises `ValueError`, and a bool or any other type `TypeError`.
- **The latest, not the next.** If a newer snapshot is already published, the call returns
  it at once. Otherwise it returns the snapshot published when it reads after a
  publication, which needn't be `sequence + 1`: a slow consumer skips publications here
  too. Passing back the `meta.sequence` you got gives strictly increasing snapshots.
- **`timeout`** is in seconds on the monotonic clock: `None` or `math.inf` waits without
  limit, `0` checks once without blocking. On timeout the call returns `None`, and only if
  nothing newer is published when it checks after the timeout has elapsed. Negative or NaN
  raises `ValueError`; a bool or a non-number raises `TypeError`.
- **Any number of waiters.** Each thread waits with its own `sequence`. A publication
  wakes every waiter registered before it, and each one returns whatever is latest when it
  reads, so two waiters woken together can return different snapshots if another
  publication comes in between.
- **No missed wakeup.** A waiter registers before it checks the latest snapshot for the
  last time, so a publication can't slip in between the check and the wait: a waiter is
  never left blocked after a publication that leaves a snapshot newer than its `sequence`
  published. It returns that snapshot, unless a `reset()` clears it first (below). There
  is one exception, an interrupted producer (last item).
- **`stop()` and `reset()` wake nobody.** The publication `stop()` makes for a pending
  window wakes waiters like any other. After `reset()`, a waiter returns the first
  publication after it, whose sequence is higher than any before the reset; a reset that
  lands between a publication and the waiter's read leaves the waiter waiting for the next
  one. A consumer that must notice shutdown uses a timeout and its own flag.
- **Not on the producer's thread.** Once the Engine has a producer, calling it from that
  thread raises `RuntimeError`: that thread can't publish while it waits.
- **Ctrl-C** in a main-thread waiter raises `KeyboardInterrupt` and leaves the Engine as it
  was, as CPython's `Lock.acquire()` allows: a SIGINT arriving as the wait starts blocking
  is acted on at its next wake, which is a publication, the timeout, or another SIGINT.
- **Pass a timeout if the producer can be interrupted.** If Ctrl-C interrupts the
  producer's thread during a publication, one waiter may stay blocked until its timeout,
  and without a timeout indefinitely: the interrupted publication may have taken that
  waiter off the registry without waking it. Later publications work normally and wake
  every other waiter.

```python title="wait_for_newer.py"
--8<-- "wait_for_newer.py"
```

```text title="Output"
--8<-- "wait_for_newer.out"
```

A consumer that handles every state it is given loops on its last sequence, with a timeout
so it can notice its own stop flag:

```python
# Sketch (not runnable): engine, stopping and handle() are yours.
last = None
while not stopping.is_set():
    snapshot = engine.wait_for_newer(last, timeout=0.1)
    if snapshot is None:
        continue            # timed out: nothing newer yet
    last = snapshot.meta.sequence
    handle(snapshot)        # publications during handle() are skipped, not queued
```

[Writing a consumer](../consumers/writing-a-consumer.md) runs this loop against a real
producer.

### What waiting costs

**The mechanism.** Each publication does one operation on the waiter registry even when
nobody waits, then releases the waiters registered before it, one lock release per waiter,
on the publishing thread. It never waits for a waiter, and a waiter never holds anything
the producer needs while running Python code. CPython's own synchronisation (the GIL, its
internal locks, scheduling) is outside that, as it is for `snapshot()`.

**Measured on one machine.** A preregistered measurement compared the Engine with
`wait_for_newer` against the build without it: an Apple M4 (16 GB), CPython 3.11.14 and
3.14.2t with the GIL disabled, NumPy 2.4.6, `event_count`, uniform synthetic events, 5 runs
per cell. The method and the data file are on
[Benchmark methodology](../reference/methodology.md#the-wait_for_newer-measurement). Nothing
here holds for other hardware, kernels or workloads without measuring them.

- **No waiter.** In the 9 `Engine.ingest()` cells per runtime (346x260, 640x480 and
  1280x720; 10k-event calls publishing every call, 100k-event calls publishing every call
  and every 16 ms), no cell was distinguishably slower than without the feature, at the
  measurement's resolution. The ratios of the medians were 0.94-1.08, and by the
  preregistered rule a difference counts only if it exceeds both the variation of the
  baseline measured against itself and the per-run ranges. That is not a finding of no
  cost: a smaller difference would not have been detected.
- **Waiters make publications slower,** most visibly when every call publishes a small
  batch. With 8 waiters and a publication on every call, the producer ingested 0.46-0.89x
  as many events per second as with none on 3.11.14, and 0.52-0.93x on 3.14.2t; the lowest
  were 10k-event calls at 346x260 and 640x480. At the default 16 ms interval with
  100k-event calls, it stayed at 0.91-1.05x with 1, 4 or 8 waiters on both runtimes.
- **Waiting against polling.** At 1280x720, with 100k-event batches arriving at 20M
  events/s on the real clock, a 16 ms interval, and consumers that each render every state
  they get (`viewer.render()`), consumers waiting with `wait_for_newer` were compared with
  consumers calling `snapshot()` every 16 ms:

    | consumers | runtime | freshness p50, waiting / polling | producer busy time per event, waiting / polling |
    |---|---|---|---|
    | 1 | 3.11.14 | 0.53 / 6.96 ms | 4.48 / 4.65 ns |
    | 1 | 3.14.2t | 0.50 / 8.65 ms | 4.56 / 4.61 ns |
    | 4 | 3.11.14 | 0.69 / 8.10 ms | 10.7 / 8.08 ns |
    | 4 | 3.14.2t | 0.56 / 8.18 ms | 10.1 / 7.10 ns |
    | 8 | 3.11.14 | 18.0 / 17.9 ms | 56.9 / 55.6 ns |
    | 8 | 3.14.2t | 7.83 / 8.91 ms | 24.3 / 20.6 ns |

    With 1 and 4 consumers, waiters' median freshness was under a millisecond and they saw
    every publication; pollers saw each state up to a poll interval later and missed 2-8%
    of the publications. With 4 and 8 consumers, waiting cost the producer more busy time
    per event than polling: every waiter is woken by the same publication and renders while
    the producer is still working (an inference from the design, not measured). With 8
    rendering consumers on 3.11.14, which share one GIL with the producer, the producer kept
    up in neither arm: it ingested 0.59-0.60 of the offered events.

**What the numbers mean.** *Freshness* is the time from the start of the `ingest()` call
that ingested the newest event in the state a consumer received, to the moment the
consumer received it. It is measured from that producer step, not from the publication,
which happens inside the step and can't be observed from outside. *Producer busy time per
event* is the total time spent inside `ingest()` calls divided by the events ingested. The
events-per-second figures are the events in the timed `ingest()` calls divided by the sum
of their durations. Each figure is the median over 5 runs; the ranges are in the data file.

Holding on to a snapshot keeps its frame alive. The Engine only keeps the latest; a consumer
that stores snapshots stores their memory.
