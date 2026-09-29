# Writing a consumer

A consumer is any code that reads `engine.snapshot()`: a display, a monitor, a logger, a
second algorithm that works on the accumulated state. It needs no registration: it calls
`snapshot()` when it wants the latest state.

```python title="producer_consumer.py"
--8<-- "producer_consumer.py"
```

```text title="Output"
--8<-- "producer_consumer.out"
```

## The pattern

1. **Run the producer on its own thread** and let it call `ingest()` in its loop. It owns
   the Engine's ingest path; consumers never slow it down by waiting.
2. **Poll at the publication cadence.** Sleep about `snapshot_interval_ms` between reads.
   Reading faster only returns the same snapshot again.
3. **Detect new publications by `meta.sequence`.** It increases by one per publication, for
   the Engine's lifetime. A jump of more than one means you skipped publications, which is
   normal for a consumer slower than the cadence.
4. **Handle `None`.** `snapshot()` is `None` before the first publication and after
   `reset()` until the next.
5. **Copy before you modify.** `snapshot.frame` is shared with every other consumer and
   read-only. Use `snapshot.copy()` (or `copy(out=...)` into a buffer you keep) before
   writing to the data or handing it to a library that ignores NumPy's read-only flag, such
   as `torch.from_numpy`.

## What a consumer gets, and doesn't

- **The latest state, not every event.** Snapshots skip publications a slow consumer
  missed, and a windowed kernel's frame covers only its own window. A consumer that needs
  every event needs the events: call it from the producer's loop, next to `ingest()` (as
  the [recorder](../data/recorder.md) is used), accepting that its cost is then the
  producer's.
- **A consistent snapshot.** Frame and metadata are from one publication, and the frame is
  complete.
- **No ordering promise across consumers.** Two consumers reading at the same moment may see
  different publications if one was published in between.

## Keeping up

A consumer's own speed is its own concern. Frames2Py keeps only the latest snapshot and
never queues publications for a slow consumer, so a consumer that falls behind loses
publications, not memory. The flip side: holding references to old snapshots keeps their
frames alive; each publication is a fresh array (about 3.5 MiB for `event_count` at
1280x720).

On standard CPython, a consumer doing heavy pure-Python work competes with the producer for
the GIL. Large NumPy operations can let other threads run (measured for frame copies on
CPython 3.11), and a free-threaded 3.14t build has no GIL to compete for. See
[Lifecycle and threads](../core/lifecycle.md).
