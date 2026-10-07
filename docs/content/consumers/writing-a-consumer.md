# Writing a consumer

A consumer is any code that reads the Engine's published state: a display, a monitor, a
logger, a second algorithm that works on the accumulated state. It needs no registration:
it calls `snapshot()` when it wants the latest state, or `wait_for_newer()` to block until
there is a newer one.

```python title="producer_consumer.py"
--8<-- "producer_consumer.py"
```

```text title="Output"
--8<-- "producer_consumer.out"
```

## The pattern

1. **Run the producer on its own thread** and let it call `ingest()` in its loop. It owns
   the Engine's ingest path; consumers never slow it down by waiting.
2. **Wait for the next publication,** as above: pass the last `meta.sequence` you handled
   to `wait_for_newer()`, with a timeout so the loop can notice its own stop flag. It
   returns as soon as there is something newer, and `None` on timeout. Its semantics are in
   [Waiting for a newer snapshot](../core/snapshots.md#waiting-for-a-newer-snapshot).
3. **Or poll at the publication cadence,** when the consumer runs on its own clock (a
   display loop, for one, like the [viewer](viewer.md)): call `snapshot()` about every
   `snapshot_interval_ms`. Reading faster only returns the same snapshot again. Polling
   sees a state up to one poll interval later than waiting; waiting wakes every waiter at
   each publication, which costs the producer more as waiters are added
   ([What waiting costs](../core/snapshots.md#what-waiting-costs)).
4. **Detect new publications by `meta.sequence`.** It is strictly increasing for the
   Engine's lifetime, across `reset()`: a higher number than the last one you handled means
   a new publication. A consumer slower than the cadence skips publications, which is
   normal; the sequence doesn't say how many were skipped.
5. **Handle `None`.** `snapshot()` is `None` before the first publication and after
   `reset()` until the next; `wait_for_newer()` returns `None` when it times out.
6. **Copy before you modify.** `snapshot.frame` is shared with every other consumer and
   read-only. Use `snapshot.copy()` (or `copy(out=...)` into a buffer you keep) before
   writing to the data or handing it to a library that ignores NumPy's read-only flag, such
   as `torch.from_numpy` ([Handing snapshots to PyTorch](pytorch.md)).

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

## An end-to-end example

[`examples/live_observation.ipynb`](https://github.com/siddiquifaras/frames2py/blob/main/examples/live_observation.ipynb)
puts this pattern in a realistic setting: a producer replays a synthetic event stream at its
own pace into two Engines, a tracker and a deliberately slow model-like consumer wait with
`wait_for_newer()`, a monitor polls, the producer calls `reset()` between two segments, and
the notebook shows from the snapshots' `sequence` and `watermark` what each consumer saw and
how stale the slow one's results were. Its data is synthetic, generated in the notebook. It
runs from a repository checkout with the `notebook` dependency group; see
[Contributing](../development/contributing.md#the-end-to-end-notebook).

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
