# Engine

`frames2py.Engine(sensor_size, kernel="event_count", *, snapshot_interval_ms=16.0)` is the
live runtime: it accumulates events from one producer thread and publishes snapshots for any
number of consumers.

```python title="quickstart.py"
--8<-- "quickstart.py"
```

## Constructor

- `sensor_size`: `(width, height)`.
- `kernel`: a configured kernel instance, or `"event_count"` (the default), `"polarity"` or
  `"time_surface"`. `ExpDecay` and `TimestampDecay` take a parameter, so they are always
  passed as instances; `"exp_decay"` as a name raises `ValueError`.
- `snapshot_interval_ms` (keyword-only): the publication cadence, below.

On a free-threaded CPython build running with the GIL disabled, construction raises
`RuntimeError` unless that minor version has been verified; today that is 3.14 only (see
[Lifecycle and threads](lifecycle.md#free-threaded-cpython)).

## `ingest(events)`

Accumulates one call's events and publishes if the cadence allows, all on the caller's
thread, before returning. The steps are on the [Architecture](architecture.md#the-ingest-path)
page and the validation rules in the [event contract](event-contract.md).

- Raises `RuntimeError` if called from a thread other than the producer's, changing nothing.
  The first `ingest()` that doesn't raise makes its thread the producer, for the Engine's
  lifetime.
- Raises `TypeError` for a malformed array and `ValueError` if any event has
  `t >= 2**63`, in both cases before any state or statistic changes.
- A no-op while the Engine is stopped.

## Publication cadence

- `snapshot_interval_ms=0` publishes on every `ingest()`.
- A positive interval publishes at most once per interval, on the first `ingest()` after it
  has elapsed since the last publication (measured with `time.monotonic_ns()`).
- The first `ingest()` always publishes, and so does the first after `reset()`. That includes
  an empty call, or one whose events are all out of bounds: the snapshot then carries
  `watermark=None` until an in-bounds event arrives.
- `stop()` publishes if in-bounds events were accumulated since the last publication.
- Nothing else publishes. There is no timer thread, so if the producer stops calling
  `ingest()` mid-window, that window's events stay unpublished until the next `ingest()` or
  `stop()`.

For windowed kernels (`event_count`, `polarity`) each publication closes the window: the
next snapshot counts only the events after it. Running kernels are not changed by
publication.

The 16 ms default is about one publication per frame of a 60 Hz display. A consumer that wants every publication should
read at about the same cadence; one that reads less often simply sees a later snapshot.

## `snapshot()`

Returns the latest published [`Snapshot`](snapshots.md), or `None` before the first
publication and after `reset()` until the next. It returns the published object itself, not
a copy: every consumer that reads the same publication gets the same object, with a
read-only frame. Takes no lock. Safe from any thread.

The Engine has no `watermark` attribute of its own: the watermark at each publication is
`snapshot.meta.watermark`.

## Seeing every publication

Publication happens only inside `ingest()` and `stop()`,
which run on the threads that call them. So code that reads `snapshot()` right after each
of its own `ingest()` calls, and once after `stop()`, sees every publication, provided no
other thread calls `reset()` meanwhile. A consumer on another thread that polls at its own
pace may skip publications; that is by design. Summing the windows of a windowed kernel
this way gives totals over the whole stream (an [Accumulator](accumulator.md) gives them
without publication):

```python title="engine_windows.py"
--8<-- "engine_windows.py"
```

```text title="Output"
--8<-- "engine_windows.out"
```

## `stats`

An `EngineStats`, a frozen dataclass, built when you read the property. Takes no lock.

| field | meaning |
|---|---|
| `events_ingested` | every event of every accepted `ingest()` call, out-of-bounds ones included |
| `events_out_of_bounds` | the subset the bounds check rejected |
| `snapshots_published` | publications since construction or the last `reset()` |
| `uptime_ns` | nanoseconds since the Engine was constructed; `reset()` doesn't change it |

The accumulated count is `events_ingested - events_out_of_bounds`; there is no separate
field. Every event of an accepted call to a running Engine is either accumulated or counted
as out of bounds. A call rejected by validation or by the timestamp-range check changes no
statistic, and neither does `ingest()` while stopped.

`stats` is a diagnostic view: its fields are read one after another while the producer may
be running, so on a live Engine they are not one instant-consistent snapshot of all
counters.

## `start()`, `stop()`, `reset()`

In short: `stop()` publishes any pending window and makes `ingest()` a no-op, leaving the
last snapshot readable; `start()` resumes; `reset()` clears the kernel state, counters,
watermark and published snapshot. They may be called from any thread. The details are in
[Lifecycle and threads](lifecycle.md).
