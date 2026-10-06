# Lifecycle and threads

## States and calls

An Engine is running from construction. `stop()` stops it, `start()` resumes it, and
`reset()` clears it without changing whether it is running.

| call | effect |
|---|---|
| `ingest(events)` | running: accumulate, publish if the cadence allows. Stopped: a no-op (after the producer check). |
| `stop()` | if in-bounds events were accumulated since the last publication, publish them; then stop. With nothing pending it publishes nothing, so a windowed kernel's last frame is not replaced by an empty one. The latest snapshot stays readable. |
| `start()` | resume ingestion. |
| `reset()` | clear the kernel state, `events_ingested`, `events_out_of_bounds`, `snapshots_published`, the watermark and the published snapshot. The next `ingest()` publishes, like the first. |
| `snapshot()`, `stats`, `wait_for_newer()` | read-only; never change anything. |

**What `reset()` leaves alone:** `SnapshotMeta.sequence` keeps counting (the Engine's
lifetime, not the session's), `stats.uptime_ns` keeps measuring from construction, the
running or stopped state is unchanged, and the producer thread stays the producer. The
Engine stays usable; there is no need to build a new one.

Call `reset()` when your source's timestamps restart or become invalid (see [timestamp
discontinuities](event-contract.md#timestamp-discontinuities)), or to start a new session
on the same sensor.

## Threads

**One producer.** The first `ingest()` that doesn't raise makes its thread the producer, for
the Engine's lifetime: `reset()` doesn't release it. A no-op `ingest()` while stopped counts;
one rejected with `TypeError` or `ValueError` doesn't. `ingest()` from any other thread
raises `RuntimeError` and changes nothing. An Engine is not multi-producer; a different
producer thread needs a new Engine.

The producer is its `threading` thread object, not its thread ident: a thread started after
the producer exits is another thread, even when CPython gives it the producer's ident. One
case remains on CPython 3.11 and 3.12. A thread created outside `threading`, by C code such
as a vendor SDK's callback thread, is represented by a dummy thread object that those
versions keep after the thread ends. If such a thread is the producer and exits, and a later
thread created outside `threading` gets its ident, the later thread is given the same object
and treated as the producer: its `ingest()` is accepted and its `wait_for_newer()` refused.
CPython 3.13 and later discard the dummy object when its thread ends.

**Any number of consumers.** `snapshot()` and `stats` may be called from any thread, at any
time, and take no lock. `wait_for_newer()` may be called from any thread but the producer's
(it raises `RuntimeError` there); it never takes the lifecycle lock, and `stop()` and
`reset()` wake no waiter
([details](snapshots.md#waiting-for-a-newer-snapshot)).

**Lifecycle calls from anywhere.** `start()`, `stop()` and `reset()` may be called from any
thread. They and `ingest()` take one private lock, so each runs whole and never interleaves
with another: a `stop()` from a control thread waits for an `ingest()` in progress to
finish, then runs. These calls are control, not consumers; consumers never take the lock.

```python title="producer_consumer.py"
--8<-- "producer_consumer.py"
```

```text title="Output"
--8<-- "producer_consumer.out"
```

## What "never waits on consumers" covers

The producer path has no synchronisation that a consumer can hold or control: no lock a
reader takes, no condition variable, no retry loop driven by readers, no queue waiting for
consumers, no buffer a consumer must hand back. That is a statement about Frames2Py's own
code. It runs inside CPython, and these can still delay the producer thread:

- the GIL, on standard builds, when a consumer thread is running Python code;
- CPython's own per-object locks, such as the list lock taken when the snapshot is stored,
  and the waiter registry's lock, which `wait_for_newer()` callers take only briefly inside
  CPython, never while running Python code;
- the work of releasing waiters: each publication releases every thread registered in
  `wait_for_newer()`, one lock release each, so it costs more with more waiters
  ([measured](snapshots.md#what-waiting-costs));
- garbage collection, the memory allocator, and the operating system's scheduling;
- CPU and memory contention from whatever else the machine runs, consumers included.

`ingest()` also does real CPU work on the caller's thread: "never waits" does not mean
"fast". The [Performance](../reference/performance.md) page has measurements, including a
producer with one consumer and with the viewer running.

## Standard CPython

On standard CPython 3.11 and later (the GIL enabled), everything above holds. Only one
thread runs Python code at a time, so a consumer doing heavy Python work competes with the
producer for the GIL. Large NumPy operations can overlap: measured on CPython 3.11, a
frame copy at 640x480 and at 1280x720 let other Python threads run while it was in
progress.

## Free-threaded CPython

- **Supported:** CPython **3.14t with the GIL disabled**, on Linux x86_64, Linux ARM64 and
  macOS ARM64. The full test suite, including the concurrency tests, runs there in CI, with
  the GIL checked to stay disabled after NumPy and every optional backend are imported.
- **The tested model** is the one described on this page: one producer thread calling
  `ingest()`, any number of threads calling `snapshot()` and `stats`, lifecycle calls from
  any thread. The concurrency tests cover these; they are tests and bounded interleaving
  checks, not a proof, and Frames2Py makes no thread-safety claim beyond this model.
- **Refused:** every other free-threaded minor version with the GIL disabled.
  `Engine(...)` raises `RuntimeError` there. That includes 3.13t, and it includes 3.15t and
  later until each is verified: the snapshot hand-off relies on CPython source behaviour
  (see [Architecture](architecture.md#how-the-hand-off-works)) that is checked per minor
  version.
- **Recommended patch level: 3.14.5 or later.** CPython 3.14.0 to 3.14.4 have a race in an
  internal lock ([gh-148820](https://github.com/python/cpython/issues/148820)) that can end
  the process with a fatal error when a signal or a spurious wakeup lands while threads
  contend for that lock. `wait_for_newer()`'s blocking path uses the locking code that fix
  changed: a waiter blocked in `Lock.acquire()` and the publication that releases it both
  take CPython's internal parking-lot mutex. The Engine doesn't check the patch level. The
  race was not reproduced in Frames2Py's tests.
- **GIL enabled:** a free-threaded build running with the GIL enabled (for example
  `PYTHON_GIL=1`) behaves as a standard build and is treated as one.
- **Throughput** on 3.14t has been measured on one machine (Apple M4); see
  [Performance](../reference/performance.md).

The package metadata carries no free-threading classifier, because the classifiers can't
say "3.14t only".
