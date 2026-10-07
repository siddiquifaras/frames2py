# Event contract

Everything that enters Frames2Py, through `Engine.ingest()`, `Accumulator.accumulate()`,
the recorder's `write()`, `replay.paced()` or `replay.windows()`, is a NumPy array of events
with this layout.

## `EVENT_DTYPE`

```python
import frames2py

print(frames2py.EVENT_DTYPE)
print(frames2py.EVENT_DTYPE.itemsize, "bytes per event")
```

```text title="Output"
[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')]
13 bytes per event
```

| field | dtype | byte offset | meaning |
|---|---|---|---|
| `t` | `<u8` (uint64, little-endian) | 0 | timestamp, microseconds |
| `x` | `<u2` (uint16) | 8 | column, 0 at the left |
| `y` | `<u2` (uint16) | 10 | row, 0 at the top |
| `p` | `u1` (uint8) | 12 | polarity: `0` is OFF, any other value is ON |

13 bytes, packed, no padding. `(0, 0)` is the top-left pixel; `x` indexes columns and `y`
rows, so the event lands in `frame[y, x]`.

**Polarity.** Only `p == 0` versus `p != 0` matters. `p = 1`, `p = 7` and `p = 255` are all
ON, and no value of `p` can address another pixel or channel.

**Timestamps** are microseconds in whatever time domain your source uses: sensor clock,
Unix time, zero at the start of a recording. Frames2Py never shifts, sorts or clamps them.
They may arrive out of order, within a call and across calls.

## Building event arrays

Any 1-D structured array whose `t`, `x`, `y` and `p` fields have exactly these dtypes is
accepted. The simplest way is to allocate one and fill the fields:

```python
import numpy as np

t = np.array([10, 30, 20], dtype=np.uint64)
x = np.array([1, 2, 3])
y = np.array([0, 0, 1])

events = np.zeros(len(t), dtype=frames2py.EVENT_DTYPE)
events["t"], events["x"], events["y"], events["p"] = t, x, y, 1
print(events)
```

```text title="Output"
[(10, 1, 0, 1) (30, 2, 0, 1) (20, 3, 1, 1)]
```

Assigning an array into a field casts it to the field's dtype, and NumPy's cast wraps values
outside the field's range silently (an int64 `-1` becomes `x = 65535`). Check your source's
ranges before converting; Frames2Py can only see the converted values.

**Extra fields are ignored**, whatever their dtype, so an array that carries more per event
can be passed as it is:

```python
richer = np.zeros(2, dtype=frames2py.EVENT_DTYPE.descr + [("confidence", "<f4")])
richer["t"] = [1, 2]

engine = frames2py.Engine((4, 3))
engine.ingest(richer)
print(engine.stats.events_ingested)
```

```text title="Output"
2
```

## What happens to a call

Each call goes through these steps, in this order. A call that fails a step raises and
changes nothing: no state, no watermark, no statistic.

**1. Structural validation** (`TypeError`). It checks the array's *structure* and never looks
at a value:

- a NumPy `ndarray`, structured, with fields `t`, `x`, `y`, `p`;
- each of those fields with exactly its `EVENT_DTYPE` dtype, byte order included (a
  big-endian `>u8` or a `uint32` timestamp is rejected; castable dtypes are not accepted);
- one-dimensional;
- C-contiguous. A strided view such as `events[::2]` is not; pass
  `np.ascontiguousarray(events[::2])`.

```python
wrong = np.zeros(3, dtype=[("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])
try:
    engine.ingest(wrong)
except TypeError as error:
    print("TypeError:", error)

try:
    engine.ingest(events[::2])
except TypeError as error:
    print("TypeError:", error)
```

```text title="Output"
TypeError: field 't' must be <u8, got <u4
TypeError: events must be C-contiguous
```

**2. Timestamp range** (`ValueError`). If any event in the call, out-of-bounds ones included,
has `t >= 2**63`, the whole call is rejected. 2^63 µs is about 292,000 years; a timestamp
that large is a corrupt or mis-converted value (for example a negative int64 cast to
uint64).

```python
bad = np.zeros(2, dtype=frames2py.EVENT_DTYPE)
bad["t"] = [100, 2**63]
try:
    engine.ingest(bad)
except ValueError as error:
    print("ValueError:", error)
print("still", engine.stats.events_ingested, "events ingested")
```

```text title="Output"
ValueError: an event has t >= 2**63; the whole call is rejected
still 2 events ingested
```

**3. Bounds check.** An event with `x >= width` or `y >= height` is out of bounds. It is
counted in `events_out_of_bounds` and has no other effect: it doesn't reach the kernel and
doesn't move the watermark. The check is one `max()` over `x` and one over `y`; only a call
that fails it pays for a mask. (Coordinates are unsigned, so there is no lower bound to
check.)

**4. Accumulation.** The kernel receives the call's in-bounds events. Each is accumulated
under the kernel's rules, whatever its timestamp.

So after an accepted call on a running Engine, every event of it is either accumulated or
counted as out of bounds, and `stats.events_ingested` has grown by the call's length.

## The watermark

The watermark is the largest timestamp among the in-bounds events accumulated since
construction or the last `reset()`, and `None` before the first. It is the maximum, not the
last element: an out-of-order call moves it to the call's largest in-bounds timestamp or
leaves it where it was. Out-of-bounds events never move it, and neither does publication or
the end of a window.

It is `Accumulator.watermark`, and each snapshot's `SnapshotMeta.watermark` is its value at
publication. `timestamp_decay` evaluates its surface at it, the temporal kernels show the
completed time bins before it, and the viewer's `time_surface` rendering fades pixels
relative to it.

## Exceptions

| condition | exception | when |
|---|---|---|
| not an ndarray, not structured, a missing field, a field of the wrong dtype or byte order, not 1-D, not C-contiguous | `TypeError` | before anything changes |
| any event with `t >= 2**63` | `ValueError` | before anything changes |
| `Engine.ingest()` from a thread other than the producer's | `RuntimeError` | before anything changes |
| `Engine.ingest()` while stopped | none; the call is a no-op | |

## Timestamp discontinuities

**Frames2Py does not detect or repair timestamp discontinuities. When your source's
timestamp domain restarts or becomes invalid, call `reset()`.**

In particular it does not detect clock resets, infer epochs, unwrap counters, drop
far-future events, warn, or reset itself. It can't reliably tell a legitimately
out-of-order event from a restarted clock, so it doesn't guess: every in-bounds event of an
accepted call is accumulated under the watermark rule above.

What that means in practice, as the test suite checks it (the same whether the jump falls
inside one call or between calls):

**A backward jump**, such as a source restarting near 0 after running for a while:

- the watermark, and `SnapshotMeta.watermark`, stay at the old maximum;
- `event_count`, `polarity` and `exp_decay` accumulate the new events as usual; their values
  don't depend on timestamps;
- `time_surface`: a pixel holding a larger old-clock timestamp keeps it; other pixels take
  the restarted clock's timestamps;
- `timestamp_decay`: each restarted-clock event contributes `exp(-(T - t) / tau_us)` at the
  unchanged watermark `T`, which for a large jump reads as 0; earlier contributions are
  unchanged;
- `stacked_histogram` and `voxel_grid`: the frame stays on the bins before the old
  watermark, so the restarted clock's events are older than the frame and count for
  nothing.

**A forward spike**, one far-future timestamp:

- the watermark advances to the spike and stays there;
- counts and `exp_decay` as above; in `time_surface` only the spike's pixel changes;
- `timestamp_decay`: everything is now evaluated at the spike's time, so the other pixels,
  including events arriving after the spike with ordinary timestamps, decay by however far
  the spike is ahead of them, which for a large spike reads as 0;
- `stacked_histogram` and `voxel_grid`: the frame moves to the bins just before the spike's,
  so it shows zeros: earlier events, and ordinary events arriving after the spike, are
  older than the frame. `replay.windows()` yields a frame for every boundary up to the
  spike.

**Recovery:** after `reset()`, a clean timestamp sequence gives the same result as a fresh
Accumulator or Engine. The Engine's `snapshot()` is `None` until the next publication, which
carries the clean sequence's watermark.

```python title="discontinuity.py"
--8<-- "discontinuity.py"
```

```text title="Output"
--8<-- "discontinuity.out"
```

**Adapters** handle the one case that is not a discontinuity: a hardware timestamp counter
that is narrower than 64 bits and wraps. The [EVT adapter](../data/evt.md) unwraps the EVT
counters into continuous 64-bit microseconds. Wraps and restarts it can't tell apart are
passed through as they are, for you to handle.
