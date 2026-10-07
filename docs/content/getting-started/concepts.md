# Concepts

The vocabulary the rest of the documentation uses.

**Event.** One brightness change at one pixel: a timestamp `t` in microseconds, a column
`x`, a row `y` and a polarity `p` (0 is OFF, anything else ON). Frames2Py takes events as
1-D NumPy arrays of [`EVENT_DTYPE`](../core/event-contract.md), many per call.

**Sensor size and frame shape.** `sensor_size` is always `(width, height)`. The arrays
Frames2Py produces are `(height, width)`, or `(height, width, channels)`: row `y`, column
`x`, with `(0, 0)` at the top left. The two temporal kernels put time first: `(bins, height,
width)` and `(2, bins, height, width)`.

**Kernel.** The representation events accumulate into, one value (or one per channel) per
pixel. Frames2Py ships [seven](../core/kernels.md). A kernel is either:

- **windowed**: publication starts a new window, so each snapshot shows only the events
  since the previous one (`event_count`, `polarity`); or
- **running**: publication leaves the state alone, so each snapshot shows everything
  accumulated since construction or `reset()` (`time_surface`, `exp_decay`,
  `timestamp_decay`), or, for the two temporal kernels (`stacked_histogram`,
  `voxel_grid`), the part of it that falls in their most recent time bins.

**Accumulator.** Events in, representation out, synchronously, through one kernel. It
validates each call, checks the timestamp range and the sensor bounds, keeps the watermark
and counts out-of-bounds events. `read()` returns a copy of the current representation. No
threads, no publication. See [Accumulator](../core/accumulator.md).

**Engine.** An Accumulator plus publication, lifecycle and statistics, for live use. One
producer thread calls `ingest()`; any number of consumers call `snapshot()`, or block in
`wait_for_newer()` until there is a newer snapshot. See
[Engine](../core/engine.md).

**Publication and snapshot.** At most once per `snapshot_interval_ms`, inside `ingest()`
(and in `stop()`), the Engine reads the kernel into a fresh array and publishes it with its
metadata as one `Snapshot`. The frame is shared by every consumer that reads it and is
never written again. There is no timer thread: publication only happens inside those calls.
See [Snapshots and consumers](../core/snapshots.md).

**Watermark.** The largest timestamp among the in-bounds events accumulated since
construction or `reset()`; `None` before the first. It is not the last event's timestamp:
events may arrive out of order. `timestamp_decay` evaluates its surface at the watermark,
the temporal kernels place their time bins relative to it, and each snapshot carries the
watermark it was published at.

**Sequence.** Each publication's number, strictly increasing for the Engine's lifetime,
across `reset()`. A consumer that sees the same sequence twice has seen the same
publication twice.

**Producer and consumer.** The producer is the one thread that calls `ingest()`: the first
thread whose `ingest()` doesn't raise becomes it, for the Engine's lifetime. A consumer is
anything that reads `snapshot()` or `stats`, or waits in `wait_for_newer()`, from any other
thread. Consumers never make the
producer wait. See [Lifecycle and threads](../core/lifecycle.md).

**Adapter.** A module that reads a recording and yields `EVENT_DTYPE` arrays: EVT 2.0 /
3.0, AEDAT 4.0, HDF5. It owns no Engine and starts no thread. See
[Adapters](../data/adapters.md).
