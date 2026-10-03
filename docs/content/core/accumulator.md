# Accumulator

`frames2py.Accumulator(sensor_size, kernel)` accumulates events through one kernel,
synchronously, on the calling thread. It is what the [Engine](engine.md) uses internally,
without publication, threads or lifecycle. Use it when you drive the loop yourself and want
the representation on demand: offline processing of a recording, tests, a pipeline stage
that runs in step with its input.

```python title="accumulator.py"
--8<-- "accumulator.py"
```

```text title="Output"
--8<-- "accumulator.out"
```

## Constructor

- `sensor_size`: `(width, height)`. The representation is `(height, width)`, or
  `(height, width, 2)` for `polarity`; the temporal kernels are time-first, `(2, bins, height,
  width)` for `StackedHistogram` and `(bins, height, width)` for `VoxelGrid`.
- `kernel`: a configured kernel instance (`frames2py.ExpDecay(0.9)`,
  `frames2py.TimestampDecay(10_000.0)`, `frames2py.VoxelGrid(bins=5, bin_us=10_000)`, any object implementing the
  [Kernel protocol](../reference/api/kernels.md)), or one of the names `"event_count"`,
  `"polarity"`, `"time_surface"` for the kernels without parameters. Any other name raises
  `ValueError`. Unlike `Engine`, `Accumulator` has no default kernel.

## Methods and properties

**`accumulate(events)`** accumulates one call's events, in the order of the
[event contract](event-contract.md#what-happens-to-a-call): structural validation
(`TypeError`), the timestamp-range check (`ValueError` if any `t >= 2**63`, rejecting the
whole call before anything changes), the bounds check (out-of-bounds events are counted and
otherwise ignored), then the kernel. For `exp_decay`, each accepted call is one decay step.

**`read()`** returns a new array holding the current representation, with the kernel's
public dtype. It changes no state: reading twice gives the same result, and reading never
closes a window. (The Engine's windowed kernels start a new window at each publication; an
Accumulator never publishes, so its `event_count` and `polarity` windows run from
construction or `reset()`.)

**`watermark`** is the largest timestamp among accumulated in-bounds events, or `None`
before the first. Out-of-bounds events never move it.

**`events_out_of_bounds`** counts the events the bounds check has rejected.

**`reset()`** clears the kernel state, the watermark and the out-of-bounds count, as if the
Accumulator were new. Call it when your source's timestamps restart
([timestamp discontinuities](event-contract.md#timestamp-discontinuities)).

The full signatures are in the [API reference](../reference/api/core.md).

## Accumulator or Engine?

| | `Accumulator` | `Engine` |
|---|---|---|
| threads | the caller's only | one producer, any number of consumers |
| output | `read()`: a copy, on demand | `snapshot()`: the latest publication, shared |
| windowed kernels | window spans everything since construction or `reset()` | each publication starts a new window |
| statistics | `watermark`, `events_out_of_bounds` | `stats` (`EngineStats`), and the watermark in each `SnapshotMeta` |
| lifecycle | `reset()` | `start()`, `stop()`, `reset()` |
| default kernel | none | `"event_count"` |
