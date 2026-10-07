# Kernels

A kernel defines what the events accumulate into. Frames2Py ships seven: five whose output is
one image-shaped frame, and two [temporal kernels](#temporal-kernels) whose output keeps time
bins, for models and other consumers of temporal representations. Their output shapes and
dtypes are part of the public API; how each stores its state internally is its own business
and may differ.

| kernel | construct with | output | mode | each in-bounds event |
|---|---|---|---|---|
| [`EventCount`](#eventcount) | `"event_count"` or `EventCount()` | `(H, W)` uint32 | windowed | adds 1 at its pixel |
| [`Polarity`](#polarity) | `"polarity"` or `Polarity()` | `(H, W, 2)` uint32 | windowed | adds 1 at its pixel, channel 0 (OFF) or 1 (ON) |
| [`TimeSurface`](#timesurface) | `"time_surface"` or `TimeSurface()` | `(H, W)` uint64 | running | keeps the largest `t` at its pixel |
| [`ExpDecay`](#expdecay) | `ExpDecay(decay)` | `(H, W)` float32 | running | adds 1; the surface decays once per call |
| [`TimestampDecay`](#timestampdecay) | `TimestampDecay(tau_us)` | `(H, W)` float32 | running | adds a weight that decays with event time |
| [`StackedHistogram`](#stackedhistogram) | `StackedHistogram(bins=..., bin_us=...)` | `(2, bins, H, W)` uint32 | running | adds 1 in its time bin and polarity channel |
| [`VoxelGrid`](#voxelgrid) | `VoxelGrid(bins=..., bin_us=...)` | `(bins, H, W)` float32 | running | adds +1 (ON) or -1 (OFF), split linearly between the two nearest knots in time |

`H, W` are the sensor's height and width. The classes are importable from `frames2py` and
from `frames2py.kernels`; the three names work wherever a kernel is accepted. Pass an
instance, `frames2py.Polarity()`, not the class `frames2py.Polarity`: a class is not checked
as such, and constructing the Engine or Accumulator with it raises an unrelated-looking
`TypeError` (`output_spec() missing 1 required positional argument`).

**Windowed** kernels start a new window at each Engine publication: a snapshot shows only
the events since the previous one. **Running** kernels are not changed by publication: a
snapshot shows everything accumulated since construction or `reset()`, except that the two
temporal kernels show only their most recent completed time bins. With an
`Accumulator`, which never publishes, a windowed kernel's window runs from construction or
`reset()`.

Every kernel is cleared by `reset()`, back to the state of a new one.

## EventCount

Events per pixel in the current window, as `(H, W)` uint32.

**Counts wrap modulo 2^32.** A pixel that receives its 4,294,967,296th event in one window
reads 0 again; counts never saturate. At 20M events/s, all on one pixel, a single window
would have to last about 3.6 minutes to wrap; with publication every 16 ms, windows are far
shorter. The test suite checks the wrap.

## Polarity

Events per pixel and polarity in the current window, as `(H, W, 2)` uint32: channel 0
counts OFF events (`p == 0`), channel 1 ON events (any other `p`). Same modulo-2^32 wrap as
`EventCount`, per channel.

## TimeSurface

The largest timestamp seen at each pixel, as `(H, W)` uint64: exact, with no conversion to
float. Because it keeps the maximum, event order doesn't matter: an older event arriving
late never overwrites a newer one.

**0 means "no event".** An event at `t = 0` is therefore indistinguishable from no event at
that pixel. That is a documented limitation of v1; if your timestamps can be 0, offset them.

## ExpDecay

An exponentially decaying event count, as `(H, W)` float32, with **decay per call**:

- once per accepted `accumulate()` or `ingest()` call, before the call's events are added,
  the whole surface is multiplied by `decay`;
- then each in-bounds event adds 1 at its pixel.

"Per call" means per call you make. A call that is empty, or whose events are all out of
bounds, still decays the surface; `ingest()` on a stopped Engine is a no-op and doesn't.
Frames2Py never splits a call into internal chunks that each decay.

**The result depends on how you batch events into calls.** The same events fed as one call
or as ten calls give different surfaces. That is the definition of this kernel, not an
artefact. If you want decay in event time, independent of batching, use `TimestampDecay`.

```python title="decay_batching.py"
--8<-- "decay_batching.py"
```

```text title="Output"
--8<-- "decay_batching.out"
```

**`decay`** must be a finite real with `0 < decay < 1`. Anything else (0, 1, a negative,
NaN, infinity, a value too large for a float) raises `ValueError` at construction.

**Numerics.** The state is float64, with the decay applied lazily as one global scale
factor, so a call costs time proportional to its events, not to the frame. When the scale
becomes very small (below 2^-959) it is folded into the stored values in one O(H x W) pass:
at `decay=0.95` that happens once every 12,960 calls, at `decay=0.5` once every 960. The
output is converted to float32 when read.

## TimestampDecay

An event-time exponential decay, as `(H, W)` float32. Each pixel reads

```text
D(T) = sum over its accumulated in-bounds events i of exp(-(T - t_i) / tau_us)
```

where `T` is the [watermark](event-contract.md#the-watermark). Each event contributes 1 at
its own timestamp and decays with the time elapsed in the event stream, whatever its
polarity.

- **Evaluated at the watermark.** `read()` and every snapshot evaluate `D` at the current
  watermark, and `SnapshotMeta.watermark` is that `T`. Without new in-bounds events the
  watermark doesn't move, so the surface doesn't change. A consumer that wants the surface
  at a later time `T2` can extrapolate exactly: `D(T2) = D(T) * exp(-(T2 - T) / tau_us)`.
  There is no `read(at=...)`.
- **Independent of order and batching.** In exact arithmetic the result is the same whatever
  order the events arrive in and however they are split into calls. In floating point the
  results may differ slightly; the test suite checks agreement within 1 float32 ULP on its
  workloads, which is a test tolerance, not a guarantee for every input.
- **`tau_us`**, the time constant in microseconds, must be finite and `> 0`; anything else
  raises `ValueError`. It has no default.
- **Extreme timestamps.** An accepted event far in the future moves the watermark there, so
  every earlier contribution decays to effectively 0, and later ordinary events contribute
  next to nothing until `reset()`. This follows from evaluating at the watermark; see
  [timestamp discontinuities](event-contract.md#timestamp-discontinuities).
- **Numerics.** The state is float64 relative to a reference time, rebased to the watermark
  once the watermark is more than 665 `tau_us` past it; rebasing doesn't change `D`.
  Timestamp differences are taken in int64 before conversion. Contributions that are far
  older than the watermark underflow to 0, below what the float32 output can show.
- **Global NumPy error modes.** Settings that make floating-point underflow raise
  (`np.seterr(under="raise")`, `np.seterr(all="raise")`) are not supported while ingesting
  into or reading this kernel. They raise `FloatingPointError`: in `read()`, and so in a
  publication, once a pixel's value falls below float32's smallest normal number (a single
  contribution about 88 `tau_us` old); in `accumulate()` or `ingest()` for an event about
  750 `tau_us` behind the reference. Other kernels are not checked under such settings
  either.

## Temporal kernels

`StackedHistogram` and `VoxelGrid` keep the recent past in time bins, time-first: `(2, bins,
H, W)` and `(bins, H, W)`. They are the two temporal representations event-vision models
commonly consume: RVT's per-polarity histogram, and the signed linear voxel grid of E2VID and
E-RAFT. Both are running kernels, constructed with keyword arguments only:

- **`bins`**, the number of output channels along time: an int, at least 1 for
  `StackedHistogram` and at least 2 for `VoxelGrid`, with no upper bound;
- **`bin_us`**, the bin width in µs: an int with `1 <= bin_us < 2**28`.

Integers of any type are accepted (`operator.index`, so NumPy integers too). A bool, a float,
a string or anything else, or a value out of range, raises `ValueError`. Neither kernel has
a name string; pass an instance.

```python title="temporal_kernels.py"
--8<-- "temporal_kernels.py"
```

```text title="Output"
--8<-- "temporal_kernels.out"
```

Every rule below is shown on small exact cases, input events to expected frame, in the
[temporal kernel semantics table](temporal-semantics.md).

### Bins on the event-time grid

Bin `k` covers `[k * bin_us, (k + 1) * bin_us)` of absolute event time, half-open, worked
out in integer arithmetic. Events with equal timestamps always share a bin. The grid doesn't
depend on the events: the first event doesn't start it.

A frame read at the [watermark](event-contract.md#the-watermark) `T` shows **only completed
bins**, the bins before `T // bin_us`. The bin `T` falls in is still being filled and is
never shown, so a frame ends at the start of that bin. In the example the watermark is 31,
so the frame ends at 30: the event at 31 isn't shown, and neither would one at exactly 30.

- **A stream that pauses** keeps its newest bin hidden until an event crosses the next bin
  edge. Publication and `stop()` don't move the watermark, so they don't show it either.
- **Late events count** if their bin is still shown: these are running kernels, and the
  result doesn't depend on arrival order. Events older than the frame count for nothing, and
  never will, because the watermark never goes back.
- **Zeros** everywhere before the first in-bounds event, in empty bins, and in bins before
  t = 0. Never NaN.
- **Out-of-bounds events** are counted by the Accumulator or Engine as for every kernel, and
  change nothing in the frame.
- **Order and batching don't matter**, exactly: the same events give the same frame, bit for
  bit, in any order and split into calls in any way.
- **`reset()`** clears every bin. An Engine's snapshot sequence carries on as usual.
- **Timestamps near 2^63:** no special case. The bin that holds `2**63 - 1` can never
  complete, so it is never shown.

No normalisation is applied. Models trained on normalised input (E2VID and E-RAFT normalise
their voxel grids; RVT clips its histogram) need that step applied to a copy of the frame.

**At a window's closing time Frames2Py differs from E2VID, E-RAFT and RVT.** Their code puts
an event at exactly the window's end on its last knot or bin. Here the bins are half-open,
so that event belongs to the next bin, the convention of DSEC's event slicer
(`t_start <= t < t_end`). With events at that time left out, the values match those
implementations' code, up to their own float32 rounding: a one-off cross-check on small random
inputs, not part of the test suite.

### StackedHistogram

`(2, bins, H, W)` uint32: events per polarity and time bin. Index `j` is grid bin
`T // bin_us - bins + j`, so `j = 0` is the oldest bin and the frame covers `bins * bin_us`
µs. Channel 0 counts OFF events (`p == 0`), channel 1 ON events (any other `p`), as in
`Polarity`.

**Counts are unclipped uint32 and wrap modulo 2^32**, like `EventCount`'s. RVT clips its
histogram at 10 and stores uint8, `(2 * bins, H, W)`. To give an RVT model its exact input,
convert a copy:

```python
import numpy as np

frame = np.zeros((2, 10, 260, 346), dtype=np.uint32)  # e.g. snapshot.frame, bins=10
rvt_input = np.minimum(frame, 10).astype(np.uint8).reshape(2 * 10, 260, 346)
print(rvt_input.shape, rvt_input.dtype)
```

```text title="Output"
(20, 260, 346) uint8
```

### VoxelGrid

`(bins, H, W)` float32: signed events, spread linearly in time over `bins` knots.

- **Knots.** Knot `j` sits at time `(T // bin_us - bins + 1 + j) * bin_us`, oldest first, so
  the last knot is at the start of the bin in progress. The frame covers the `bins - 1`
  completed bins between the first and last knots: `(bins - 1) * bin_us` µs. That is one bin
  less than a `StackedHistogram` with the same `bins` shows.
- **Weights.** An event in that span at `t = q * bin_us + r` adds `s * (bin_us - r) / bin_us`
  to the knot at `q * bin_us` and `s * r / bin_us` to the next one, with `s = +1` for ON
  (`p != 0`) and `-1` for OFF (`p == 0`). Each event contributes 1 in total, ON and OFF
  cancel exactly, and an event on a knot goes wholly to it. The edge knots are one-sided: the
  first takes only events after it, the last only events before it. This is E2VID's
  interpolation with the window's first and last times fixed at the first and last knots.
- **Exact numerators.** Each knot's value is an integer numerator `N`, the sum of its
  `s * (bin_us - r)` and `s * r` terms, divided by `bin_us`. `N` is kept exactly, modulo
  2^64, and read as a signed 64-bit integer `n`: `n == N` while `-2**63 <= N <= 2**63 - 1`,
  and it wraps beyond. Modular arithmetic is what keeps order and batching exactly
  irrelevant. Each event moves a numerator by at most `bin_us`, so wrapping takes more than
  `2**63 / bin_us` events at one pixel within two bins: about 9.2e14 at `bin_us = 10_000`.
- **Rounding.** The output is `float32(float64(n) / float64(bin_us))`, the float32 nearest
  the exact quotient whenever `|N| <= 2**53`. Beyond that it can differ from correct
  rounding. That bound is on the numerator; the `bin_us < 2**28` bound doesn't imply it.

**Normalising for a model.** E2VID scales the nonzero values of its voxel grid to mean 0
and standard deviation 1 and leaves zeros at 0 (`rpg_e2vid`, `utils/inference_utils.py`).
On a copy of the frame:

```python
import numpy as np

frame = np.zeros((3, 2, 2), dtype=np.float32)  # e.g. snapshot.frame of VoxelGrid(bins=3, ...)
frame[:, 0, 1] = [0.8, 0.2, -0.5]
x = frame.copy()                                # never normalise the shared frame in place
nonzero = x != 0
if nonzero.any():                               # E2VID: mean and std of the nonzero values only
    mean = x[nonzero].mean()
    std = np.sqrt((x[nonzero] ** 2).mean() - mean**2)
    x[nonzero] = (x[nonzero] - mean) / std
print(np.round(x[:, 0, 1], 4), x.dtype)
```

```text title="Output"
[ 1.1922  0.0627 -1.2549] float32
```

E-RAFT's version differs in two details: it uses the sample standard deviation
(`ddof=1`), and when that is 0 it only subtracts the mean. Match the convention of the model
you feed.

### Memory

Sizes in bytes, for an `H x W` sensor:

| kernel | state | each published frame |
|---|---|---|
| `StackedHistogram` | `2 * (bins + 1) * H * W * 4` | `2 * bins * H * W * 4` |
| `VoxelGrid` | `2 * bins * H * W * 8` | `bins * H * W * 4` |

At 1280x720: `StackedHistogram(bins=10, ...)` keeps 77.3 MiB of state and publishes 70.3
MiB frames; `VoxelGrid(bins=5, ...)` keeps 70.3 MiB and publishes 17.6 MiB frames. After its
first read, `VoxelGrid` also keeps one `H * W * 8`-byte working plane (7.0 MiB at 1280x720). An Engine
allocates a new frame for every publication, and the frames consumers still hold stay
alive. The state has one set of planes per bin a later read can still show; when the
watermark enters a new bin, the planes of the bin it entered are zeroed, work proportional
to `H x W` per bin entered.

`bins` has no upper limit. Frames2Py doesn't promise to allocate eagerly: depending on the
platform, a large state can be created successfully and raise `MemoryError` later, when its
memory is first used. Sizes NumPy itself refuses fail at construction.

### Throughput

These are reference measurements from **one machine**, an Apple M4 (16 GB), on CPython
3.11.14 and on CPython 3.14.2t with the GIL disabled, both with NumPy 2.4.6. They are not
guarantees for other hardware or workloads. The figures are from the second run of the
temporal kernels' gate (commit `8131aca`); its method and qualifications are
on [Benchmark methodology](../reference/methodology.md#the-temporal-kernel-gate), and every
cell is in
[`benchmarks/results/temporal_gate_run2.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/temporal_gate_run2.csv).
The Engine's producer-thread check changed after `8131aca`: `ingest()` now looks up the
calling thread's object instead of its thread ident, one lookup and one comparison per call.
The figures below were not re-measured after that change.
The gate's target is 20M events/s at kernel level and through `Engine.ingest()`, for five
parameter sets (`StackedHistogram` 5 x 10,000, 15 x 3,333 and 10 x 5,000 µs; `VoxelGrid` 5 x
12,500 and 15 x 3,571 µs), three resolutions, batches of 10k, 100k and 1M events, and
publication every call (0 ms) or every 16 ms. **The target is not met everywhere.**

- **Accumulation alone** (kernel level): at least 29M events/s in every cell, up to about
  185M.
- **Through the Engine**, 12 of the 150 cells stay below 20M events/s on each runtime (138
  pass). The gate's first run, on the code before the changes the second run measured,
  missed 13 cells on 3.11.14 and 15 on 3.14.2t, all at Engine level too. In the second run,
  every miss is at 1280x720, or at 640x480 with `VoxelGrid` 15 bins, and publishes large
  frames often:

| parameter set | sensor | batch @ interval | M events/s, 3.11.14 / 3.14.2t |
|---|---|---|---|
| `StackedHistogram` 15 x 3,333 | 1280x720 | 100k @ 0 ms | 14.5 / 14.3 (uniform), 15.3 / 15.5 (clustered) |
| `StackedHistogram` 10 x 5,000 | 1280x720 | 100k @ 0 ms, uniform | 20.5 / 20.3 over 5 runs, inside the gate's uncertainty band; in 20 more runs 16 and 12 reached 20M, short of the 18 required (it passed in the gate's first run) |
| `VoxelGrid` 5 x 12,500 | 1280x720 | 100k @ 0 ms | 17.8 / 17.6 (uniform), 19.3 / 19.2 (clustered) |
| `VoxelGrid` 15 x 3,571 | 640x480 | 100k @ 0 ms | 16.2 / 15.8 (uniform), 16.8 / 17.3 (clustered) |
| `VoxelGrid` 15 x 3,571 | 1280x720 | 10k @ 16 ms | 15.1 / 13.1 (uniform), 18.7 / 15.3 (clustered) |
| `VoxelGrid` 15 x 3,571 | 1280x720 | 100k @ 0 ms | 6.8 / 6.6 (uniform), 7.2 / 7.0 (clustered) |
| `VoxelGrid` 15 x 3,571 | 1280x720 | 100k @ 16 ms, uniform | 19.0 / 17.6 |

`VoxelGrid` with 15 bins at 1280x720 sustained 6.6-7.2M events/s when it published every
100k events, 13-19M with 10k-event calls at 16 ms, and 28-34M with 1M-event calls. Each
publication there produces a new 52.7 MiB frame (see [Memory](#memory)).

Some small-batch cells vary widely between runs: with 10k-event calls at 16 ms, one run of a
cell can be several times as fast as another (for example 29.8-105.9M events/s through
the Engine for `StackedHistogram` 15 x 3,333 at 346x260). The per-run ranges are in the data
file.

**For high event rates with large configurations:** fewer bins, a lower resolution, larger
batches, or a publication interval (frames are produced only at publication) all reduce the
work per event. `StackedHistogram` with 5 bins and `VoxelGrid` with 5 bins at 640x480 and below
met 20M events/s in every cell. For offline use, [`windows()`](../data/replay.md#frames-in-event-time)
has no rate to keep up with.

### Viewing and offline frames

[`viewer.render()`](../consumers/viewer.md) draws single frames only: a temporal frame raises
`TypeError`. To turn a recording into a frame every N µs of event time, for training or
evaluation, use [`frames2py.replay.windows()`](../data/replay.md#frames-in-event-time).

## Custom kernels

`Accumulator` and `Engine` accept any object that implements the
[`frames2py.kernels.Kernel`](../reference/api/kernels.md) protocol: `name`, `output_spec`,
`init_state`, `begin_call`, `accumulate(events, state, watermark)`,
`read(state, out, watermark)`, `close_window` and `reset`. The Accumulator keeps
validation, the range and bounds checks and the watermark; a kernel only sees a call's
in-bounds events. It is a protocol to implement, not a plugin system; the seven kernels
above are the only ones Frames2Py ships.

**The watermark argument.** `accumulate()` receives the watermark including the call's
events: the largest timestamp among accumulated in-bounds events. `read()` receives the
time at which to evaluate the representation: the latest accumulated watermark, or any later
time. Both get `None` before the first in-bounds event.

- `Accumulator` and `Engine` always pass the accumulated watermark to `read()`.
- A later time comes from callers that evaluate a kernel at a chosen time.
  [`frames2py.replay.windows()`](../data/replay.md#frames-in-event-time) reads each frame at
  its boundary, later than every event accumulated, so it needs a kernel that evaluates at
  the time it is given.
- `read()` must never change the kernel's state, whatever time it is given.
- The seven built-in kernels meet these rules.

**Changed in 1.1.** In 1.0, `read()` only ever received the accumulated watermark. A
kernel that relied on that may need adapting before it is used with a caller that reads at
a later time. For example, a kernel might ignore the argument and use a watermark it stored
in `accumulate()`.
