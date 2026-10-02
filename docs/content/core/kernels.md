# Kernels

A kernel defines what the events accumulate into. Frames2Py ships five. Their output shapes
and dtypes are part of the public API; how each stores its state internally is its own
business and may differ.

| kernel | construct with | output | mode | each in-bounds event |
|---|---|---|---|---|
| [`EventCount`](#eventcount) | `"event_count"` or `EventCount()` | `(H, W)` uint32 | windowed | adds 1 at its pixel |
| [`Polarity`](#polarity) | `"polarity"` or `Polarity()` | `(H, W, 2)` uint32 | windowed | adds 1 at its pixel, channel 0 (OFF) or 1 (ON) |
| [`TimeSurface`](#timesurface) | `"time_surface"` or `TimeSurface()` | `(H, W)` uint64 | running | keeps the largest `t` at its pixel |
| [`ExpDecay`](#expdecay) | `ExpDecay(decay)` | `(H, W)` float32 | running | adds 1; the surface decays once per call |
| [`TimestampDecay`](#timestampdecay) | `TimestampDecay(tau_us)` | `(H, W)` float32 | running | adds a weight that decays with event time |

`H, W` are the sensor's height and width. The classes are importable from `frames2py` and
from `frames2py.kernels`; the three names work wherever a kernel is accepted. Pass an
instance, `frames2py.Polarity()`, not the class `frames2py.Polarity`: a class is not checked
as such, and constructing the Engine or Accumulator with it raises an unrelated-looking
`TypeError` (`output_spec() missing 1 required positional argument`).

**Windowed** kernels start a new window at each Engine publication: a snapshot shows only
the events since the previous one. **Running** kernels are not changed by publication: a
snapshot shows everything accumulated since construction or `reset()`. With an
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
  into or reading this kernel: an event about 800 `tau_us` behind the reference raises
  `FloatingPointError` inside the call. Other kernels are not checked under such settings
  either.

## Custom kernels

`Accumulator` and `Engine` accept any object that implements the
[`frames2py.kernels.Kernel`](../reference/api/kernels.md) protocol: `name`, `output_spec`,
`init_state`, `begin_call`, `accumulate(events, state, watermark)`,
`read(state, out, watermark)`, `close_window` and `reset`. The Accumulator keeps
validation, the range and bounds checks and the watermark; a kernel only sees a call's
in-bounds events. It is a protocol to implement, not a plugin system; the five kernels above
are the only ones Frames2Py ships.

**The watermark argument.** `accumulate()` receives the watermark including the call's
events: the largest timestamp among accumulated in-bounds events. `read()` receives the
time at which to evaluate the representation: the latest accumulated watermark, or any later
time. Both get `None` before the first in-bounds event.
- `Accumulator` and `Engine` always pass the accumulated watermark to `read()`.
- A later time comes from callers that evaluate a kernel at a chosen time. The offline
  windows helper planned for 1.1, not available yet, will read at frame boundaries, so it
  will need this.
- `read()` must never change the kernel's state, whatever time it is given.
- The five built-in kernels meet these rules.

**Changed in 1.1.** In 1.0, `read()` only ever received the accumulated watermark. A
kernel that relied on that may need adapting before it is used with a caller that reads at
a later time. For example, a kernel might ignore the argument and use a watermark it stored
in `accumulate()`, or update its state inside `read()`.
