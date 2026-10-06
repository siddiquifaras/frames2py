# Temporal kernel semantics table

Input events and the exact frame each [temporal kernel](kernels.md#temporal-kernels)
produces from them. Every expected frame below was worked out by hand from the kernels'
definitions, not by running Frames2Py. The test suite checks the same cases against an
independent per-event reference implementation and against `StackedHistogram` and
`VoxelGrid` through both the `Accumulator` and the `Engine`, and checks that this page shows
exactly those cases.

## Setup and notation

Every case uses a sensor 2 pixels wide and 1 high (`sensor_size=(2, 1)`), so frames are
`(2, bins, 1, 2)` and `(bins, 1, 2)`, and `bins=3, bin_us=10` unless its row says otherwise:
grid bin `k` is `[10k, 10k + 10)` µs.

- **Events.** `+12` is an ON event (`p = 1`) at `t = 12` µs on pixel `x = 0`; `-12` is an OFF
  event (`p = 0`). `@1` puts it on pixel `x = 1`. `@2` (`x = 2`), `@5` and `@0,1` (`y = 1`)
  are outside the sensor. `(p=2)` gives another `p`: any `p != 0` is ON.
- **Calls.** `|` separates `accumulate()` or `ingest()` calls; `()` is a call with no
  events; `reset` calls `reset()`.
- **T** is the [watermark](event-contract.md#the-watermark) after the last call: the largest
  timestamp among accumulated in-bounds events, or `None`.
- **Frame.** Only nonzero pixels are listed; everything else is 0. Values are oldest first:
  `ON x0: 2 1 1` means channel 1 (ON), pixel `x = 0`, `j = 0, 1, 2`. A `VoxelGrid` value is
  the float32 nearest the number shown (`0.8` is `np.float32(0.8)`; `2/3` is
  `np.float32(2 / 3)`).

## The closing time

**Bins are half-open and only completed bins are shown.** A frame read at watermark `T`
shows the bins before `T // bin_us`; the bin `T` falls in is still filling and is hidden.
An event at exactly a frame's closing time, `(T // bin_us) * bin_us`, is therefore in the
hidden bin, not on the frame's last bin or knot. E2VID, E-RAFT and RVT place such an event on
their final knot or bin; Frames2Py follows the half-open convention of DSEC's event slicer
(`t_start <= t < t_end`). The cases "bins are half-open", "an event on a knot gives it full
weight" and "a bin shows once the watermark passes it" show it.

The worked example, the first row of both tables, drawn on the time axis:

![The worked example on a time axis in microseconds. Grid bin 0 covers 0 up to but not including 10, bin 1 covers 10 to 20, bin 2 covers 20 to 30, bin 3 covers 30 to 40. ON events at 0, 5, 12, 25 and 31 give the watermark T = 31, so bin 3 is in progress and hidden, and the closing time is 30. StackedHistogram(bins=3, bin_us=10) shows bins 0, 1 and 2, the events from 0 up to 30: ON counts 2, 1, 1. VoxelGrid(bins=3, bin_us=10) has knots at 10, 20 and 30 and spans the events from 10 up to 30: the event at 12 gives 0.8 to knot 10 and 0.2 to knot 20, the event at 25 gives 0.5 to knots 20 and 30, so its values are 0.8, 0.7, 0.5; the events at 0 and 5 are older than the span. The event at 31, and one at exactly 30, show once the watermark reaches 40.](../assets/diagram-temporal-bins.svg)

## StackedHistogram

`StackedHistogram(bins=3, bin_us=10)`: `(2, 3, 1, 2)` uint32, channel 0 OFF and 1 ON, `j = 0`
the oldest bin. The frame covers the 3 completed bins before `T`'s.

| case | `bins, bin_us` | calls | T | frame | why |
|---|---|---|---|---|---|
| worked example | 3, 10 | `+0 +5 +12 +25 +31` | 31 | `ON x0: 2 1 1` | bins `[0, 10)`, `[10, 20)`, `[20, 30)` are shown; 31 is in the bin in progress |
| arrival order doesn't matter | 3, 10 | `+31 +25 +12 +5 +0` | 31 | `ON x0: 2 1 1` | the watermark is the largest timestamp, not the last event's |
| partition into calls doesn't matter | 3, 10 | `+25 +0 | () | +31 +12 | +5` | 31 | `ON x0: 2 1 1` | the same events split into four calls, one empty |
| equal timestamps share a bin | 3, 10 | `+15 +15 +15 +30@1` | 30 | `ON x0: 0 3 0` | all three are in `[10, 20)` |
| bins are half-open | 3, 10 | `+9 +10 +19 +20 +29 +30` | 30 | `ON x0: 1 2 2` | 10 and 20 open their bins; 30 opens the bin in progress and is hidden |
| polarity is p != 0 | 3, 10 | `-5 +5 +5(p=2) +5(p=255) +30@1` | 30 | `OFF x0: 1 0 0; ON x0: 3 0 0` | every `p != 0` is ON; no `p` value reaches the other pixel |
| out-of-bounds events don't move the watermark | 3, 10 | `+5 +12 +1000@2 +1000@0,1` | 12 | `ON x0: 0 0 1` | T = 12 shows bins -2, -1 and 0; bins before t = 0 are empty |
| no in-bounds event | 3, 10 | `() | +7@5` | None | all 0 | no watermark yet, so every value is 0 |
| empty window | 3, 10 | `+5 +100` | 100 | all 0 | T = 100 shows bins 7, 8, 9: 5 is older and 100 is in progress |
| the in-progress bin is hidden | 3, 10 | `+21 +25` | 25 | all 0 | T = 25 shows bins -1, 0, 1; `[20, 30)` is partly filled and not shown |
| a bin shows once the watermark passes it | 3, 10 | `+21 +25 | +30@1` | 30 | `ON x0: 0 0 2` | the event at 30 completes `[20, 30)` |
| late event inside the frame counts | 3, 10 | `+35 +3` | 35 | `ON x0: 1 0 0` | 3 arrives after 35 and is still in a shown bin |
| late event older than the frame doesn't | 3, 10 | `+55 +12` | 55 | all 0 | T = 55 shows bins 2, 3, 4; 12 is older |
| reset | 3, 10 | `+5 +15 +35 | reset | +100 +112 +125` | 125 | `ON x0: 0 1 1` | after the reset only 100, 112 and 125 exist; T = 125 shows bins 9, 10, 11 |
| bins=1 | 1, 10 | `+3 +14` | 14 | `ON x0: 1` | `bins=1` shows only the last completed bin |

## VoxelGrid

`VoxelGrid(bins=3, bin_us=10)`: `(3, 1, 2)` float32, knot `j = 0` the oldest. With `T` in bin
`k_T`, the knots are at `(k_T - 2) * 10`, `(k_T - 1) * 10` and `k_T * 10`, and the frame
covers the 2 completed bins between the first and last knots. An event at `t = q * 10 + r`
in that span adds `(10 - r) / 10` to the knot at `q * 10` and `r / 10` to the next, signed
`+` for ON and `-` for OFF.

| case | `bins, bin_us` | calls | T | frame | why |
|---|---|---|---|---|---|
| worked example | 3, 10 | `+0 +5 +12 +25 +31` | 31 | `x0: 0.8 0.7 0.5` | knots at 10, 20, 30; 12 gives 0.8 and 0.2, 25 gives 0.5 and 0.5; 0 and 5 are older than the span, 31 is in the bin in progress |
| arrival order doesn't matter | 3, 10 | `+31 +25 +12 +5 +0` | 31 | `x0: 0.8 0.7 0.5` | as above, in reverse order |
| partition into calls doesn't matter | 3, 10 | `+25 +0 | () | +31 +12 | +5` | 31 | `x0: 0.8 0.7 0.5` | the same events split into four calls, one empty |
| equal timestamps | 3, 10 | `+12 +12 +12 +30@1` | 30 | `x0: 2.4 0.6 0` | each event at 12 gives 0.8 and 0.2 |
| an event on a knot gives it full weight | 3, 10 | `+10 +20 +30` | 30 | `x0: 1 1 0` | 30 is on the last knot's time but in the bin in progress, so it is hidden |
| polarity is signed by p != 0, and cancels exactly | 3, 10 | `-15 +15(p=2) +12(p=255)@1 +30@1` | 30 | `x1: 0.8 0.2 0` | at `x = 0`, -0.5 -0.5 and +0.5 +0.5 cancel to exact zeros |
| out-of-bounds events don't move the watermark | 3, 10 | `+5 +12 +1000@2` | 12 | `x0: 0 0.5 0.5` | T = 12: knots at -10, 0, 10, span `[-10, 10)`; 5 gives 0.5 and 0.5 |
| no in-bounds event | 3, 10 | `() | +7@5` | None | all 0 | no watermark yet, so every value is 0 |
| empty window | 3, 10 | `+5 +100` | 100 | all 0 | knots at 80, 90, 100: 5 is older and 100 is in progress |
| the in-progress bin is hidden | 3, 10 | `+21 +25` | 25 | all 0 | T = 25: knots at 0, 10, 20, span `[0, 20)` |
| late event inside the span counts | 3, 10 | `+35 +12` | 35 | `x0: 0.8 0.2 0` | 12 arrives after 35 and is inside the span `[10, 30)` |
| late event older than the span doesn't | 3, 10 | `+35 +3` | 35 | all 0 | 3 is before the span `[10, 30)` |
| reset | 3, 10 | `+5 +15 +35 | reset | +101 +115 +125` | 125 | `x0: 0.9 0.6 0.5` | T = 125: knots at 100, 110, 120, span `[100, 120)`; 101 gives 0.9 and 0.1, 115 gives 0.5 and 0.5 |
| two knots, rounding | 2, 3 | `+4 +6` | 6 | `x0: 2/3 1/3` | knots at 3 and 6, span `[3, 6)`: 4 gives 2/3 and 1/3, each rounded to float32 |

`windows()`, which yields frames at chosen event times, has hand-computed cases of its own
in the test suite; its semantics are under
[Frames in event time](../data/replay.md#frames-in-event-time).
