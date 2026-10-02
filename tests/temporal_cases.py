"""Hand-computed cases for the temporal kernels: input events to the exact expected frame.

Every expected value here was worked out by hand from the kernel definitions (see
``tests/temporal_oracle.py`` for the definitions), not produced by running code. The same
cases check the oracle and the kernels.

All cases use a 2x1 sensor (frames ``(..., 1, 2)``), ``bins = 3`` and ``bin_us = 10``
unless they say otherwise, so grid bin k is ``[10k, 10k + 10)``. A case is a sequence of
calls; ``RESET`` between calls resets. Expected values are sparse: every entry not listed
is 0.

- ``stacked_histogram``: keys ``(channel, j, x)``; channel 0 OFF, 1 ON; j = 0 is the oldest.
- ``voxel_grid``: keys ``(j, x)``, values the exact numerator N; the frame's float32 value
  is ``N / bin_us`` rounded as the kernel defines.
"""

from __future__ import annotations

import dataclasses
from typing import Final

SENSOR: Final = (2, 1)
RESET: Final = "reset"

Row = tuple[int, int, int, int]  # t, x, y, p
Call = tuple[Row, ...]


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    kernel: str
    calls: tuple[Call | str, ...]
    expected: dict[tuple[int, ...], int]
    watermark: int | None
    bins: int = 3
    bin_us: int = 10
    note: str = ""


def on(t: int, x: int = 0) -> Row:
    return (t, x, 0, 1)


def off(t: int, x: int = 0) -> Row:
    return (t, x, 0, 0)


WORKED = (on(0), on(5), on(12), on(25), on(31))

CASES: Final = (
    # --- stacked_histogram -------------------------------------------------------
    Case("histogram: worked example", "stacked_histogram", (WORKED,),
         {(1, 0, 0): 2, (1, 1, 0): 1, (1, 2, 0): 1}, 31,
         note="T=31: bins 0, 1, 2 shown, [0, 30); t=31 is in the in-progress bin [30, 40)"),
    Case("histogram: arrival order doesn't matter", "stacked_histogram", (tuple(reversed(WORKED)),),
         {(1, 0, 0): 2, (1, 1, 0): 1, (1, 2, 0): 1}, 31,
         note="the watermark is the largest t, not the last event's"),
    Case("histogram: partition into calls doesn't matter", "stacked_histogram",
         ((on(25), on(0)), (), (on(31), on(12)), (on(5),)),
         {(1, 0, 0): 2, (1, 1, 0): 1, (1, 2, 0): 1}, 31),
    Case("histogram: equal timestamps share a bin", "stacked_histogram",
         ((on(15), on(15), on(15), on(30, x=1)),),
         {(1, 1, 0): 3}, 30),
    Case("histogram: bins are half-open", "stacked_histogram",
         ((on(9), on(10), on(19), on(20), on(29), on(30)),),
         {(1, 0, 0): 1, (1, 1, 0): 2, (1, 2, 0): 2}, 30,
         note="t=10 and t=20 open their bins; t=30 opens the in-progress bin and is hidden"),
    Case("histogram: polarity is p != 0", "stacked_histogram",
         ((off(5), on(5), (5, 0, 0, 2), (5, 0, 0, 255), on(30, x=1)),),
         {(0, 0, 0): 1, (1, 0, 0): 3}, 30,
         note="no p value reaches the other pixel"),
    Case("histogram: out-of-bounds events don't move the watermark", "stacked_histogram",
         ((on(5), on(12), (1000, 2, 0, 1), (1000, 0, 1, 1)),),
         {(1, 2, 0): 1}, 12,
         note="T=12: bins -2, -1, 0 shown; bins before t=0 are empty"),
    Case("histogram: no in-bounds event", "stacked_histogram",
         ((), ((7, 5, 0, 1),)), {}, None),
    Case("histogram: empty window", "stacked_histogram",
         ((on(5), on(100)),), {}, 100,
         note="T=100 shows bins 7, 8, 9: t=5 is older, t=100 is in progress"),
    Case("histogram: the in-progress bin is hidden", "stacked_histogram",
         ((on(21), on(25)),), {}, 25,
         note="T=25 shows bins -1, 0, 1; [20, 30) is partly filled and not shown"),
    Case("histogram: a bin shows once the watermark passes it", "stacked_histogram",
         ((on(21), on(25)), (on(30, x=1),)), {(1, 2, 0): 2}, 30),
    Case("histogram: late event inside the frame counts", "stacked_histogram",
         ((on(35), on(3)),), {(1, 0, 0): 1}, 35),
    Case("histogram: late event older than the frame doesn't", "stacked_histogram",
         ((on(55), on(12)),), {}, 55),
    Case("histogram: reset", "stacked_histogram",
         ((on(5), on(15), on(35)), RESET, (on(100), on(112), on(125))),
         {(1, 1, 0): 1, (1, 2, 0): 1}, 125,
         note="after reset only t=100, 112, 125 exist; T=125 shows bins 9, 10, 11"),
    Case("histogram: bins=1", "stacked_histogram", ((on(3), on(14)),), {(1, 0, 0): 1}, 14, bins=1),
    # --- voxel_grid ----------------------------------------------------------------
    Case("voxel: worked example", "voxel_grid", (WORKED,),
         {(0, 0): 8, (1, 0): 7, (2, 0): 5}, 31,
         note="knots at 10, 20, 30; span [10, 30): t=12 gives 8, 2; t=25 gives 5, 5"),
    Case("voxel: arrival order doesn't matter", "voxel_grid", (tuple(reversed(WORKED)),),
         {(0, 0): 8, (1, 0): 7, (2, 0): 5}, 31),
    Case("voxel: partition into calls doesn't matter", "voxel_grid",
         ((on(25), on(0)), (), (on(31), on(12)), (on(5),)),
         {(0, 0): 8, (1, 0): 7, (2, 0): 5}, 31),
    Case("voxel: equal timestamps", "voxel_grid",
         ((on(12), on(12), on(12), on(30, x=1)),),
         {(0, 0): 24, (1, 0): 6}, 30),
    Case("voxel: an event on a knot gives it full weight", "voxel_grid",
         ((on(10), on(20), on(30)),),
         {(0, 0): 10, (1, 0): 10}, 30,
         note="t=30 sits on the last knot's time but is in the in-progress bin, so it is hidden"),
    Case("voxel: polarity is signed by p != 0, and cancels exactly", "voxel_grid",
         ((off(15), (15, 0, 0, 2), (12, 1, 0, 255), on(30, x=1)),),
         {(0, 1): 8, (1, 1): 2}, 30,
         note="pixel 0: -5 -5 and +5 +5 cancel to exact zeros"),
    Case("voxel: out-of-bounds events don't move the watermark", "voxel_grid",
         ((on(5), on(12), (1000, 2, 0, 1)),),
         {(1, 0): 5, (2, 0): 5}, 12,
         note="T=12: knots at -10, 0, 10, span [-10, 10); t=5 gives 5, 5"),
    Case("voxel: no in-bounds event", "voxel_grid", ((), ((7, 5, 0, 1),)), {}, None),
    Case("voxel: empty window", "voxel_grid", ((on(5), on(100)),), {}, 100),
    Case("voxel: the in-progress bin is hidden", "voxel_grid",
         ((on(21), on(25)),), {}, 25,
         note="T=25: knots at 0, 10, 20, span [0, 20)"),
    Case("voxel: late event inside the span counts", "voxel_grid",
         ((on(35), on(12)),), {(0, 0): 8, (1, 0): 2}, 35),
    Case("voxel: late event older than the span doesn't", "voxel_grid",
         ((on(35), on(3)),), {}, 35),
    Case("voxel: reset", "voxel_grid",
         ((on(5), on(15), on(35)), RESET, (on(101), on(115), on(125))),
         {(0, 0): 9, (1, 0): 6, (2, 0): 5}, 125,
         note="T=125: knots at 100, 110, 120, span [100, 120); t=101 gives 9, 1 and t=115 gives 5, 5"),
    Case("voxel: two knots, rounding", "voxel_grid",
         ((on(4), on(6)),), {(0, 0): 2, (1, 0): 1}, 6, bins=2, bin_us=3,
         note="knots at 3, 6, span [3, 6): t=4 gives 2/3 and 1/3"),
)


@dataclasses.dataclass(frozen=True)
class WindowCase:
    """``windows(batches, SENSOR, kernel, every_us=...)``: expected ``(boundary, frame)`` pairs.

    ``kernel`` is ``(name, parameters)``. Frames are sparse like ``Case.expected``: keys are
    ``(x,)`` for ``event_count``; ``(channel, j, x)`` for ``stacked_histogram``; ``(j, x)``
    with numerators for ``voxel_grid``.
    """

    name: str
    kernel: tuple[str, dict[str, int]]
    batches: tuple[Call, ...]
    every_us: int
    expected: tuple[tuple[int, dict[tuple[int, ...], int]], ...]
    note: str = ""


COUNT: Final = ("event_count", {})
WINDOW_CASES: Final = (
    WindowCase("windows: a frame per boundary, the tail dropped", COUNT,
               ((on(3), on(12)), (on(15), on(27), on(41))), 10,
               ((10, {(0,): 1}), (20, {(0,): 2}), (30, {(0,): 1}), (40, {})),
               note="t=41 passes 30 and 40; boundary 50 is never reached, so its frame isn't yielded"),
    WindowCase("windows: batches don't matter", COUNT,
               ((on(3),), (on(12), on(15)), (), (on(27),), (on(41),)), 10,
               ((10, {(0,): 1}), (20, {(0,): 2}), (30, {(0,): 1}), (40, {}))),
    WindowCase("windows: out-of-bounds events cross no boundary", COUNT,
               ((on(3), on(12), (100, 2, 0, 1), on(15)),), 10,
               ((10, {(0,): 1}),)),
    WindowCase("windows: a late event joins the open window", COUNT,
               ((on(3), on(25), on(8), on(31)),), 10,
               ((10, {(0,): 1}), (20, {}), (30, {(0,): 2})),
               note="t=8 arrives after the watermark passed 20, so it is counted in the frame at 30"),
    WindowCase("windows: the first boundary is after the first event", COUNT,
               ((on(10), on(25)),), 10,
               ((20, {(0,): 1}),),
               note="t=10 is on a boundary; the first frame is at 20"),
    WindowCase("windows: histogram read at the boundary", ("stacked_histogram", {"bins": 2, "bin_us": 10}),
               ((on(5), on(12), on(18), on(26), on(41)),), 20,
               ((20, {(1, 0, 0): 1, (1, 1, 0): 2}), (40, {(1, 0, 0): 1}))),
    WindowCase("windows: gaps get frames, each read at its own boundary",
               ("stacked_histogram", {"bins": 2, "bin_us": 10}),
               ((on(5), on(95)),), 20,
               ((20, {(1, 0, 0): 1}), (40, {}), (60, {}), (80, {})),
               note="at the accumulated watermark (5) the frame at 20 would show bins -2, -1: empty"),
    WindowCase("windows: voxel grid", ("voxel_grid", {"bins": 3, "bin_us": 10}),
               ((on(12), on(25), on(31), on(64)),), 30,
               ((30, {(0, 0): 8, (1, 0): 7, (2, 0): 5}), (60, {}))),
)
