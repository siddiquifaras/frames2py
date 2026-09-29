"""What a benchmark measures: a target turns a cell into a callable over batches.

A target is either kernel level (accumulation alone) or engine level (the whole
``Engine.ingest()`` path at the cell's publication interval). The two are never
reported as each other.

``prepare()`` is called once per run with the run's batches and must return fresh
state, so runs are independent. A target that can't run a cell says so through
``supports()``; the runner records the cell as unsupported rather than measuring
something else.

A target also names the per-run statistic its level is summarised by
(``measure.STATISTICS``) and may raise a cell's timed-call count, for example so
that enough publications happen during the timed calls.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, Protocol

from benchmarks.matrix import Cell
from benchmarks.measure import Call, Counters, Hook
from benchmarks.workloads import EventArray

Level = Literal["kernel", "engine"]
LEVELS: Final = ("kernel", "engine")


def _no_counters() -> Mapping[str, int]:
    return {}


@dataclasses.dataclass(frozen=True, slots=True)
class Prepared:
    """A cell ready to measure.

    Attributes:
        call: Processes one batch. Its return value is ignored. This is the timed
            region.
        counters: Target counters to report as deltas across the timed calls,
            e.g. publications.
        details: Facts about this configuration worth recording, e.g. state dtype.
        before_call: Runs before every call, outside the timed region.
        after_call: Runs after every call, outside the timed region.
        finish: Runs once after the timed calls, outside timing. Returns what the
            target observed and checked; ``{"valid": False, ...}`` marks the run's
            result as not usable.
    """

    call: Call
    counters: Counters = _no_counters
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    before_call: Hook | None = None
    after_call: Hook | None = None
    finish: Callable[[], Mapping[str, Any]] | None = None


class Target(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def level(self) -> Level: ...

    @property
    def statistic(self) -> str: ...

    def supports(self, cell: Cell) -> bool: ...

    def timed_calls(self, cell: Cell, default: int, warmup_calls: int) -> int: ...

    def prepare(self, cell: Cell, batches: Sequence[EventArray]) -> Prepared: ...

    def describe(self) -> dict[str, Any]: ...


def _v1_kernel() -> Target:
    from benchmarks.targets.v1 import V1KernelTarget

    return V1KernelTarget()


def _v1_engine() -> Target:
    from benchmarks.targets.v1 import V1EngineTarget

    return V1EngineTarget()


TARGETS: Final[dict[str, Callable[[], Target]]] = {
    "v1-kernel": _v1_kernel,
    "v1-engine": _v1_engine,
}
