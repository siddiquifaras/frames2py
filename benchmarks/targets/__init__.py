"""What a benchmark measures: a target turns a cell into a callable over batches.

A target is either kernel level (accumulation alone) or engine level (the whole
``Engine.ingest()`` path at the cell's publication interval). The two are never
reported as each other.

``prepare()`` is called once per run and must return fresh state, so runs are
independent. A target that can't run a cell says so through ``supports()``; the
runner records the cell as unsupported rather than measuring something else.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Final, Literal, Protocol

from benchmarks.matrix import Cell
from benchmarks.measure import Call, Counters

Level = Literal["kernel", "engine"]
LEVELS: Final = ("kernel", "engine")


def _no_counters() -> Mapping[str, int]:
    return {}


@dataclasses.dataclass(frozen=True, slots=True)
class Prepared:
    """A cell ready to measure.

    Attributes:
        call: Processes one batch. Its return value is ignored.
        counters: Target counters to report as deltas across the timed calls,
            e.g. publications.
        details: Facts about this configuration worth recording, e.g. state dtype.
    """

    call: Call
    counters: Counters = _no_counters
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class Target(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def level(self) -> Level: ...

    def supports(self, cell: Cell) -> bool: ...

    def prepare(self, cell: Cell) -> Prepared: ...

    def describe(self) -> dict[str, Any]: ...


def _prototype_kernel() -> Target:
    from benchmarks.targets.prototype import PrototypeKernelTarget

    return PrototypeKernelTarget()


def _prototype_engine() -> Target:
    from benchmarks.targets.prototype import PrototypeEngineTarget

    return PrototypeEngineTarget()


TARGETS: Final[dict[str, Callable[[], Target]]] = {
    "prototype-kernel": _prototype_kernel,
    "prototype-engine": _prototype_engine,
}
