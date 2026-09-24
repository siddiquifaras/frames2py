"""Targets for the prototype that is still in ``src/``.

They measure the prototype as a comparison point for its replacement. They reach
into prototype internals on purpose and are isolated here so nothing else depends
on them. Delete them with the prototype.

The prototype has four kernels. ``timestamp_decay`` cells are unsupported.
Its kernels use their own defaults: float32 state for count kernels and
``exp_decay``, float64 for ``time_surface`` at kernel level (the Engine passes
float32), and ``decay=0.95`` for ``exp_decay``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import frames2py
from frames2py.core.engine import Engine
from frames2py.kernels.base import get_kernel

from benchmarks.matrix import PROTOTYPE_KERNELS, Cell
from benchmarks.targets import Level, Prepared
from benchmarks.workloads import EventArray


def _details(kernel: Any, state_dtype: Any) -> dict[str, Any]:
    details: dict[str, Any] = {"state_dtype": str(state_dtype)}
    if kernel.name == "exp_decay":
        details["decay"] = kernel._decay
    return details


def _describe(name: str, level: Level) -> dict[str, Any]:
    return {
        "name": name,
        "level": level,
        "implementation": "frames2py prototype",
        "frames2py_version": frames2py.__version__,
        "kernels": list(PROTOTYPE_KERNELS),
    }


class PrototypeKernelTarget:
    """``kernel.accumulate(events, state)`` of a prototype NumPy kernel."""

    name = "prototype-kernel"
    level: Level = "kernel"

    def supports(self, cell: Cell) -> bool:
        return cell.kernel in PROTOTYPE_KERNELS

    def prepare(self, cell: Cell) -> Prepared:
        kernel = get_kernel(cell.kernel)
        state = kernel.init_state(cell.sensor_size)

        def call(events: EventArray) -> None:
            kernel.accumulate(events, state)

        return Prepared(call=call, details=_details(kernel, state.buf.dtype))

    def describe(self) -> dict[str, Any]:
        return _describe(self.name, self.level)


class PrototypeEngineTarget:
    """``Engine.ingest(events)`` of the prototype Engine, default settings."""

    name = "prototype-engine"
    level: Level = "engine"

    def supports(self, cell: Cell) -> bool:
        return cell.kernel in PROTOTYPE_KERNELS

    def prepare(self, cell: Cell) -> Prepared:
        engine = Engine(cell.sensor_size, kernel=cell.kernel, snapshot_interval_ms=cell.interval_ms)

        def counters() -> Mapping[str, int]:
            stats = engine.stats
            return {
                "snapshots_published": stats.snapshots_published,
                "events_dropped": stats.events_dropped,
            }

        return Prepared(
            call=engine.ingest,
            counters=counters,
            details=_details(engine.kernel, engine._state.buf.dtype),
        )

    def describe(self) -> dict[str, Any]:
        return _describe(self.name, self.level)
