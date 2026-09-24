"""Run a set of cells against a target and build the result document.

Runs are the outer loop: run 1 covers every cell, then run 2, and so on. By
default each run is a separate Python process. On the prototype Engine (Apple M4,
Python 3.11, NumPy 2.4), runs after the first in a shared process were 7-15%
slower (median over cells), enough to move a 5-run median outside run-to-run
noise. Within a run,
each cell regenerates its batches from the seed and gets a freshly prepared
target.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmarks import environment, results
from benchmarks.matrix import Cell
from benchmarks.measure import Policy, measure_memory, summarize, time_calls
from benchmarks.targets import TARGETS, Target

Progress = Callable[[str], None]
RunRecord = dict[str, Any]

_REPO = Path(__file__).resolve().parent.parent


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def measure_run(
    cells: Sequence[Cell],
    target: Target,
    policy: Policy,
    memory: bool,
    progress: Progress = _stderr,
    label: str = "",
) -> list[RunRecord]:
    """One run over *cells*, all of which *target* supports."""
    records: list[RunRecord] = []
    for index, cell in enumerate(cells, 1):
        batches = cell.workload().batches(policy.batches_needed(cell.batch_size))
        prepared = target.prepare(cell)
        call_ns, deltas = time_calls(
            prepared.call,
            batches,
            policy.warmup_calls,
            policy.timed_calls_for(cell.batch_size),
            prepared.counters,
        )
        record: RunRecord = {"call_ns": call_ns, "counters": deltas, "details": dict(prepared.details)}
        del prepared
        if memory and policy.memory_calls:
            record["memory"] = measure_memory(
                target.prepare(cell).call, batches, policy.warmup_calls, policy.memory_calls
            )
        records.append(record)
        progress(f"{label}cell {index}/{len(cells)} {cell.label}")
    return records


def _run_in_subprocess(
    cells: Sequence[Cell], target_name: str, policy: Policy, memory: bool, label: str
) -> list[RunRecord]:
    request = {
        "target": target_name,
        "cells": [cell.to_record() for cell in cells],
        "policy": {
            "warmup_calls": policy.warmup_calls,
            "timed_calls": policy.timed_calls,
            "memory_calls": policy.memory_calls,
        },
        "memory": memory,
        "label": label,
    }
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks", "worker"],
        input=json.dumps(request),
        stdout=subprocess.PIPE,
        text=True,
        cwd=_REPO,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark worker failed with exit code {completed.returncode}")
    runs: list[RunRecord] = json.loads(completed.stdout)
    return runs


def worker(request: dict[str, Any]) -> list[RunRecord]:
    """Entry point of a worker process: one run, as requested by the parent."""
    policy = Policy(runs=1, **request["policy"])
    cells = [Cell.from_record(record) for record in request["cells"]]
    target = TARGETS[request["target"]]()
    return measure_run(cells, target, policy, bool(request["memory"]), label=request["label"])


def run_suite(
    suite: str,
    cells: Sequence[Cell],
    target: Target,
    policy: Policy,
    progress: Progress = _stderr,
    *,
    process_per_run: bool = True,
) -> dict[str, Any]:
    """Measure *cells* on *target* and return a result document.

    With *process_per_run*, the target must be registered in ``TARGETS``; each
    run constructs its own instance in a fresh process.
    """
    if process_per_run and TARGETS.get(target.name) is None:
        raise ValueError(f"target {target.name!r} is not registered; can't run it in a subprocess")
    env = environment.capture()
    supported = [cell for cell in cells if target.supports(cell)]
    runs: list[list[RunRecord]] = []
    for run in range(policy.runs):
        memory = run == policy.runs - 1
        label = f"run {run + 1}/{policy.runs} "
        if process_per_run:
            runs.append(_run_in_subprocess(supported, target.name, policy, memory, label))
        else:
            runs.append(measure_run(supported, target, policy, memory, progress, label))

    per_cell = {cell: [run[i] for run in runs] for i, cell in enumerate(supported)}
    records: list[dict[str, Any]] = []
    for cell in cells:
        record: dict[str, Any] = {**cell.to_record(), "in_gate": cell.in_gate}
        if cell not in per_cell:
            record["status"] = "unsupported"
        else:
            cell_runs = per_cell[cell]
            call_ns = [run["call_ns"] for run in cell_runs]
            record.update(
                status="measured",
                workload=cell.workload().describe(),
                details=cell_runs[-1]["details"],
                call_ns=call_ns,
                counter_deltas=[run["counters"] for run in cell_runs],
                summary=summarize(cell.batch_size, call_ns),
                memory=cell_runs[-1].get("memory"),
            )
        records.append(record)

    return {
        "schema": results.SCHEMA,
        "suite": suite,
        "environment": env,
        "max_rss_bytes": environment.max_rss_bytes(children=process_per_run),
        "policy": {**policy.to_record(), "process_per_run": process_per_run},
        "target": target.describe(),
        "cells": records,
    }
