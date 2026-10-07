"""Gate classification: the 20M events/s requirement, with its uncertainty band.

Every figure is recomputed from the raw per-call times in the result documents,
never taken from their summaries.

Per level (kernel, engine) and cell:

- Stage 1, from a ``PRIMARY_RUNS``-run document: the statistic is the median of
  the per-run throughputs (the level's own per-run statistic, ``measure``).
  - ``clear_pass``: statistic >= ``CLEAR_PASS_AT``
  - ``clear_miss``: statistic < ``CLEAR_MISS_BELOW``
  - ``borderline``: anything between
- Stage 2, only for borderline results, from a ``STAGE2_RUNS``-run document with
  the identical code, runtime and method:
  - ``stage2_pass``: the median of the runs >= the requirement *and* at least
    ``STAGE2_REQUIRED_AT_REQUIREMENT`` runs >= the requirement
  - ``not_met``: otherwise
- ``invalid``: a run's result checks failed, a run was not on the document's
  runtime, the runs did not see identical input, the run count is wrong, or the
  document's power record shows the machine slept during the run or ended outside
  full wake (documents from before power records existed have none and are not
  affected).

The band is uncertainty around the requirement, not a different requirement.

A cell passes only if both levels pass. A runtime meets the gate only if every
gate cell passes on it.

Documents of the ``temporal-gate`` suite are classified against the temporal-kernel gate's
cells (``benchmarks/temporal_gate_preregistration.md``), with its further rules: a cell is
also ``invalid`` if its document or any of its runs was not on AC power, or had Low Power
Mode on, at its start or end; those problems are marked as environment problems, the only
ones that may be re-run. Each level's document must end, as it started, on a clean tree at
the same commit, and every run of both levels must record the same ``src/`` tree hash, or
the documents are not classified. Documents of any other suite are classified
against the v1 gate, as before.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any, Final

from benchmarks import power
from benchmarks.matrix import GATE_THRESHOLD_EVENTS_PER_S, REFERENCE_MACHINE, Cell, gate_cells, temporal_gate_cells
from benchmarks.measure import run_throughput

REQUIREMENT: Final = GATE_THRESHOLD_EVENTS_PER_S
CLEAR_PASS_AT: Final = 22_000_000
CLEAR_MISS_BELOW: Final = 18_000_000
PRIMARY_RUNS: Final = 5
STAGE2_RUNS: Final = 20
STAGE2_REQUIRED_AT_REQUIREMENT: Final = 18

RUNTIME_KEYS: Final = ("python", "free_threaded_build", "gil_enabled", "numpy")
TEMPORAL_SUITE: Final = "temporal-gate"
AC_POWER: Final = "AC Power"
ENVIRONMENT_PROBLEMS: Final = ("environment:", "the machine slept", "the run ended outside full wake")
"""Prefixes of the problems that come from the environment controls rather than the results."""


def cells_for(document: Mapping[str, Any]) -> tuple[Cell, ...]:
    """The gate cells *document* is classified against, by its suite."""
    return temporal_gate_cells() if document.get("suite") == TEMPORAL_SUITE else gate_cells()


def _power_problems(where: str, record: Mapping[str, Any] | None) -> list[str]:
    record = record or {}
    found = []
    if record.get("source") != AC_POWER:
        found.append(f"environment: {where} not on AC power (source {record.get('source')!r})")
    if record.get("low_power_mode") is not False:
        found.append(f"environment: {where} Low Power Mode {record.get('low_power_mode')!r}")
    return found


def environment_problems(record: Mapping[str, Any], document: Mapping[str, Any]) -> list[str]:
    """The temporal gate's power rules (its preregistration, sections 3 and 10)."""
    found = _power_problems("document start", (document.get("environment") or {}).get("power"))
    found += _power_problems("document end", (document.get("environment_end") or {}).get("power"))
    for i, checks in enumerate(record.get("checks") or []):
        power_record = (checks or {}).get("power") or {}
        for moment in ("start", "end"):
            found += _power_problems(f"run {i + 1} {moment}", power_record.get(moment))
    return found

Condition = tuple[str, tuple[int, int], int, float, str]


def runtime_of(document: Mapping[str, Any]) -> dict[str, Any]:
    env = document["environment"]
    return {key: env.get(key) for key in RUNTIME_KEYS}


def is_reference_machine(env: Mapping[str, Any]) -> bool:
    return env.get("chip") == REFERENCE_MACHINE["chip"] and (
        env.get("memory_bytes") == REFERENCE_MACHINE["memory_bytes"]
    )


def per_run_values(record: Mapping[str, Any], statistic: str) -> list[float]:
    """Each run's throughput, recomputed from its raw call times."""
    return [run_throughput(int(record["batch_size"]), run, statistic) for run in record["call_ns"]]


def problems(record: Mapping[str, Any], document: Mapping[str, Any], runs: int) -> list[str]:
    """Why a measured cell record can't be classified; empty if it can."""
    found: list[str] = []
    call_ns = record.get("call_ns") or []
    if len(call_ns) != runs:
        found.append(f"{len(call_ns)} runs, expected {runs}")
    for i, checks in enumerate(record.get("checks") or [None] * len(call_ns)):
        if not checks or not checks.get("valid", False):
            found.append(f"run {i + 1} failed its checks: {(checks or {}).get('failures')}")
    expected = runtime_of(document)
    for i, runtime in enumerate(record.get("runtime") or [None] * len(call_ns)):
        seen = {key: (runtime or {}).get(key) for key in RUNTIME_KEYS}
        if seen != expected:
            found.append(f"run {i + 1} ran on {seen}, the document on {expected}")
    digests = record.get("workload_sha256") or []
    if len(set(digests)) != 1 or len(digests) != len(call_ns):
        found.append("runs did not see identical input")
    if power.slept_during(document.get("power")):
        found.append(f"the machine slept for {document['power']['slept_ns'] / 1e9:.1f} s during the run")
    if power.ended_outside_full_wake(document.get("power")):
        caps = document["power"]["end"].get("system_capabilities")
        found.append(f"the run ended outside full wake (system capabilities {caps!r})")
    if document.get("suite") == TEMPORAL_SUITE:
        found += environment_problems(record, document)
    return found


def stage1(value: float) -> str:
    if value >= CLEAR_PASS_AT:
        return "clear_pass"
    if value < CLEAR_MISS_BELOW:
        return "clear_miss"
    return "borderline"


def stage2(values: Iterable[float]) -> str:
    values = list(values)
    at_requirement = sum(v >= REQUIREMENT for v in values)
    if statistics.median(values) >= REQUIREMENT and at_requirement >= STAGE2_REQUIRED_AT_REQUIREMENT:
        return "stage2_pass"
    return "not_met"


def _records(document: Mapping[str, Any]) -> dict[Condition, Mapping[str, Any]]:
    return {
        Cell.from_record(record).condition: record
        for record in document["cells"]
        if record.get("status") == "measured"
    }


def _same_code_and_runtime(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    found = []
    if runtime_of(first) != runtime_of(second):
        found.append(f"runtimes differ: {runtime_of(first)} vs {runtime_of(second)}")
    for doc in (first, second):
        env = doc["environment"]
        if env.get("tracked_changes") is not False or env.get("untracked_files") != []:
            found.append(f"{doc['target']['name']} was not measured from a clean working tree")
    if first["environment"].get("commit") != second["environment"].get("commit"):
        found.append("commits differ")
    if TEMPORAL_SUITE in (first.get("suite"), second.get("suite")):
        if first.get("suite") != second.get("suite"):
            found.append("suites differ")
        for doc in (first, second):
            end = doc.get("environment_end") or {}
            if end.get("tracked_changes") is not False or end.get("untracked_files") != []:
                found.append(f"{doc['target']['name']} did not end on a clean working tree")
            if end.get("commit") != doc["environment"].get("commit"):
                found.append(f"{doc['target']['name']} ended on a different commit")
    return found


def src_trees(*documents: Mapping[str, Any] | None) -> set[str | None]:
    """Every ``src/`` tree hash the runs of *documents* recorded."""
    found: set[str | None] = set()
    for document in documents:
        for record in (document or {}).get("cells", []):
            for runtime in record.get("runtime") or []:
                found.add((runtime or {}).get("src_tree"))
    return found


def level_results(
    primary: Mapping[str, Any], second_stage: Mapping[str, Any] | None = None
) -> dict[Condition, dict[str, Any]]:
    """Stage 1 and, where it applies, stage 2 for every gate cell at one level."""
    statistic = primary["target"]["statistic"]
    if primary["policy"]["runs"] != PRIMARY_RUNS:
        raise ValueError(f"a primary document has {PRIMARY_RUNS} runs, this one {primary['policy']['runs']}")
    if second_stage is not None:
        if second_stage["policy"]["runs"] != STAGE2_RUNS:
            raise ValueError(f"a stage-2 document has {STAGE2_RUNS} runs")
        if second_stage["target"] != primary["target"]:
            raise ValueError("stage 2 measured a different target")
        mismatch = _same_code_and_runtime(primary, second_stage)
        if mismatch:
            raise ValueError("stage 2 is not comparable with stage 1: " + "; ".join(mismatch))
        if {k: v for k, v in second_stage["policy"].items() if k != "runs"} != {
            k: v for k, v in primary["policy"].items() if k != "runs"
        }:
            raise ValueError("stage 2 used a different policy")
    first = _records(primary)
    second = _records(second_stage) if second_stage is not None else {}
    results: dict[Condition, dict[str, Any]] = {}
    for cell in cells_for(primary):
        record = first.get(cell.condition)
        entry: dict[str, Any] = {"stage1": None, "stage2": None, "result": "missing"}
        if record is None:
            results[cell.condition] = entry
            continue
        found = problems(record, primary, PRIMARY_RUNS)
        if found:
            entry.update(stage1="invalid", result="invalid", problems=found,
                         environment_only=all(p.startswith(ENVIRONMENT_PROBLEMS) for p in found))
            results[cell.condition] = entry
            continue
        values = per_run_values(record, statistic)
        entry.update(stage1=stage1(statistics.median(values)), stage1_statistic=statistics.median(values),
                     stage1_runs=values)
        if entry["stage1"] == "clear_pass":
            entry["result"] = "pass"
        elif entry["stage1"] == "clear_miss":
            entry["result"] = "miss"
        else:
            again = second.get(cell.condition)
            if again is None:
                entry["result"] = "pending"
            else:
                found = problems(again, second_stage or {}, STAGE2_RUNS)
                if found:
                    entry.update(stage2="invalid", result="invalid", problems=found,
                                 environment_only=all(p.startswith(ENVIRONMENT_PROBLEMS) for p in found))
                else:
                    values = per_run_values(again, statistic)
                    entry.update(stage2=stage2(values), stage2_statistic=statistics.median(values),
                                 stage2_runs=values,
                                 stage2_at_requirement=sum(v >= REQUIREMENT for v in values))
                    entry["result"] = "pass" if entry["stage2"] == "stage2_pass" else "miss"
        results[cell.condition] = entry
    unexpected = set(second) - {c for c, e in results.items() if e["stage1"] == "borderline"}
    if unexpected:
        raise ValueError(f"stage 2 measured {len(unexpected)} cells that were not borderline in stage 1")
    return results


def borderline_cells(primary: Mapping[str, Any]) -> list[Cell]:
    """The cells a stage-2 run of this level must measure, in gate order."""
    results = level_results(primary)
    return [cell for cell in cells_for(primary) if results[cell.condition]["stage1"] == "borderline"]


def cell_verdict(kernel: str, engine: str) -> str:
    if kernel == "pass" and engine == "pass":
        return "PASS"
    if "miss" in (kernel, engine):
        return "NOT MET"
    if "invalid" in (kernel, engine):
        return "INVALID"
    return "INCOMPLETE"


def verdicts(
    kernel_level: Mapping[str, Any],
    engine_level: Mapping[str, Any],
    kernel_stage2: Mapping[str, Any] | None = None,
    engine_stage2: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-cell verdicts and the runtime's gate result, from both levels.

    Each entry of ``cells`` is the cell's own record (``Cell.to_record()``) plus its
    ``label``, the per-level results under ``kernel_level`` and ``engine_level``, and
    its ``verdict``.
    """
    if kernel_level["target"]["level"] != "kernel" or engine_level["target"]["level"] != "engine":
        raise ValueError("pass the kernel-level document first and the engine-level document second")
    mismatch = _same_code_and_runtime(kernel_level, engine_level)
    if mismatch:
        raise ValueError("the two levels are not comparable: " + "; ".join(mismatch))
    trees = None
    if kernel_level.get("suite") == TEMPORAL_SUITE:
        trees = src_trees(kernel_level, engine_level, kernel_stage2, engine_stage2)
        if len(trees) != 1 or None in trees:
            raise ValueError(f"the runs recorded src/ tree hashes {sorted(map(str, trees))}; they must agree")
    kernel = level_results(kernel_level, kernel_stage2)
    engine = level_results(engine_level, engine_stage2)
    cells = []
    for cell in cells_for(kernel_level):
        k, e = kernel[cell.condition], engine[cell.condition]
        cells.append({**cell.to_record(), "label": cell.label, "kernel_level": k, "engine_level": e,
                      "verdict": cell_verdict(k["result"], e["result"])})
    counts = {name: sum(c["verdict"] == name for c in cells) for name in ("PASS", "NOT MET", "INVALID", "INCOMPLETE")}
    reference = is_reference_machine(kernel_level["environment"]) and is_reference_machine(engine_level["environment"])
    if not reference:
        gate = "NOT ON THE REFERENCE MACHINE"
    elif counts["PASS"] == len(cells):
        gate = "MET"
    elif counts["NOT MET"]:
        gate = "NOT MET"
    else:
        gate = "INCONCLUSIVE"
    return {
        "suite": kernel_level.get("suite"),
        "runtime": runtime_of(kernel_level),
        "commit": kernel_level["environment"].get("commit"),
        "src_tree": next(iter(trees)) if trees else None,
        "reference_machine": reference,
        "rules": {
            "requirement": REQUIREMENT,
            "clear_pass_at": CLEAR_PASS_AT,
            "clear_miss_below": CLEAR_MISS_BELOW,
            "primary_runs": PRIMARY_RUNS,
            "stage2_runs": STAGE2_RUNS,
            "stage2_required_at_requirement": STAGE2_REQUIRED_AT_REQUIREMENT,
            "kernel_statistic": kernel_level["target"]["statistic"],
            "engine_statistic": engine_level["target"]["statistic"],
        },
        "stage2_documents": {"kernel": kernel_stage2 is not None, "engine": engine_stage2 is not None},
        "counts": counts,
        "level_counts": {
            level: {name: sum((r["stage1"] or "missing") == name for r in results.values())
                    for name in ("clear_pass", "clear_miss", "borderline", "invalid", "missing")}
            for level, results in (("kernel", kernel), ("engine", engine))
        },
        "gate": gate,
        "cells": cells,
    }


def _mev(value: float | None) -> str:
    return "-" if value is None else f"{value / 1e6:.1f}"


def summary(result: Mapping[str, Any]) -> str:
    """Markdown: the counts, then every cell."""
    runtime = result["runtime"]
    lines = [
        f"Runtime: Python {runtime['python']}, free-threaded build {runtime['free_threaded_build']}, "
        f"GIL enabled {runtime['gil_enabled']}, NumPy {runtime['numpy']}; commit {result['commit']}",
        f"Gate: {result['gate']}. " + ", ".join(f"{v} {k}" for k, v in result["counts"].items())
        + f" of {len(result['cells'])} cells.",
        "Stage 1 per level: " + "; ".join(
            f"{level}: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v)
            for level, counts in result["level_counts"].items()
        ),
        "",
        "| cell | kernel M ev/s | kernel | engine M ev/s | engine | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for cell in result["cells"]:
        k, e = cell["kernel_level"], cell["engine_level"]
        def stage(entry: Mapping[str, Any]) -> str:
            text = str(entry["stage1"])
            if entry.get("stage2"):
                text += f" -> {entry['stage2']} ({_mev(entry.get('stage2_statistic'))}, "
                text += f"{entry.get('stage2_at_requirement')}/{STAGE2_RUNS} >= 20M)"
            return text
        lines.append(
            f"| {cell['label']} | {_mev(k.get('stage1_statistic'))} | {stage(k)} "
            f"| {_mev(e.get('stage1_statistic'))} | {stage(e)} | {cell['verdict']} |"
        )
    return "\n".join(lines)
