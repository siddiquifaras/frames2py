"""Interpretation: tables from result documents, and the threshold comparison.

The gate verdict for a cell needs both a kernel-level and an engine-level result
for it, from the reference machine. Anything less is reported as incomplete, not
as a pass. The verdict applies to whatever target was measured; a prototype
result says nothing about v1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from benchmarks.environment import working_tree_clean
from benchmarks.matrix import GATE_THRESHOLD_EVENTS_PER_S, REFERENCE_MACHINE, Cell, gate_cells


def is_reference_machine(env: dict[str, Any]) -> bool:
    return env.get("chip") == REFERENCE_MACHINE["chip"] and (
        env.get("memory_bytes") == REFERENCE_MACHINE["memory_bytes"]
    )


def conditions(document: dict[str, Any]) -> str:
    """One line with the conditions every number in the document depends on."""
    env = document["environment"]
    target = document["target"]
    memory = env.get("memory_bytes")
    memory_text = f"{memory / 2**30:g} GiB" if isinstance(memory, int) else "memory unknown"
    power = env.get("power") or {}
    commit = f"{(env.get('commit') or 'unknown')[:12]} ({_tree_state(env)})"
    return (
        f"{target['name']} ({target['level']} level) on {env.get('chip') or 'unknown chip'}, "
        f"{memory_text}, {env.get('os')}, Python {env.get('python')}, NumPy {env.get('numpy')}, "
        f"power {power.get('source') or 'unknown'}, "
        f"low power mode {power.get('low_power_mode')}, commit {commit}, "
        f"{document['policy']['runs']} run(s)"
    )


def _tree_state(env: dict[str, Any]) -> str:
    clean = working_tree_clean(env)
    if clean:
        return "clean working tree"
    parts = ["not reproducible from the commit alone" if clean is False else "working tree state unknown"]
    tracked = env.get("tracked_changes")
    if tracked is None:
        parts.append("tracked changes unknown")
    elif tracked:
        parts.append("tracked changes")
    untracked = env.get("untracked_files")
    if untracked is None:
        parts.append("untracked files unknown")
    elif untracked:
        parts.append(f"{len(untracked)} untracked file(s)")
    if env.get("commit") is None:
        parts.append("commit unknown")
    return "; ".join(parts)


def _mev(value: float) -> str:
    return f"{value / 1e6:.1f}"


def _ms(value: float) -> str:
    return f"{value / 1e6:.3f}"


def _mib(value: float) -> str:
    return f"{value / 2**20:.1f}"


def table(document: dict[str, Any]) -> str:
    """A Markdown table of every cell in the document."""
    lines = [
        f"Conditions: {conditions(document)}",
        "",
        "| cell | in gate | M events/s | run range | p50 / p95 / p99 ms (samples) "
        "| peak temp MiB | >= 20M |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in document["cells"]:
        cell = Cell.from_record(record)
        gate = "yes" if record["in_gate"] else "no"
        if record["status"] != "measured":
            lines.append(f"| {cell.label} | {gate} | {record['status']} | | | | |")
            continue
        summary = record["summary"]
        latency = summary["latency_ns"]
        low, high = summary["events_per_s_run_range"]
        memory = record.get("memory")
        peak = _mib(max(memory["peak_temporary_bytes"])) if memory else "not measured"
        meets = "yes" if summary["events_per_s"] >= GATE_THRESHOLD_EVENTS_PER_S else "no"
        lines.append(
            f"| {cell.label} | {gate} | {_mev(summary['events_per_s'])} "
            f"| {_mev(low)}–{_mev(high)} "
            f"| {_ms(latency['p50'])} / {_ms(latency['p95'])} / {_ms(latency['p99'])} "
            f"({latency['samples']}) | {peak} | {meets} |"
        )
    return "\n".join(lines)


def _measured(document: dict[str, Any]) -> dict[Any, float]:
    return {
        Cell.from_record(record).condition: record["summary"]["events_per_s"]
        for record in document["cells"]
        if record["status"] == "measured"
    }


def gate_verdicts(
    kernel_level: dict[str, Any], engine_level: dict[str, Any]
) -> list[tuple[Cell, str, float | None, float | None]]:
    """Per gate cell: ``pass``, ``fail`` or ``incomplete``, with both throughputs.

    A cell passes only if both levels reach the threshold. It is incomplete if
    either level is missing. The machine check is separate: see ``gate_summary``.
    """
    if kernel_level["target"]["level"] != "kernel":
        raise ValueError("kernel_level document is not a kernel-level result")
    if engine_level["target"]["level"] != "engine":
        raise ValueError("engine_level document is not an engine-level result")
    kernel = _measured(kernel_level)
    engine = _measured(engine_level)
    verdicts: list[tuple[Cell, str, float | None, float | None]] = []
    for cell in gate_cells():
        k = kernel.get(cell.condition)
        e = engine.get(cell.condition)
        if k is None or e is None:
            verdict = "incomplete"
        elif k >= GATE_THRESHOLD_EVENTS_PER_S and e >= GATE_THRESHOLD_EVENTS_PER_S:
            verdict = "pass"
        else:
            verdict = "fail"
        verdicts.append((cell, verdict, k, e))
    return verdicts


def gate_summary(kernel_level: dict[str, Any], engine_level: dict[str, Any]) -> str:
    verdicts = gate_verdicts(kernel_level, engine_level)
    counts = {name: sum(1 for _, v, _, _ in verdicts if v == name) for name in ("pass", "fail", "incomplete")}
    reference = all(
        is_reference_machine(doc["environment"]) for doc in (kernel_level, engine_level)
    )
    lines = [
        f"Kernel level: {conditions(kernel_level)}",
        f"Engine level: {conditions(engine_level)}",
        "",
        f"{counts['pass']} pass, {counts['fail']} fail, {counts['incomplete']} incomplete "
        f"of {len(verdicts)} gate cells.",
    ]
    if not reference:
        lines.append(
            "Not measured on the reference machine (Apple M4, 16 GiB): "
            "these results don't count toward the gate."
        )
    lines += ["", "| cell | kernel M events/s | engine M events/s | verdict |", "|---|---|---|---|"]
    lines += [
        f"| {cell.label} | {_optional(k)} | {_optional(e)} | {verdict} |"
        for cell, verdict, k, e in verdicts
    ]
    return "\n".join(lines)


def _optional(value: float | None) -> str:
    return "missing" if value is None else _mev(value)


def labels(cells: Iterable[Cell]) -> str:
    return "\n".join(f"{'gate ' if c.in_gate else 'char '} {c.label}  seed={c.seed}" for c in cells)
