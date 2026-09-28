"""Interpretation: tables from result documents.

Gate verdicts are in ``gate``. A table's ">= 20M" column compares one level's
statistic with the requirement and is not a verdict. A prototype result says
nothing about v1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from benchmarks.environment import working_tree_clean
from benchmarks.matrix import GATE_THRESHOLD_EVENTS_PER_S, Cell


def conditions(document: dict[str, Any]) -> str:
    """One line with the conditions every number in the document depends on."""
    env = document["environment"]
    target = document["target"]
    memory = env.get("memory_bytes")
    memory_text = f"{memory / 2**30:g} GiB" if isinstance(memory, int) else "memory unknown"
    power = env.get("power") or {}
    commit = f"{(env.get('commit') or 'unknown')[:12]} ({_tree_state(env)})"
    return (
        f"{target['name']} ({target['level']} level, statistic "
        f"{target.get('statistic', 'median_call')}) on {env.get('chip') or 'unknown chip'}, "
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
        "| peak temp MiB | timed publications | checks | >= 20M |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in document["cells"]:
        cell = Cell.from_record(record)
        gate = "yes" if record["in_gate"] else "no"
        if record["status"] != "measured":
            lines.append(f"| {cell.label} | {gate} | {record['status']} | | | | | | |")
            continue
        summary = record["summary"]
        latency = summary["latency_ns"]
        low, high = summary["events_per_s_run_range"]
        memory = record.get("memory")
        peak = _mib(max(memory["peak_temporary_bytes"])) if memory else "not measured"
        meets = "yes" if summary["events_per_s"] >= GATE_THRESHOLD_EVENTS_PER_S else "no"
        published = sorted({d.get("snapshots_published") for d in record.get("counter_deltas", [])} - {None})
        checks = record.get("checks") or []
        valid = "ok" if all(c.get("valid", False) for c in checks) else "FAILED"
        lines.append(
            f"| {cell.label} | {gate} | {_mev(summary['events_per_s'])} "
            f"| {_mev(low)}–{_mev(high)} "
            f"| {_ms(latency['p50'])} / {_ms(latency['p95'])} / {_ms(latency['p99'])} "
            f"({latency['samples']}) | {peak} | {'/'.join(map(str, published)) or '-'} "
            f"| {valid if checks else '-'} | {meets} |"
        )
    return "\n".join(lines)


def labels(cells: Iterable[Cell]) -> str:
    return "\n".join(f"{'gate ' if c.in_gate else 'char '} {c.label}  seed={c.seed}" for c in cells)
