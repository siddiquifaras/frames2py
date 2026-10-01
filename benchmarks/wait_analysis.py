"""Analysis of the ``wait_for_newer`` producer-cost measurement, by the preregistered rule
(``benchmarks/wait_preregistration.md`` section 8).

    uv run python -m benchmarks.wait_analysis [CAMPAIGN_DIR]

Reads the driver's ledger and the run records, uses only VALID runs, and writes
``summary.json`` and ``summary.md`` next to the ledger. Nothing here chooses which runs count:
the ledger's last attempt per spec and pass decides, as section 9 says.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from benchmarks.wait import LABELS, REPETITIONS, RUNTIME_VERSIONS, Spec, specs

METRICS: Final = {
    "M1": {"median_ns": "lower", "p95": "lower", "p99": "lower"},
    "M2": {"events_per_s": "higher", "p50": "lower", "p95": "lower", "p99": "lower"},
}
"""Per experiment: each preregistered per-run metric and which direction is faster."""
WAITERS: Final = (1, 4, 8)


def valid_records(directory: Path) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Per spec id, the result of each pass's VALID attempt; and every attempt's outcome."""
    entries = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines() if line.strip()]
    outcomes: Counter[str] = Counter(e["outcome"] for e in entries)
    last: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        last[(Spec.from_record(entry["spec"]).id, entry["pass"])] = entry
    out: dict[str, list[dict[str, Any]]] = {}
    for (spec_id, _), entry in sorted(last.items()):
        if entry["outcome"] != "VALID":
            continue
        record = json.loads((directory / "runs" / f"{entry['run_id']}.json").read_text())
        out.setdefault(spec_id, []).append(record["result"])
    return out, outcomes


def compare(x: Sequence[float], y: Sequence[float], band: float) -> dict[str, Any]:
    """*x* against *y* by section 8: the ratio of medians and whether it is distinguishable,
    which needs the ratio outside ``[1/band, band]`` and per-run ranges that don't overlap."""
    ratio = statistics.median(x) / statistics.median(y)
    outside = not (1 / band <= ratio <= band)
    disjoint = max(x) < min(y) or max(y) < min(x)
    return {"ratio": ratio, "x_range": [min(x), max(x)], "y_range": [min(y), max(y)],
            "outside_band": outside, "ranges_disjoint": disjoint, "distinguishable": outside and disjoint}


def is_slower(direction: str, ratio: float) -> bool:
    """Whether *x* is slower than *y* when ``ratio = median(x) / median(y)``."""
    return ratio > 1 if direction == "lower" else ratio < 1


def _cells(experiment: str) -> list[tuple[str, int, int, int, float]]:
    seen = []
    for s in specs():
        key = (s.runtime, s.width, s.height, s.batch_size, s.interval_ms)
        if s.experiment == experiment and key not in seen:
            seen.append(key)
    return seen


def _spec(experiment: str, cell: tuple[str, int, int, int, float], label: str) -> Spec:
    runtime, width, height, batch, interval = cell
    return Spec(experiment, runtime, width, height, batch, interval, label)


def _cell_name(experiment: str, cell: tuple[str, int, int, int, float]) -> str:
    runtime, width, height, batch, interval = cell
    return f"{RUNTIME_VERSIONS[runtime]} {width}x{height} {batch} @ {interval:g} ms"


def analyse(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The section 8 analysis of *records* (per spec id, its valid runs' results)."""
    def values(experiment: str, cell: tuple[str, int, int, int, float], label: str, metric: str) -> list[float]:
        return [float(r[metric]) for r in records.get(_spec(experiment, cell, label).id, [])]

    summary: dict[str, Any] = {"bands": {}, "w0": [], "excluded": [], "characterisation": [], "valid_runs": {}}
    for s in specs():
        summary["valid_runs"][s.id] = len(records.get(s.id, []))
    slower: list[dict[str, Any]] = []
    faster: list[dict[str, Any]] = []
    compared = 0
    for experiment, metrics in METRICS.items():
        cells = _cells(experiment)
        for metric in metrics:
            ratios, left_out = [], []
            for cell in cells:
                b, bp = values(experiment, cell, "B", metric), values(experiment, cell, "B'", metric)
                if len(b) < REPETITIONS or len(bp) < REPETITIONS:
                    left_out.append(_cell_name(experiment, cell))
                    continue
                r = statistics.median(bp) / statistics.median(b)
                ratios.append(max(r, 1 / r))
            band = max(ratios) if ratios else None
            summary["bands"][f"{experiment} {metric}"] = {"band": band, "aa_cells": len(ratios),
                                                          "aa_cells_left_out": left_out}
        for cell in cells:
            name = _cell_name(experiment, cell)
            for metric, direction in metrics.items():
                band = summary["bands"][f"{experiment} {metric}"]["band"]
                b, f0 = values(experiment, cell, "B", metric), values(experiment, cell, "F0", metric)
                if band is None or len(b) < REPETITIONS or len(f0) < REPETITIONS:
                    summary["excluded"].append({"experiment": experiment, "cell": name, "metric": metric,
                                                "valid": {"B": len(b), "F0": len(f0)}, "band": band})
                    continue
                compared += 1
                result = {"experiment": experiment, "cell": name, "metric": metric, "band": band,
                          **compare(f0, b, band)}
                result["direction"] = ("slower" if is_slower(direction, result["ratio"]) else "faster"
                                       if result["ratio"] != 1 else "equal")
                summary["w0"].append(result)
                if result["distinguishable"]:
                    (slower if result["direction"] == "slower" else faster).append(result)
                for w in WAITERS:
                    fw = values(experiment, cell, f"F{w}", metric)
                    if len(fw) < REPETITIONS or len(f0) < REPETITIONS:
                        continue
                    row = {"experiment": experiment, "cell": name, "metric": metric, "waiters": w,
                           "ratio_to_F0": statistics.median(fw) / statistics.median(f0),
                           "range": [min(fw), max(fw)], "F0_range": [min(f0), max(f0)]}
                    if experiment == "M1" and metric == "median_ns":
                        row["per_waiter_ns"] = (statistics.median(fw) - statistics.median(f0)) / w
                    summary["characterisation"].append(row)
    expected = sum(len(METRICS[e]) * len(_cells(e)) for e in METRICS)
    if slower:
        verdict = "NOT MET"
    elif compared < expected:
        verdict = "NOT ESTABLISHED"
    else:
        verdict = "MET"
    summary.update(verdict=verdict, compared=compared, expected=expected, distinguishably_slower=slower,
                   distinguishably_faster=faster)
    summary["diagnostics"] = diagnostics(records)
    return summary


def diagnostics(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per spec, over the timed region of each valid run: cycles per CPU ns and the P-core share
    of CPU time, from the process's ``proc_pid_rusage``. For investigating a speed-up (section 8),
    not part of the rule."""
    out: dict[str, Any] = {}
    for spec_id, results in records.items():
        rates, shares = [], []
        for r in results:
            a, b = r.get("rusage_before"), r.get("rusage_after")
            if not a or not b:
                continue
            cpu = (b["user_time"] - a["user_time"]) + (b["system_time"] - a["system_time"])
            pcpu = (b["user_ptime"] - a["user_ptime"]) + (b["system_ptime"] - a["system_ptime"])
            if cpu > 0:
                rates.append((b["cycles"] - a["cycles"]) / cpu)
                shares.append(pcpu / cpu)
        if rates:
            out[spec_id] = {"cycles_per_cpu_ns": [min(rates), statistics.median(rates), max(rates)],
                            "p_core_share": [min(shares), statistics.median(shares), max(shares)]}
    return out


def markdown(summary: dict[str, Any], outcomes: Counter[str]) -> str:
    lines = ["# wait_for_newer producer cost: analysis", "",
             f"Verdict on the W = 0 criterion: **{summary['verdict']}** "
             f"({summary['compared']} of {summary['expected']} preregistered cell-metric comparisons made).", "",
             f"Attempts by outcome: {dict(outcomes)}", "", "## A/A bands", "",
             "| experiment metric | band | A/A cells | left out |", "|---|---|---|---|"]
    for key, b in summary["bands"].items():
        band = "n/a" if b["band"] is None else f"{b['band']:.4f}"
        lines.append(f"| {key} | {band} | {b['aa_cells']} | {', '.join(b['aa_cells_left_out']) or '-'} |")
    lines += ["", "## W = 0: F0 against B", "",
              "| experiment | cell | metric | F0/B | band | outside band | ranges disjoint | distinguishable |",
              "|---|---|---|---|---|---|---|---|"]
    for r in summary["w0"]:
        lines.append(f"| {r['experiment']} | {r['cell']} | {r['metric']} | {r['ratio']:.4f} | {r['band']:.4f} | "
                     f"{r['outside_band']} | {r['ranges_disjoint']} | "
                     f"{r['direction'] if r['distinguishable'] else 'no'} |")
    if summary["excluded"]:
        lines += ["", "Excluded (fewer than 5 valid runs or no band):", ""]
        lines += [f"- {e['experiment']} {e['cell']} {e['metric']}: valid {e['valid']}" for e in summary["excluded"]]
    lines += ["", "## W >= 1: characterisation (F_W against F0; described, not tested)", "",
              "| experiment | cell | metric | W | F_W/F0 | per waiter (ns) |", "|---|---|---|---|---|---|"]
    for r in summary["characterisation"]:
        per = f"{r['per_waiter_ns']:.0f}" if "per_waiter_ns" in r else ""
        lines.append(f"| {r['experiment']} | {r['cell']} | {r['metric']} | {r['waiters']} | "
                     f"{r['ratio_to_F0']:.4f} | {per} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from benchmarks.wait_driver import CAMPAIGN_DIR

    args = list(sys.argv[1:] if argv is None else argv)
    directory = Path(args[0]) if args else CAMPAIGN_DIR
    records, outcomes = valid_records(directory)
    summary = analyse(records)
    summary["labels"] = {k: list(v) for k, v in LABELS.items()}
    (directory / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (directory / "summary.md").write_text(markdown(summary, outcomes))
    print(f"{summary['verdict']}: {directory / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
