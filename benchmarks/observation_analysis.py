"""Analysis of the observation study (preregistration sections 10, 15 to 18).

Everything here reads the raw run files the harness wrote (``<run id>.json`` and ``.npz``)
and the driver's ledger. Nothing is measured here.

Readings the preregistration leaves to the analysis, fixed here before the campaign:

- **Censored lag.** A batch scheduled in the window but never released has no ``E_k``. Its
  lag counts as infinite in the medians of 15.2 (it is larger than any released batch's), and
  it is left out of the least-squares slope, which is fitted to released batches.
- **Cell labels.** A cell is SUSTAINED if at least 4 of its 5 valid runs are, NOT_SUSTAINED
  if at least 4 are not, and ``mixed`` otherwise (the threshold T1 uses).
- **Quantifiers.** Where a judgement rule names no quantifier, a support condition must hold
  in every cell it ranges over and a contradiction condition in any one.
- **Rates over W.** Rates from the monitor's cumulative counters divide the change between
  the samples nearest T0 and T1 by the time between those samples.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from benchmarks.measure import nearest_rank
from benchmarks.observation import (
    P1_ARMS,
    P2_ARMS,
    POLL_ARMS,
    QUEUE_ARMS,
    REPETITIONS,
    Condition,
    resolver,
)

MIB: Final = 2**20
PERCENTILES: Final = (50, 95, 99)
GROWTH_LIMIT_MIB_S: Final = 0.5
DEPTH_SLOPE_LIMIT: Final = 0.5
FRESHNESS_THRESHOLDS_MS: Final = (32.0, 50.0, 100.0)
BAND_METRICS: Final = ("busy_ns_per_event", "step_p99_us", "freshness_p50_ms", "freshness_p95_ms",
                       "peak_footprint_mib", "process_cpu_cores")
VALID: Final = "VALID"

Arrays = dict[str, NDArray[Any]]


# ---------------------------------------------------------------- helpers


def percentiles(values: Sequence[int] | NDArray[Any], scale: float = 1.0) -> dict[str, float | int | None]:
    """Nearest-rank p50, p95, p99 and max of integer samples, divided by *scale*, with n."""
    data = [int(v) for v in values]
    if not data:
        return {"p50": None, "p95": None, "p99": None, "max": None, "n": 0}
    out: dict[str, float | int | None] = {f"p{q}": nearest_rank(data, q) / scale for q in PERCENTILES}
    out["max"] = max(data) / scale
    out["n"] = len(data)
    return out


def slope(x: Sequence[float] | NDArray[Any], y: Sequence[float] | NDArray[Any]) -> float | None:
    """The least-squares slope of *y* against *x*, or ``None`` with fewer than two points."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(xa) < 2 or float(np.ptp(xa)) == 0.0:
        return None
    xm = xa - xa.mean()
    return float((xm * (ya - ya.mean())).sum() / (xm * xm).sum())


def ranks(values: NDArray[Any]) -> NDArray[np.float64]:
    """Ranks from 1, ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=np.float64)
    ranked[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranked[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return ranked


def spearman(x: NDArray[Any], y: NDArray[Any]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def rank_value(values: Sequence[float], percentile: float) -> float:
    """``nearest_rank`` for values that may include ``inf``."""
    ordered = sorted(values)
    return ordered[max(math.ceil(percentile / 100 * len(ordered)), 1) - 1]


def median_lag_ms(lags_ns: Sequence[float]) -> float | None:
    """Median of lags that may include ``inf`` (censored), in ms."""
    if not lags_ns:
        return None
    return float(statistics.median(lags_ns)) / 1e6


# ---------------------------------------------------------------- sustained (15.2)


def sustained_thresholds(condition: Condition) -> tuple[float, float]:
    """``(slope ms/s, median lag ms)``: one publication interval of backlog across the window,
    and one publication interval."""
    return condition.interval_ms / condition.window_s, condition.interval_ms


def is_sustained(slope_ms_per_s: float | None, last_second_median_ms: float | None, memory_ceiling: bool,
                 condition: Condition) -> bool:
    limit_slope, limit_lag = sustained_thresholds(condition)
    if memory_ceiling or slope_ms_per_s is None or last_second_median_ms is None:
        return False
    return slope_ms_per_s <= limit_slope and last_second_median_ms <= limit_lag


# ---------------------------------------------------------------- per-run metrics (10)


def load(json_path: Path) -> tuple[dict[str, Any], Arrays]:
    record: dict[str, Any] = json.loads(json_path.read_text())
    npz = json_path.with_suffix(".npz")
    arrays: Arrays = {}
    if npz.exists():
        with np.load(npz) as data:
            arrays = {k: data[k] for k in data.files}
    return record, arrays


def _condition(record: dict[str, Any]) -> Condition:
    request = record["request"]
    return Condition.from_record(request["condition"]) if "condition" in request else Condition()


def producer_metrics(record: dict[str, Any], arrays: Arrays) -> dict[str, Any]:
    """10.1 over the window W."""
    c = _condition(record)
    t0, t1, t_start = record["t0"], record["t1"], record["t_start"]
    window_s = (t1 - t0) / 1e9
    count = record["batches"]
    a = t_start + arrays["schedule_offset"]
    q, s, e = arrays["producer_q"], arrays["producer_s"], arrays["producer_e"]
    cq, cs, ce, p = arrays["producer_cq"], arrays["producer_cs"], arrays["producer_ce"], arrays["producer_p"]
    released_a = a[:count]
    b = c.batch_size
    in_w_sched = (a >= t0) & (a < t1)
    out: dict[str, Any] = {"window_s": window_s}
    out["offered_rate"] = int(in_w_sched.sum()) * b / window_s
    e_in = (e >= t0) & (e < t1)
    out["achieved_rate"] = int(e_in.sum()) * b / window_s
    out["achieved_ratio"] = out["achieved_rate"] / out["offered_rate"] if out["offered_rate"] else None
    step_in = (s >= t0) & (s < t1)
    steps = (e - s)[step_in]
    out["step_us"] = percentiles(steps, 1e3)
    step_p = out["step_us"]
    out["step_p99_us"] = step_p["p99"]
    out["jitter_us"] = (None if step_p["p99"] is None or step_p["p50"] is None
                        else float(step_p["p99"]) - float(step_p["p50"]))
    released_in_w = (released_a >= t0) & (released_a < t1)
    lags = (e - released_a)[released_in_w]
    unreleased_in_w = int(in_w_sched[count:].sum())
    all_lags = [float(v) for v in lags] + [math.inf] * unreleased_in_w
    out["lag_ms"] = {
        "median": median_lag_ms(all_lags),
        "p99": None if not all_lags else rank_value(all_lags, 99) / 1e6,
        "max": None if not all_lags else max(all_lags) / 1e6,
        "unreleased_in_window": unreleased_in_w,
    }
    last = (a >= t1 - 1_000_000_000) & (a < t1)
    last_released = last[:count]
    last_lags = [float(v) for v in (e - released_a)[last_released]] + [math.inf] * int(last[count:].sum())
    out["lag_last_second_median_ms"] = median_lag_ms(last_lags)
    lag_slope = slope(released_a[released_in_w] / 1e9, lags / 1e6)
    out["lag_slope_ms_per_s"] = lag_slope
    out["release_lateness_ms"] = percentiles((q - released_a)[released_in_w], 1e6)
    backlog = int(((a <= t1)[count:]).sum())
    out["source_backlog_batches"] = backlog
    out["source_backlog_events"] = backlog * b
    out["implied_buffer_mib"] = backlog * b * 13 / MIB
    off_cpu = ((e - s) - (ce - cs))[step_in]
    out["off_cpu_ms_per_s"] = float(off_cpu.sum()) / 1e6 / window_s
    out["off_cpu_step_p99_us"] = percentiles(off_cpu.clip(min=0), 1e3)["p99"]
    busy = int(steps.sum())
    stepped_events = int(step_in.sum()) * b
    out["busy_fraction"] = busy / 1e9 / window_s
    out["busy_throughput"] = stepped_events / (busy / 1e9) if busy else None
    out["busy_ns_per_event"] = busy / stepped_events if stepped_events else None
    out["producer_cpu_cores"] = float((ce - cq)[step_in].sum()) / 1e9 / window_s
    out["stepped_events"] = stepped_events
    out["step_cpu_ns_per_event"] = float((ce - cs)[step_in].sum()) / stepped_events if stepped_events else None
    first_seen: dict[int, int] = {}
    for k, value in enumerate(p.tolist()):
        if value and value not in first_seen:
            first_seen[value] = k
    out["publication_steps"] = first_seen
    out["publication_rate"] = (sum(1 for k in first_seen.values() if t0 <= s[k] < t1) / window_s
                               if record["request"]["arm"] != "RB" else None)
    sustained = is_sustained(lag_slope, out["lag_last_second_median_ms"], bool(record.get("memory_ceiling")), c)
    out["sustained"] = sustained
    return out


def _queue_depth(enqueue: list[int], dequeue: list[int], t0: int, t1: int) -> dict[str, float | None]:
    """Depth as a step function of enqueue and dequeue instants, sampled every 100 ms in W."""
    events = sorted([(t, 1) for t in enqueue] + [(t, -1) for t in dequeue])
    grid = np.arange(t0, t1, 100_000_000, dtype=np.int64)
    if not len(grid):
        return {"p50": None, "max": None, "slope_per_s": None}
    times = np.array([t for t, _ in events], dtype=np.int64)
    depth = np.cumsum([d for _, d in events]) if events else np.zeros(0, dtype=np.int64)
    idx = np.searchsorted(times, grid, side="right") - 1
    sampled = np.where(idx >= 0, depth[np.clip(idx, 0, None)] if len(depth) else 0, 0).astype(np.int64)
    return {"p50": float(nearest_rank(sampled.tolist(), 50)), "max": float(sampled.max()),
            "slope_per_s": slope(grid / 1e9, sampled)}


def queue_depth(record: dict[str, Any], arrays: Arrays, prod: dict[str, Any], index: int) -> dict[str, float | None]:
    """10.2: enqueue at the step's E_k, dequeue at O (FF) or at the raw block's start."""
    arm = record["request"]["arm"]
    t0, t1 = record["t0"], record["t1"]
    ql = record["queues"][index]
    e = arrays["producer_e"]
    abandoned = set(ql["abandoned"])
    if arm in ("B", "C", "E"):
        steps = prod["publication_steps"]
        enq = [int(e[steps[seq]]) for seq in sorted(steps) if seq not in abandoned]
        deq = [int(v) for v in arrays[f"c{index}_o"]]
        deq += [int(e[steps[new]]) for _, new in ql["evicted"]]
    else:
        count = record["batches"]
        enq = [int(e[k]) for k in range(count) if k not in abandoned]
        deq = []
        if arm == "RB":
            for start, _, _, _, batches, _ in arrays[f"c{index}_blocks"].tolist():
                deq += [int(start)] * int(batches)
        else:
            deq = [int(v) for v in arrays["recorder_o"]]
    return _queue_depth(enq, deq, t0, t1)


def consumer_metrics(record: dict[str, Any], arrays: Arrays, prod: dict[str, Any]) -> dict[str, Any]:
    """10.2, pooled over the run's consumers, with per-consumer values."""
    arm = record["request"]["arm"]
    c = _condition(record)
    t0, t1 = record["t0"], record["t1"]
    window_s = (t1 - t0) / 1e9
    a = record["t_start"] + arrays["schedule_offset"]
    s, e = arrays["producer_s"], arrays["producer_e"]
    count = record["batches"]
    find = resolver(arrays["m"].tolist(), s[:count].tolist())
    pooled: dict[str, list[int]] = defaultdict(list)
    per: list[dict[str, Any]] = []
    steps: dict[int, int] = prod["publication_steps"]
    published_in_w = sorted(seq for seq, k in steps.items() if t0 <= s[k] < t1)
    window_events: dict[int, int] = {}
    previous = -1
    for seq in sorted(steps):
        k = steps[seq]
        window_events[seq] = (k - previous) * c.batch_size
        previous = k
    series: list[tuple[int, int]] = []
    for i in range(record["consumers"]):
        o, w = arrays[f"c{i}_o"], arrays[f"c{i}_w"]
        co, cw = arrays[f"c{i}_co"], arrays[f"c{i}_cw"]
        seqs, wms = arrays[f"c{i}_seq"], arrays[f"c{i}_wm"]
        in_w = (o >= t0) & (o < t1)
        fresh, e2e, done, post = [], [], [], []
        for ob, wb, wm in zip(o[in_w].tolist(), w[in_w].tolist(), wms[in_w].tolist()):
            kstar = find(int(wm), int(ob))
            if kstar is None:
                continue
            fresh.append(ob - int(s[kstar]))
            e2e.append(ob - int(a[kstar]))
            done.append(wb - int(s[kstar]))
            post.append(ob - int(e[kstar]))
            series.append((ob, ob - int(s[kstar])))
        proc = (w - o)[in_w]
        cpu = (cw - co)[in_w]
        mine: dict[str, Any] = {
            "observation_rate": int(in_w.sum()) / window_s,
            "freshness_ms": percentiles(fresh, 1e6),
            "end_to_end_ms": percentiles(e2e, 1e6),
            "completion_age_ms": percentiles(done, 1e6),
            "post_step_ms": percentiles([v for v in post], 1e6) if arm != "A" else None,
            "processing_ms": percentiles(proc, 1e6),
            "processing_cpu_ms": percentiles(cpu, 1e6),
        }
        ratio = [float(pv) / float(cv) for pv, cv in zip(proc.tolist(), cpu.tolist()) if cv > 0]
        mine["contention_p50"] = rank_value(ratio, 50) if ratio else None
        mine["contention_p99"] = rank_value(ratio, 99) if ratio else None
        samples = arrays[f"c{i}_samples"].tolist()
        mine["consumer_cpu_cores"] = ((samples[3] - samples[1]) / 1e9 / window_s
                                      if samples[4] and samples[5] else None)
        mine["polls_per_s"] = samples[6] / window_s if arm in POLL_ARMS else None
        if arm in ("B", "C", "E") or arm in POLL_ARMS or arm == "A":
            ql = record["queues"][i] if arm in ("B", "C", "E") else None
            # Backlog at stop (7.4): still queued, or a blocked put abandoned when the run stopped.
            remaining = set(ql["remaining"]) | set(ql["abandoned"]) if ql else set()
            observed = set(seqs.tolist())
            covered = [seq for seq in published_in_w if seq in observed or seq in remaining]
            within = [seq for seq in published_in_w
                      if seq in set(seqs[in_w].tolist())]
            mine["coverage"] = len(covered) / len(published_in_w) if published_in_w else None
            mine["coverage_within_w"] = len(within) / len(published_in_w) if published_in_w else None
            mine["skipped"] = len(published_in_w) - len(covered) if arm in POLL_ARMS else 0
            mine["declared_drops"] = (sum(1 for old, _ in ql["evicted"] if old in set(published_in_w))
                                      if ql else 0)
            total = sum(window_events[seq] for seq in published_in_w)
            unobserved = sum(window_events[seq] for seq in published_in_w if seq not in observed
                             and seq not in remaining)
            mine["unobserved_fraction"] = unobserved / total if total else None
        if arm == "RB":
            blocks = arrays[f"c{i}_blocks"]
            in_blocks = (blocks[:, 2] >= t0) & (blocks[:, 2] < t1) if len(blocks) else np.zeros(0, dtype=bool)
            mine["raw_rate"] = float(blocks[in_blocks, 5].sum()) / window_s if len(blocks) else 0.0
            mine["accumulation_cpu_ns"] = float((blocks[in_blocks, 3] - blocks[in_blocks, 1]).sum()) if len(blocks) else 0.0
            mine["accumulated_events"] = float(blocks[in_blocks, 5].sum()) if len(blocks) else 0.0
        if arm in QUEUE_ARMS:
            mine["queue_depth"] = queue_depth(record, arrays, prod, i)
        per.append(mine)
        pooled["freshness"] += fresh
        pooled["end_to_end"] += e2e
        pooled["completion"] += done
        pooled["post"] += post
        pooled["processing"] += proc.tolist()
        pooled["cpu"] += cpu.tolist()
    out: dict[str, Any] = {"per_consumer": per}
    out["freshness_ms"] = percentiles(pooled["freshness"], 1e6)
    out["freshness_p50_ms"] = out["freshness_ms"]["p50"]
    out["freshness_p95_ms"] = out["freshness_ms"]["p95"]
    out["end_to_end_ms"] = percentiles(pooled["end_to_end"], 1e6)
    out["completion_age_ms"] = percentiles(pooled["completion"], 1e6)
    out["post_step_ms"] = percentiles(pooled["post"], 1e6) if arm != "A" else None
    out["processing_ms"] = percentiles(pooled["processing"], 1e6)
    out["processing_cpu_ms"] = percentiles(pooled["cpu"], 1e6)
    out["observation_rate"] = (statistics.mean(m["observation_rate"] for m in per) if per else None)
    for key in ("coverage", "unobserved_fraction", "consumer_cpu_cores", "raw_rate", "contention_p50",
                "contention_p99"):
        values = [m[key] for m in per if m.get(key) is not None]
        out[key] = statistics.mean(values) if values else None
    out["consumer_cpu_total_cores"] = sum(m["consumer_cpu_cores"] or 0.0 for m in per) if per else 0.0
    out["skipped"] = sum(m.get("skipped", 0) for m in per)
    out["declared_drops"] = sum(m.get("declared_drops", 0) for m in per)
    depths = [m["queue_depth"] for m in per if "queue_depth" in m]
    out["queue_depth_slope"] = (statistics.mean(d["slope_per_s"] for d in depths if d["slope_per_s"] is not None)
                                if any(d["slope_per_s"] is not None for d in depths) else None)
    out["queue_depth_max"] = max((d["max"] or 0.0) for d in depths) if depths else None
    if series:
        series.sort()
        xs = np.array([x for x, _ in series], dtype=np.float64)
        ys = np.array([y for _, y in series], dtype=np.float64)
        out["freshness_spearman"] = spearman(xs, ys)
    else:
        out["freshness_spearman"] = None
    out["freshness_series"] = [[(x - t0) / 1e9, y / 1e6] for x, y in series]
    if arm == "RB" and per:
        events = statistics.mean(m["accumulated_events"] for m in per)
        cpu_total = sum(m["accumulation_cpu_ns"] for m in per)
        out["accumulation_cpu_ns_per_event"] = cpu_total / events if events else None
    return out


def _at(times: NDArray[Any], values: NDArray[Any], t: int, after: bool) -> tuple[int, float] | None:
    if not len(times):
        return None
    if after:
        idx = int(np.searchsorted(times, t, side="left"))
        if idx >= len(times):
            return None
    else:
        idx = int(np.searchsorted(times, t, side="right")) - 1
        if idx < 0:
            return None
    return int(times[idx]), float(values[idx])


def system_metrics(record: dict[str, Any], arrays: Arrays, stepped_events: int) -> dict[str, Any]:
    """10.3 from the monitor's samples."""
    t0, t1 = record["t0"], record["t1"]
    out: dict[str, Any] = {}
    times = arrays.get("monitor_t", np.zeros(0, dtype=np.int64))
    gr = arrays.get("monitor_getrusage")
    out["ru_maxrss_mib"] = float(gr[-1, 2]) / MIB if gr is not None and len(gr) else None
    if "rusage_phys_footprint" not in arrays or not len(times):
        return out
    fp = arrays["rusage_phys_footprint"].astype(np.float64)
    out["peak_footprint_mib"] = float(arrays["rusage_lifetime_max_phys_footprint"][-1]) / MIB
    out["baseline_footprint_mib"] = float(fp[0]) / MIB
    in_w = (times >= t0) & (times < t1)
    out["footprint_growth_mib_s"] = slope(times[in_w] / 1e9, fp[in_w] / MIB)
    start, end = _at(times, fp, t0, True), _at(times, fp, t1, False)
    out["footprint_delta_mib"] = None if start is None or end is None else (end[1] - start[1]) / MIB
    out["footprint_series"] = [[(int(t) - record["t_start"]) / 1e9, float(v) / MIB]
                               for t, v in zip(times.tolist(), fp.tolist())]

    def rate(fields: Iterable[str]) -> tuple[float, float] | None:
        total = np.zeros(len(times), dtype=np.float64)
        for f in fields:
            total = total + arrays[f"rusage_{f}"].astype(np.float64)
        a, b = _at(times, total, t0, True), _at(times, total, t1, False)
        if a is None or b is None or b[0] <= a[0]:
            return None
        return b[1] - a[1], (b[0] - a[0]) / 1e9

    cpu = rate(("user_time", "system_time"))
    pcpu = rate(("user_ptime", "system_ptime"))
    out["process_cpu_cores"] = None if cpu is None else cpu[0] / 1e9 / cpu[1]
    out["p_core_share"] = None if cpu is None or pcpu is None or not cpu[0] else pcpu[0] / cpu[0]
    cycles = rate(("cycles",))
    out["cycles_per_cpu_ns"] = None if cycles is None or cpu is None or not cpu[0] else cycles[0] / cpu[0]
    runnable = rate(("runnable_time",))
    out["runnable_ms_per_s"] = None if runnable is None else runnable[0] / 1e6 / runnable[1]
    energy = rate(("energy_nj",))
    out["energy_j"] = None if energy is None else energy[0] / 1e9
    out["energy_j_per_1e9_events"] = (None if energy is None or not stepped_events
                                      else energy[0] / 1e9 / stepped_events * 1e9)
    g0, g1 = record.get("gc_t0"), record.get("gc_t1")
    out["gc_collections"] = (None if not g0 or not g1 else
                             [b["collections"] - a["collections"] for a, b in zip(g0, g1)])
    out["run_duration_s"] = record.get("child_duration_ns", 0) / 1e9
    return out


def run_metrics(record: dict[str, Any], arrays: Arrays) -> dict[str, Any]:
    prod = producer_metrics(record, arrays)
    cons = consumer_metrics(record, arrays, prod) if record["consumers"] else {}
    system = system_metrics(record, arrays, prod["stepped_events"])
    out = {**{k: v for k, v in prod.items() if k != "publication_steps"}, **cons, **system}
    arm = record["request"]["arm"]
    if arm == "RB":
        out["accumulation_path_cpu_ns_per_event"] = cons.get("accumulation_cpu_ns_per_event")
    else:
        out["accumulation_path_cpu_ns_per_event"] = prod["step_cpu_ns_per_event"]
    if arm in P2_ARMS:
        out["readback_ok"] = (record.get("readback") or {}).get("ok")
        out["drain_timeout"] = bool(record.get("drain_timeout"))
        out["completion_delay_ms"] = (None if record.get("completion_delay_ns") is None
                                      else record["completion_delay_ns"] / 1e6)
        if record["queues"]:
            out["recorder_backlog"] = queue_depth(record, arrays, prod, 0)
    return out


# ---------------------------------------------------------------- cells (16)


Key = tuple[str, str, int, str]
"""``(arm, workload, n, runtime)``."""


def cell_key(record: dict[str, Any]) -> Key:
    r = record["request"]
    return r["arm"], r["workload"], int(r["n"]), r["runtime"]


def aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median, min and max of each scalar metric over a cell's valid runs (16.1)."""
    out: dict[str, Any] = {"n_valid": len(runs)}
    keys = sorted({k for r in runs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for key in keys:
        values = [float(r[key]) for r in runs if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
                  and r[key] is not None]
        finite = [v for v in values if math.isfinite(v)]
        if values:
            out[key] = {"median": statistics.median(values), "min": min(values), "max": max(values),
                        "n": len(values), "finite": len(finite)}
    sustained = [bool(r["sustained"]) for r in runs if "sustained" in r]
    yes = sum(sustained)
    out["memory_ceiling_runs"] = sum("MEMORY_CEILING" in str(r.get("flags", "")) for r in runs)
    if any("readback_ok" in r for r in runs):
        out["readback_all_ok"] = all(r.get("readback_ok") is True for r in runs)
        out["drain_timeout_runs"] = sum(bool(r.get("drain_timeout")) for r in runs)
    out["sustained_runs"] = yes
    out["label"] = ("SUSTAINED" if yes >= 4 else "NOT_SUSTAINED" if len(sustained) - yes >= 4 else "mixed")
    return out


def median_of(cell: dict[str, Any] | None, metric: str) -> float | None:
    if cell is None or metric not in cell:
        return None
    return float(cell[metric]["median"])


def band(cells_: dict[Key, dict[str, Any]], repetitions: int = REPETITIONS) -> dict[str, Any]:
    """16.3: the largest H′/H ratio of cell medians, per metric, over the A/A cells."""
    out: dict[str, Any] = {"excluded": []}
    for metric in BAND_METRICS:
        worst: float | None = None
        for key, h in cells_.items():
            if key[0] != "H":
                continue
            hp = cells_.get(("H'",) + key[1:])
            if hp is None:
                continue
            if h["n_valid"] < repetitions or hp["n_valid"] < repetitions:
                if list(key) not in out["excluded"]:
                    out["excluded"].append(list(key))
                continue
            a, b = median_of(hp, metric), median_of(h, metric)
            if a is None or b is None or a <= 0 or b <= 0 or not math.isfinite(a) or not math.isfinite(b):
                continue
            r = max(a / b, b / a)
            worst = r if worst is None else max(worst, r)
        out[metric] = worst
    return out


def distinguishable(x: dict[str, Any] | None, y: dict[str, Any] | None, metric: str,
                    band_m: float | None) -> bool | None:
    """16.3: outside the band and non-overlapping ranges. ``None`` if it can't be evaluated."""
    if x is None or y is None or band_m is None or metric not in x or metric not in y:
        return None
    mx, my = x[metric]["median"], y[metric]["median"]
    if mx <= 0 or my <= 0 or not math.isfinite(mx) or not math.isfinite(my):
        return None
    ratio = mx / my
    outside = ratio > band_m or ratio < 1 / band_m
    disjoint = x[metric]["max"] < y[metric]["min"] or y[metric]["max"] < x[metric]["min"]
    return bool(outside and disjoint)


def excess_p99(cells_: dict[Key, dict[str, Any]], key: Key) -> float | None:
    """Cell p99(N) − cell p99(N = 0), µs; B, C and E use A's N = 0 cell (10.1)."""
    arm, _, n, rt = key
    if n == 0 or arm == "RB":
        return None
    ref_arm = "A" if arm in ("B", "C", "E") else arm
    here, ref = median_of(cells_.get(key), "step_p99_us"), median_of(cells_.get((ref_arm, "none", 0, rt)), "step_p99_us")
    return None if here is None or ref is None else here - ref


# ---------------------------------------------------------------- hypotheses (17)


Verdict = dict[str, Any]


def _verdict(supported: bool | None, contradicted: bool | None, evidence: dict[str, Any]) -> Verdict:
    if supported is None and contradicted is None:
        verdict = "inconclusive"
    elif supported and contradicted:
        verdict = "inconclusive"
        evidence = {**evidence, "note": "both the support and the contradiction condition hold"}
    elif supported:
        verdict = "supported"
    elif contradicted:
        verdict = "contradicted"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "supported_condition": supported, "contradicted_condition": contradicted,
            "evidence": evidence}


def _all(values: Iterable[bool | None]) -> bool | None:
    """``False`` if any is false, ``None`` if any is unknown (or there are none), else ``True``."""
    items = list(values)
    if any(v is False for v in items):
        return False
    if not items or any(v is None for v in items):
        return None
    return True


def _any(values: Iterable[bool | None]) -> bool | None:
    items = [v for v in values if v is not None]
    return any(items) if items else None


def judge(cells_: dict[Key, dict[str, Any]], bands: dict[str, Any], p2: dict[Key, dict[str, Any]],
          p2_runs: dict[Key, list[dict[str, Any]]]) -> dict[str, Verdict]:
    """Every hypothesis by its preregistered rule (17), except H7, which is interpretation."""
    out: dict[str, Verdict] = {}
    get = cells_.get
    workloads, ns, rts = ("W1", "W3", "W5"), (1, 4), ("A", "B")

    def label(key: Key) -> str | None:
        cell = get(key)
        return None if cell is None else str(cell["label"])

    # H1
    ev: dict[str, Any] = {}
    s_need, s_one, c_hit = [], [], []
    for rt in rts:
        b = median_of(get(("A", "none", 0, rt)), "busy_fraction")
        for w in workloads:
            for n in ns:
                cell = get(("A", w, n, rt))
                lat = median_of(cell, "processing_p50_ms")
                if cell is None or lat is None or b is None:
                    continue
                load = n * lat + b * 16.0
                ev[f"{w} N={n} {rt}"] = {"load_ms": load, "label": cell["label"]}
                if load > 16.0:
                    s_need.append(cell["label"] == "NOT_SUSTAINED")
                else:
                    s_one.append(cell["label"] == "SUSTAINED")
                if n * lat >= 32.0:
                    c_hit.append(cell["label"] == "SUSTAINED")
    out["H1"] = _verdict(None if not s_need else (all(s_need) and any(s_one)), _any(c_hit), ev)

    # H2
    ev, s_cells, c_cells = {}, [], []
    for rt in rts:
        for w in ("W3", "W5"):
            for n in ns:
                cell = get(("B", w, n, rt))
                if cell is None:
                    continue
                pub, obs = median_of(cell, "publication_rate"), median_of(cell, "observation_rate")
                growth = median_of(cell, "footprint_growth_mib_s")
                within = None if pub is None or not obs else abs(pub - obs) <= 0.2 * obs
                ok = (cell["label"] == "NOT_SUSTAINED" and bool(within)
                      and growth is not None and growth <= GROWTH_LIMIT_MIB_S)
                s_cells.append(ok)
                if n == 1:
                    c_cells.append(cell["label"] == "SUSTAINED")
                ev[f"{w} N={n} {rt}"] = {"label": cell["label"], "publication_rate": pub,
                                         "observation_rate": obs, "growth": growth}
    out["H2"] = _verdict(_all(s_cells), _any(c_cells), ev)

    # H3
    ev, s_cells, c_cells = {}, [], []
    for rt in rts:
        for w in ("W3", "W5"):
            for n in ns:
                cell = get(("C", w, n, rt))
                if cell is None:
                    continue
                growth, drops = median_of(cell, "footprint_growth_mib_s"), median_of(cell, "declared_drops")
                s_cells.append(cell["label"] == "SUSTAINED" and growth is not None and growth <= GROWTH_LIMIT_MIB_S
                               and drops is not None and drops > 0)
                if w == "W3":
                    c_cells.append(cell["label"] == "NOT_SUSTAINED")
                ev[f"{w} N={n} {rt}"] = {"label": cell["label"], "growth": growth, "drops": drops}
    out["H3"] = _verdict(_all(s_cells), _any(c_cells), ev)

    # H4
    ev, s_cells, c_cells = {}, [], []
    for rt in rts:
        for w in ("W3", "W5"):
            for n in ns:
                cell = get(("E", w, n, rt))
                if cell is None:
                    continue
                depth, rho = median_of(cell, "queue_depth_slope"), median_of(cell, "freshness_spearman")
                drops, lat = median_of(cell, "declared_drops"), median_of(cell, "processing_p50_ms")
                s_cells.append(depth is not None and depth > DEPTH_SLOPE_LIMIT and rho is not None and rho > 0.8
                               and drops == 0)
                slower = lat is not None and lat > 16.0
                c_cells.append(slower and depth is not None and depth <= DEPTH_SLOPE_LIMIT)
                ev[f"{w} N={n} {rt}"] = {"depth_slope": depth, "spearman": rho, "drops": drops, "processing_ms": lat}
    out["H4"] = _verdict(_all(s_cells), _any(c_cells), ev)

    # H5
    ev, s_cells, c_cells = {}, [], []
    for rt in rts:
        for w in workloads:
            x1, x4 = excess_p99(cells_, ("F", w, 1, rt)), excess_p99(cells_, ("F", w, 4, rt))
            if x1 is not None and x4 is not None:
                s_cells.append(x4 > x1)
            d = distinguishable(get(("F", w, 4, rt)), get(("G", w, 4, rt)), "step_p99_us", bands.get("step_p99_us"))
            c_cells.append(None if d is None else not d)
            ev[f"{w} {rt}"] = {"excess_n1_us": x1, "excess_n4_us": x4, "F_vs_G_N4_distinguishable": d}
    c_all = _all(c_cells)
    out["H5"] = _verdict(_all(s_cells), c_all, ev)

    # H6
    ev = {}
    s_rt, c_rt = [], []
    for rt in rts:
        pairs = [(w, n) for w in workloads for n in ns] + [("none", 0)]
        tie_all, dist_prod = [], []
        for w, n in pairs:
            g, h = get(("G", w, n, rt)), get(("H", w, n, rt))
            if g is None or h is None or g["n_valid"] < REPETITIONS or h["n_valid"] < REPETITIONS:
                ev[f"{w} N={n} {rt}"] = f"left out: fewer than {REPETITIONS} valid runs"
                continue
            metrics = ["busy_ns_per_event", "step_p99_us"] + (["freshness_p50_ms"] if n else [])
            dist = {m: distinguishable(g, h, m, bands.get(m)) for m in metrics}
            ev[f"{w} N={n} {rt}"] = {m: {"distinguishable": dist[m], "ratio": _ratio(g, h, m)} for m in metrics}
            if any(v is None for v in dist.values()):
                continue
            tie_all.append(not any(dist.values()))
            dist_prod.append(bool(dist["busy_ns_per_event"] or dist["step_p99_us"]))
        if tie_all:
            s_rt.append(sum(tie_all) >= 0.75 * len(tie_all))
            c_rt.append(sum(dist_prod) > 0.5 * len(dist_prod))
    out["H6"] = _verdict(_all(s_rt) if s_rt else None, _any(c_rt), ev)

    # H8
    ev = {}
    readback = [r.get("readback_ok") for runs in p2_runs.values() for r in runs]
    ep_h = [p2.get(("EP-H", "W1" if n else "none", n, rt)) for n in (0, 1) for rt in rts]
    s8 = (all(v is True for v in readback) if readback else None)
    s8 = None if s8 is None else (s8 and all(c is not None and c["label"] == "SUSTAINED" for c in ep_h))
    drain = [bool(r.get("drain_timeout")) for (arm, *_), runs in p2_runs.items() if arm in ("EP-QB", "EP-QE")
             for r in runs]
    ep_h0 = [p2.get(("EP-H", "none", 0, rt)) for rt in rts]
    c8 = _any([any(drain) if drain else None] + [None if c is None else c["label"] == "NOT_SUSTAINED" for c in ep_h0])
    ev = {"readback_all_ok": s8 if readback else None, "drain_timeouts": sum(drain),
          "EP-H labels": {f"N_SO={n} {rt}": (None if p2.get(("EP-H", "W1" if n else "none", n, rt)) is None
                                            else p2[("EP-H", "W1" if n else "none", n, rt)]["label"])
                          for n in (0, 1) for rt in rts}}
    out["H8"] = _verdict(s8, c8, ev)

    # H9
    ev, s_cells, c_cells = {}, [], []
    for rt in rts:
        for w in ("W1", "W3"):
            for n in ns:
                cell = get(("H", w, n, rt))
                if cell is None:
                    continue
                growth, p95 = median_of(cell, "footprint_growth_mib_s"), median_of(cell, "freshness_p95_ms")
                s_cells.append(cell["label"] == "SUSTAINED" and growth is not None and growth <= GROWTH_LIMIT_MIB_S
                               and p95 is not None and p95 <= 32.0)
                c_cells.append(cell["label"] == "NOT_SUSTAINED")
                ev[f"{w} N={n} {rt}"] = {"label": cell["label"], "growth": growth, "freshness_p95_ms": p95}
    out["H9"] = _verdict(_all(s_cells), _any(c_cells), ev)

    # H10
    ev, s_cells = {}, []
    for arm in ("F", "G", "H"):
        for n in ns:
            xa, xb = excess_p99(cells_, (arm, "W5", n, "A")), excess_p99(cells_, (arm, "W5", n, "B"))
            s_cells.append(None if xa is None else xa >= 5000.0)
            s_cells.append(None if xb is None else xb < 5000.0)
            ev[f"{arm} N={n}"] = {"excess_A_us": xa, "excess_B_us": xb}
    g4, h4 = excess_p99(cells_, ("G", "W5", 4, "A")), excess_p99(cells_, ("H", "W5", 4, "A"))
    c10 = None if g4 is None or h4 is None else (g4 < 5000.0 and h4 < 5000.0)
    out["H10"] = _verdict(_all(s_cells), c10, ev)

    # H11
    ev, flags = {}, []
    for rt in rts:
        for arm in ("F", "G", "H"):
            for n in ns:
                d = distinguishable(get((arm, "W3", n, rt)), get((arm, "none", 0, rt)), "step_p99_us",
                                    bands.get("step_p99_us"))
                flags.append(d)
                ev[f"{arm} N={n} {rt}"] = {"distinguishable_from_N0": d}
    known = [f for f in flags if f is not None]
    s11 = None if not known or len(known) < len(flags) else not any(known)
    c11 = None if not known else sum(known) > 0.5 * len(known)
    out["H11"] = _verdict(s11, c11, ev)

    # H12
    ev, ratios = {}, []
    for rt in rts:
        for w in workloads:
            one = median_of(get(("RB", w, 1, rt)), "accumulation_path_cpu_ns_per_event")
            four = median_of(get(("RB", w, 4, rt)), "accumulation_path_cpu_ns_per_event")
            r = None if one is None or four is None or not one else four / one
            ratios.append(r)
            ev[f"{w} {rt}"] = {"ratio_N4_over_N1": r}

    def smallest_unsustained(arm: str) -> float:
        for n in ns:
            if label((arm, "W1", n, "A")) == "NOT_SUSTAINED":
                return float(n)
        return math.inf

    rb_n, b_n = smallest_unsustained("RB"), smallest_unsustained("B")
    ev["smallest NOT_SUSTAINED N, runtime A, W1"] = {"RB": rb_n, "B": b_n}
    known_r = [r for r in ratios if r is not None]
    s12 = None if len(known_r) < len(ratios) else (all(3.0 <= r <= 5.0 for r in known_r)
                                                  and math.isfinite(rb_n) and rb_n < b_n)
    c12 = None if not known_r else any(r < 2.0 for r in known_r)
    out["H12"] = _verdict(s12, c12, ev)

    # H13
    ev, s_cells, c_cells = {}, [], []
    for rt in rts:
        for arm in ("B", "C", "E", "RB"):
            v = median_of(get((arm, "W1", 1, rt)), "post_step_p50_ms")
            s_cells.append(None if v is None else v < 2.0)
            ev[f"{arm} {rt}"] = v
        for arm in ("F", "G", "H"):
            v = median_of(get((arm, "W1", 1, rt)), "post_step_p50_ms")
            s_cells.append(None if v is None else 4.0 <= v <= 12.0)
            c_cells.append(None if v is None else v < 2.0)
            ev[f"{arm} {rt}"] = v
    out["H13"] = _verdict(_all(s_cells), _any(c_cells), ev)

    # H14
    ev, higher, pcore, lower = {}, [], [], []
    for rt in rts:
        for arm in ("F", "G", "H"):
            zero = get((arm, "none", 0, rt))
            for w in workloads:
                with_one = get((arm, w, 1, rt))
                t0_, t1_ = median_of(zero, "busy_throughput"), median_of(with_one, "busy_throughput")
                p0, p1 = median_of(zero, "p_core_share"), median_of(with_one, "p_core_share")
                if t0_ is None or t1_ is None:
                    continue
                higher.append(t1_ > t0_)
                lower.append(t1_ <= t0_)
                pcore.append(p0 is not None and p1 is not None and p1 > p0)
                ev[f"{arm} {w} {rt}"] = {"busy_throughput_N0": t0_, "busy_throughput_N1": t1_,
                                        "p_core_share_N0": p0, "p_core_share_N1": p1}
    s14 = None if not higher else (sum(higher) > 0.5 * len(higher) and sum(pcore) > 0.5 * len(pcore))
    c14 = None if not lower else sum(lower) > 0.5 * len(lower)
    out["H14"] = _verdict(s14, c14, ev)
    return out


def _ratio(x: dict[str, Any], y: dict[str, Any], metric: str) -> float | None:
    a, b = median_of(x, metric), median_of(y, metric)
    return None if a is None or b is None or not b else a / b


# ---------------------------------------------------------------- the dataset


def flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    """Scalar per-run values for aggregation: nested percentile dicts become ``name_pNN``."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict) and set(value) >= {"p50", "p95", "p99", "max"}:
            base = key[:-3] if key.endswith(("_ms", "_us")) else key
            unit = key[-3:] if key.endswith(("_ms", "_us")) else ""
            for q in ("p50", "p95", "p99", "max"):
                out[f"{base}_{q}{unit}"] = value[q]
            out[f"{base}_samples"] = value["n"]
        elif isinstance(value, dict) and key == "lag_ms":
            for q, v in value.items():
                out[f"lag_{q}_ms" if q != "unreleased_in_window" else "lag_unreleased"] = v
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
    return out


def select_attempts(ledger: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The attempts analysed: for each (experiment, pass), those of the highest revision that
    ran it (24: a pass re-run after an amendment replaces the earlier one; both are kept)."""
    latest: dict[tuple[str, int], int] = {}
    for a in ledger:
        key = (a["experiment"], a["pass"])
        latest[key] = max(latest.get(key, -1), a["revision"])
    return [a for a in ledger if a["revision"] == latest[(a["experiment"], a["pass"])]]


def read_ledger(campaign: Path) -> list[dict[str, Any]]:
    path = campaign / "attempts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyse(campaign: Path, out: Path) -> dict[str, Any]:
    """Every preregistered table, and the data tables of the figures, from a campaign
    directory. Writes into *out*, which must not exist."""
    if out.exists():
        raise FileExistsError(f"{out} exists; analysis output is never overwritten")
    ledger = read_ledger(campaign)
    attempts = select_attempts(ledger)
    per_run: dict[Key, list[dict[str, Any]]] = defaultdict(list)
    run_rows: list[dict[str, Any]] = []
    series: dict[str, Any] = {}
    for a in attempts:
        if a["outcome"] != VALID:
            continue
        record, arrays = load(campaign / "runs" / f"{a['run_id']}.json")
        metrics = run_metrics(record, arrays)
        flat = flatten(metrics)
        flat.update({"run_id": a["run_id"], "pass": a["pass"], "flags": ",".join(a.get("flags", []))})
        key = cell_key(record)
        per_run[key].append(flat)
        run_rows.append({"arm": key[0], "workload": key[1], "n": key[2], "runtime": key[3], **flat})
        series[a["run_id"]] = {"freshness": metrics.get("freshness_series", []),
                               "footprint": metrics.get("footprint_series", [])}
    cells_ = {k: aggregate(v) for k, v in per_run.items() if k[0] in P1_ARMS}
    p2 = {k: aggregate(v) for k, v in per_run.items() if k[0] in P2_ARMS}
    p2_runs = {k: v for k, v in per_run.items() if k[0] in P2_ARMS}
    bands = band(cells_)
    verdicts = judge(cells_, bands, p2, p2_runs)
    out.mkdir(parents=True)
    tables = write_tables(out, cells_, p2, bands, verdicts, run_rows, ledger, attempts, series)
    summary = {"cells": len(cells_), "p2_cells": len(p2), "valid_runs": len(run_rows), "attempts": len(ledger),
               "band": bands, "verdicts": verdicts, "tables": tables}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default) + "\n")
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


# ---------------------------------------------------------------- tables (18)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        return f"{value:.{digits}g}"
    return str(value)


def _cell_value(cell: dict[str, Any] | None, metric: str) -> str:
    if cell is None or metric not in cell:
        return "n/a"
    m = cell[metric]
    return f"{_fmt(m['median'])} [{_fmt(m['min'])}, {_fmt(m['max'])}]"


def _arm_order(keys: Iterable[Key]) -> list[Key]:
    order = {arm: i for i, arm in enumerate(P1_ARMS + P2_ARMS)}
    return sorted(keys, key=lambda k: (k[3], k[1], k[2], order[k[0]]))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v, 6) if isinstance(v, float) else v for k, v in row.items()})


T2_METRICS: Final = (
    "achieved_ratio", "step_p50_us", "step_p99_us", "jitter_us", "lag_median_ms", "lag_slope_ms_per_s",
    "lag_last_second_median_ms", "release_lateness_p50_ms", "source_backlog_batches", "implied_buffer_mib",
    "off_cpu_ms_per_s", "busy_fraction", "busy_throughput", "busy_ns_per_event", "producer_cpu_cores",
    "accumulation_path_cpu_ns_per_event", "publication_rate", "observation_rate", "freshness_p50_ms",
    "freshness_p95_ms", "freshness_p99_ms", "freshness_max_ms", "end_to_end_p95_ms", "completion_age_p95_ms",
    "post_step_p50_ms", "processing_p50_ms", "processing_cpu_p50_ms", "contention_p50", "coverage",
    "skipped", "declared_drops", "unobserved_fraction", "consumer_cpu_total_cores", "queue_depth_slope",
    "queue_depth_max", "raw_rate", "peak_footprint_mib", "footprint_growth_mib_s", "baseline_footprint_mib",
    "process_cpu_cores", "p_core_share", "cycles_per_cpu_ns", "runnable_ms_per_s", "energy_j_per_1e9_events",
)


def write_tables(out: Path, cells_: dict[Key, dict[str, Any]], p2: dict[Key, dict[str, Any]], bands: dict[str, Any],
                 verdicts: dict[str, Verdict], run_rows: list[dict[str, Any]], ledger: Sequence[dict[str, Any]],
                 attempts: Sequence[dict[str, Any]], series: dict[str, Any]) -> list[str]:
    written: list[str] = []
    md: list[str] = ["# Observation study: tables", ""]

    # T7 and F15: the primary result, the full freshness distributions.
    rows = []
    for key in _arm_order(cells_):
        cell = cells_[key]
        if key[2] == 0:
            continue
        rows.append({"arm": key[0], "workload": key[1], "n": key[2], "runtime": key[3], "n_valid": cell["n_valid"],
                     **{f"{m}": _cell_value(cell, m) for m in ("freshness_p50_ms", "freshness_p95_ms",
                                                               "freshness_p99_ms", "freshness_max_ms",
                                                               "freshness_samples", "end_to_end_p95_ms",
                                                               "completion_age_p95_ms")}})
    _write_csv(out / "T7_freshness.csv", rows)
    md += ["## T7, freshness distributions (primary)", "", _markdown(rows), ""]
    written.append("T7_freshness.csv")

    # T1 (illustrative).
    rows = []
    for key in _arm_order(cells_):
        cell = cells_[key]
        p95 = median_of(cell, "freshness_p95_ms")
        growth = median_of(cell, "footprint_growth_mib_s")
        coverage_min = cell.get("coverage", {}).get("min") if isinstance(cell.get("coverage"), dict) else None
        producer_ok = (cell["sustained_runs"] >= 4 and growth is not None and growth <= GROWTH_LIMIT_MIB_S
                       and not cell["memory_ceiling_runs"])
        row: dict[str, Any] = {"workload": key[1], "n": key[2], "runtime": key[3], "arm": key[0],
                               "n_valid": cell["n_valid"]}
        for theta in FRESHNESS_THRESHOLDS_MS:
            row[f"SO-live({theta:g} ms)"] = bool(producer_ok and p95 is not None and p95 <= theta)
        row["SO-complete"] = bool(producer_ok and coverage_min == 1.0)
        rows.append(row)
    for key in _arm_order(p2):
        cell = p2[key]
        growth = median_of(cell, "footprint_growth_mib_s")
        bounded = growth is not None and growth <= GROWTH_LIMIT_MIB_S
        drained = not cell["memory_ceiling_runs"] and not cell.get("drain_timeout_runs")
        rows.append({"workload": key[1], "n": key[2], "runtime": key[3], "arm": key[0], "n_valid": cell["n_valid"],
                     "EP-events": bool(cell.get("readback_all_ok") and cell["sustained_runs"] >= 4
                                       and (bounded or drained))})
    _write_csv(out / "T1_decision_boundary.csv", rows)
    md += ["## T1, decision boundary (illustrative thresholds)", "", _markdown(rows), ""]
    written.append("T1_decision_boundary.csv")

    # T2.
    rows = []
    for key in _arm_order({**cells_, **p2}):
        cell = cells_.get(key) or p2[key]
        rows.append({"arm": key[0], "workload": key[1], "n": key[2], "runtime": key[3], "n_valid": cell["n_valid"],
                     "label": cell["label"], **{m: _cell_value(cell, m) for m in T2_METRICS}})
    _write_csv(out / "T2_cell_metrics.csv", rows)
    written.append("T2_cell_metrics.csv")

    # T3.
    rows = []
    for key in _arm_order(k for k in cells_ if k[0] == "G"):
        g, h = cells_[key], cells_.get(("H",) + key[1:])
        row = {"workload": key[1], "n": key[2], "runtime": key[3]}
        for m in BAND_METRICS:
            row[f"{m} G/H"] = None if h is None else _ratio(g, h, m)
            row[f"{m} band"] = bands.get(m)
            row[f"{m} distinguishable"] = distinguishable(g, h, m, bands.get(m))
        rows.append(row)
    _write_csv(out / "T3_G_vs_H.csv", rows)
    md += ["## T3, G against H", "", f"Band (A/A, H′ against H): {json.dumps({m: bands.get(m) for m in BAND_METRICS})}",
           "", _markdown(rows), ""]
    written.append("T3_G_vs_H.csv")

    # T4.
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for a in ledger:
        counts[(a["experiment"], a["cell"]["arm"], a["cell"]["runtime"], a["outcome"])] += 1
        for flag in a.get("flags", []):
            counts[(a["experiment"], a["cell"]["arm"], a["cell"]["runtime"], flag)] += 1
    rows = [{"experiment": k[0], "arm": k[1], "runtime": k[2], "outcome_or_flag": k[3], "count": v}
            for k, v in sorted(counts.items())]
    _write_csv(out / "T4_outcomes.csv", rows)
    md += ["## T4, outcomes (every attempt in the ledger)", "", _markdown(rows), ""]
    written.append("T4_outcomes.csv")

    # T5.
    rows = [{"feature": f, "H (Frames2Py Engine)": h, "G (reference swap)": g} for f, h, g in CONTRACT_FEATURES]
    _write_csv(out / "T5_contract_features.csv", rows)
    md += ["## T5, contract features", "", _markdown(rows), ""]
    written.append("T5_contract_features.csv")

    # T6.
    rows = [{"run_id": a["run_id"], "session": a.get("session"), "outcome": a["outcome"],
             "reasons": "; ".join(a.get("reasons", [])), "load_average": a.get("pre", {}).get("load_average"),
             "power_source_start": a.get("pre", {}).get("power", {}).get("source"),
             "thermal_ok_end": a.get("post", {}).get("thermal", {}).get("ok")} for a in ledger]
    _write_csv(out / "T6_environment.csv", rows)
    written.append("T6_environment.csv")

    _write_csv(out / "runs.csv", run_rows)
    written.append("runs.csv")
    (out / "series.json").write_text(json.dumps(series) + "\n")
    written.append("series.json")
    written += write_figure_data(out, cells_, p2, bands)

    md += ["## Hypotheses", ""]
    for name, v in verdicts.items():
        md.append(f"- **{name}**: {v['verdict']} (support condition {v['supported_condition']}, "
                  f"contradiction condition {v['contradicted_condition']})")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    written.append("tables.md")
    return written


CONTRACT_FEATURES: Final = (
    ("read-only published frames", "yes (`Snapshot.frame` is marked read-only)", "no"),
    ("producer-thread ownership enforcement", "yes (`ingest()` from another thread raises)", "no"),
    ("start, stop and reset semantics; stop publishes the pending window", "yes", "no"),
    ("the sequence continues across reset()", "yes", "no"),
    ("EngineStats", "yes", "no"),
    ("fail-closed free-threaded runtime check", "yes (decisions.md 42)", "no"),
    ("documented basis for the handoff's memory ordering", "yes (decisions.md 6)", "a threading.Lock"),
)


def _markdown(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


FIGURES: Final = {
    "F1": ("achieved_ratio", "N"), "F2": ("step_p99_us", "N"), "F3": ("freshness_p50_ms", "N"),
    "F4": ("freshness_p95_ms", "processing_p50_ms"), "F7": ("observation_rate", "processing_p50_ms"),
    "F8": ("skipped", "N"), "F9": ("process_cpu_cores", "N"), "F10": ("step_p99_us", "runtime"),
}


def write_figure_data(out: Path, cells_: dict[Key, dict[str, Any]], p2: dict[Key, dict[str, Any]],
                      bands: dict[str, Any]) -> list[str]:
    """The data each figure of 18.2 would plot. No figure is drawn: no plotting dependency."""
    rows = []
    wanted = ("achieved_ratio", "step_p99_us", "freshness_p50_ms", "freshness_p95_ms", "processing_p50_ms",
              "observation_rate", "skipped", "declared_drops", "unobserved_fraction", "process_cpu_cores",
              "producer_cpu_cores", "consumer_cpu_total_cores", "p_core_share", "queue_depth_max",
              "queue_depth_slope", "release_lateness_p50_ms", "lag_median_ms", "busy_ns_per_event")
    for key in _arm_order({**cells_, **p2}):
        cell = cells_.get(key) or p2[key]
        rows.append({"arm": key[0], "workload": key[1], "n": key[2], "runtime": key[3],
                     **{f"{m}_median": median_of(cell, m) for m in wanted},
                     **{f"{m}_min": (cell[m]["min"] if m in cell else None) for m in wanted},
                     **{f"{m}_max": (cell[m]["max"] if m in cell else None) for m in wanted}})
    _write_csv(out / "figure_data_cells.csv", rows)
    return ["figure_data_cells.csv"]

