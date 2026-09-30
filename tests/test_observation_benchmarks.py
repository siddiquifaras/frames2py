"""The observation study's harness (``benchmarks/observation*.py``): the queue policies, the
source, watermark resolution, queue-depth reconstruction, the SUSTAINED and outcome
classifiers, the integrity accounting of every arm, the V1 equivalence check and the driver's
parsers and pass planning. No timing assertions: runs are tiny, and nothing here is a
measurement.
"""

from __future__ import annotations

import functools
import json
import math
import queue
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmarks import observation as ob
from benchmarks import observation_analysis as an
from benchmarks import observation_driver as drv
from benchmarks import observation_validation as val
from benchmarks import power
from tests.adapters.backends import require_backend
from tests.fake_power import FakeMac

TINY = ob.Condition(sensor_size=(64, 48), batch_size=5_000, rate_hz=1_000_000, warmup_s=0.2, window_s=0.4,
                    pool_events=320_000)
"""5 ms batches of one event per µs, 16 ms cadence, a 0.6 s stream."""


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> FakeMac:
    backend = FakeMac(0x1F)
    monkeypatch.setattr(power, "hold_awake", functools.partial(power.hold_awake, backend=backend, platform="darwin"))
    return backend


def run_tiny(tmp_path: Path, arm: str, workload: str, n: int, condition: ob.Condition = TINY) -> dict[str, Any]:
    request = {"run_id": f"{ob.arm_slug(arm)}_{workload}_{n}", "experiment": "test", "arm": arm,
               "workload": workload, "n": n, "runtime": "A", "condition": condition.to_record(),
               "out_dir": str(tmp_path), "tmp_dir": str(tmp_path / "tmp"), "check_runtime": False}
    record = ob.run(request)
    (tmp_path / f"{request['run_id']}.json").write_text(json.dumps(record, default=str))
    return record


# ---------------------------------------------------------------- PolicyQueue


operations = st.lists(st.one_of(st.tuples(st.just("put"), st.integers(0, 10**6)), st.just(("get", 0))),
                      max_size=60)


@given(operations, st.sampled_from([None, 1, 3]))
def test_block_and_unbounded_queues_behave_like_the_standard_queue(ops: list[tuple[str, int]],
                                                                   capacity: int | None) -> None:
    """FIFO order and no loss, against ``queue.Queue``, for puts that fit and gets that find."""
    policy = ob.Policy.UNBOUNDED if capacity is None else ob.Policy.BLOCK
    ours: ob.PolicyQueue[int] = ob.PolicyQueue(capacity, policy, threading.Event())
    theirs: queue.Queue[int] = queue.Queue(capacity or 0)
    for op, value in ops:
        if op == "put" and not theirs.full():
            ours.put(value, value)
            theirs.put(value)
        elif op == "get" and not theirs.empty():
            got = ours.get_nowait()
            assert got is not None and got[0] == theirs.get_nowait()
    assert ours.remaining() == list(theirs.queue)


def test_a_blocking_queue_hands_every_item_over_in_order_between_threads() -> None:
    stop = threading.Event()
    q: ob.PolicyQueue[int] = ob.PolicyQueue(2, ob.Policy.BLOCK, stop)
    received: list[int] = []

    def consume() -> None:
        while len(received) < 500:
            got = q.get(1.0)
            if got is not None:
                received.append(got[0])

    t = threading.Thread(target=consume)
    t.start()
    for key in range(500):
        assert q.put(key, key)
    t.join(30)
    assert received == list(range(500))
    assert q.abandoned == [] and q.evicted == [] and q.rejected == []


@given(st.integers(1, 6), st.integers(0, 30))
def test_drop_policies_discard_exactly_the_overflow(capacity: int, puts: int) -> None:
    oldest: ob.PolicyQueue[int] = ob.PolicyQueue(capacity, ob.Policy.DROP_OLDEST, threading.Event())
    newest: ob.PolicyQueue[int] = ob.PolicyQueue(capacity, ob.Policy.DROP_NEWEST, threading.Event())
    for key in range(puts):
        oldest.put(key, key)
        newest.put(key, key)
    overflow = max(0, puts - capacity)
    assert [e for e, _ in oldest.evicted] == list(range(overflow))
    assert oldest.remaining() == list(range(overflow, puts))
    assert newest.rejected == list(range(capacity, puts))
    assert newest.remaining() == list(range(min(capacity, puts)))


def test_a_blocked_put_is_abandoned_when_stop_is_set_and_counted_as_backlog() -> None:
    stop = threading.Event()
    q: ob.PolicyQueue[int] = ob.PolicyQueue(1, ob.Policy.BLOCK, stop)
    assert q.put(1, 1)
    threading.Timer(0.2, stop.set).start()
    assert q.put(2, 2) is False
    assert q.abandoned == [2]
    assert q.offered == q.enqueued + len(q.abandoned)


# ---------------------------------------------------------------- the source and the schedule


def test_the_stream_is_in_timestamp_order_across_pool_wraps() -> None:
    source = ob.Source(TINY)
    previous_max = -1
    for k in range(3 * source.pool_batches):
        batch = source.materialise(k)
        assert int(batch["t"].min()) > previous_max or (k == 0 and int(batch["t"].min()) == 0)
        assert int(batch["t"].max()) == source.max_t(k)
        previous_max = int(batch["t"].max())
    assert source.pool[0]["t"].min() == 0  # the pool itself is untouched


def test_the_schedule_releases_batch_k_at_its_last_event_time() -> None:
    source = ob.Source(TINY)
    offsets, maxima = source.schedule()
    period = TINY.batch_size * 10**9 // TINY.rate_hz
    assert offsets == [(k + 1) * period for k in range(len(offsets))]
    assert offsets[-1] <= TINY.stream_ns < offsets[-1] + period
    assert maxima == [source.max_t(k) for k in range(len(offsets))]


def test_the_preregistered_capacities() -> None:
    assert ob.PREREGISTERED.k_raw == 13 and ob.PREREGISTERED.k_ff == 4
    assert ob.PREREGISTERED.stream_ns == 15 * 10**9 and ob.PREREGISTERED.warmup_ns == 5 * 10**9


# ---------------------------------------------------------------- cells and passes


def test_the_experiments_have_the_preregistered_cells() -> None:
    p1, p2 = ob.p1_cells(), ob.p2_cells()
    assert len(p1) == 118 and len(set(c.id for c in p1)) == 118
    assert len(p2) == 12 and len(set(c.id for c in p2)) == 12
    assert Counter(c.arm for c in p1 if c.n == 0) == {arm: 2 for arm in ("A", "F", "G", "H", "H'")}
    assert {c.arm for c in p1} == set(ob.P1_ARMS)


def test_each_pass_is_a_reproducible_permutation_of_its_experiment() -> None:
    for experiment in ("P1", "P2"):
        orders = [ob.pass_order(experiment, p) for p in range(1, 6)]
        assert all(sorted(o, key=lambda c: c.id) == sorted(ob.cells(experiment), key=lambda c: c.id) for o in orders)
        assert orders[0] == ob.pass_order(experiment, 1)
        assert len({tuple(c.id for c in o) for o in orders}) == 5


def test_an_interrupted_pass_resumes_after_its_last_completed_run() -> None:
    order = ob.pass_order("P2", 1)

    def entry(cell: ob.Cell, outcome: str, at: str) -> dict[str, Any]:
        return {"experiment": "P2", "pass": 1, "revision": 0, "cell": cell.to_record(), "outcome": outcome,
                "ended_at": at}

    entries = [entry(order[0], "VALID", "1"), entry(order[1], "INVALID_ENV", "2"), entry(order[2], "VALID", "3")]
    todo, attempts = drv.pass_plan("P2", 1, 0, entries)
    assert [c.id for c in todo] == [c.id for c in order[3:]] + [order[1].id]
    assert attempts[order[1].id] == 1
    exhausted = entries + [entry(order[1], "INVALID_ENV", "4"), entry(order[1], "INVALID_ENV", "5")]
    todo, _ = drv.pass_plan("P2", 1, 0, exhausted)
    assert order[1].id not in [c.id for c in todo]
    other_revision = drv.pass_plan("P2", 1, 1, entries)[0]
    assert [c.id for c in other_revision] == [c.id for c in order]


def test_a_pass_requeues_environment_failures_at_its_end_at_most_twice() -> None:
    script = {("a", 1): "VALID", ("b", 1): "INVALID_ENV", ("c", 1): "VALID", ("b", 2): "INVALID_ENV",
              ("b", 3): "INVALID_ENV", ("d", 1): "INVALID_ENV", ("d", 2): "VALID"}
    calls: list[tuple[str, int]] = []

    def run(key: str, n: int) -> str:
        calls.append((key, n))
        return script[(key, n)]

    final = drv.run_pass_with_retries(["a", "b", "c", "d"], run)
    assert calls == [("a", 1), ("b", 1), ("c", 1), ("d", 1), ("b", 2), ("d", 2), ("b", 3)]
    assert final == {"a": "VALID", "b": "INVALID_ENV", "c": "VALID", "d": "VALID"}
    stopped = drv.run_pass_with_retries(["a", "b"], lambda key, n: "INVALID_INTEGRITY")
    assert stopped == {"a": "INVALID_INTEGRITY"}


def test_an_attempt_that_finished_after_its_driver_stopped_is_recorded_once(tmp_path: Path) -> None:
    cell = ob.pass_order("P1", 1)[0]
    run_id = f"{cell.id}_r0_p1_a2"
    (tmp_path / "runs").mkdir()
    request = {**cell.to_record(), "run_id": run_id, "out_dir": str(tmp_path / "runs")}
    (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({**good_record(), "request": request}))
    ledger = drv.Ledger(tmp_path)
    ledger.append({"run_id": f"{cell.id}_r0_p1_a1", "experiment": "P1", "pass": 1, "revision": 0,
                   "cell": cell.to_record(), "outcome": "INVALID_ENV", "ended_at": "1"})
    added = drv.reconcile(tmp_path, ledger, "s")
    assert [(e["run_id"], e["outcome"], e["attempt"]) for e in added] == [(run_id, "INVALID_ENV", 2)]
    assert drv.reconcile(tmp_path, ledger, "s") == []
    todo, attempts = drv.pass_plan("P1", 1, 0, ledger.entries())
    assert attempts[cell.id] == 2 and todo[-1].id == cell.id


def test_an_interrupted_attempt_keeps_an_integrity_finding(tmp_path: Path) -> None:
    cell = ob.pass_order("P1", 1)[0]
    run_id = f"{cell.id}_r0_p1_a1"
    (tmp_path / "runs").mkdir()
    record = {**good_record(integrity={"harness": [], "problems": ["lost"]}),
              "request": {**cell.to_record(), "run_id": run_id, "out_dir": str(tmp_path / "runs")}}
    (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps(record))
    assert [e["outcome"] for e in drv.reconcile(tmp_path, drv.Ledger(tmp_path), "s")] == ["INVALID_INTEGRITY"]


# ---------------------------------------------------------------- watermark resolution (9.2)


@given(st.lists(st.integers(0, 5), min_size=1, max_size=30), st.integers(0, 5), st.integers(0, 40))
def test_a_watermark_resolves_to_the_first_matching_batch_already_stepped(m: list[int], w: int, o: int) -> None:
    s = list(range(1, len(m) + 1))  # S_k = k + 1
    expected = next((k for k in range(len(m)) if m[k] == w and s[k] < o), None)
    assert ob.resolver(m, s)(w, o) == expected


# ---------------------------------------------------------------- queue depth (10.2)


@given(st.lists(st.integers(0, 1000), max_size=40), st.lists(st.integers(0, 1000), max_size=40))
def test_queue_depth_counts_what_was_enqueued_and_not_yet_dequeued(enq: list[int], deq: list[int]) -> None:
    t0, t1 = 0, 1000
    depth = an._queue_depth([t * 1_000_000 for t in enq], [t * 1_000_000 for t in deq], t0, t1 * 1_000_000)
    grid = range(0, t1, 100)
    expected = [sum(e <= g for e in enq) - sum(d <= g for d in deq) for g in grid]
    assert depth["max"] == max(expected)


# ---------------------------------------------------------------- SUSTAINED (15.2)


def lag_record(lags_ms: list[float], unreleased: int = 0, condition: ob.Condition = ob.PREREGISTERED
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """A run whose batch k is released on schedule and finishes ``lags_ms[k]`` after A_k."""
    period = round(condition.batch_period_ns)
    total = len(lags_ms) + unreleased
    offsets = np.array([(k + 1) * period for k in range(total)], dtype=np.int64)
    released = len(lags_ms)
    e = offsets[:released] + (np.array(lags_ms) * 1e6).astype(np.int64)
    s = offsets[:released] + 1
    zeros = np.zeros(released, dtype=np.int64)
    record = {"request": {"arm": "H", "condition": condition.to_record()}, "t_start": 0,
              "t0": condition.warmup_ns, "t1": condition.stream_ns, "batches": released, "memory_ceiling": False}
    arrays = {"schedule_offset": offsets, "producer_q": offsets[:released], "producer_s": s, "producer_e": e,
              "producer_cq": zeros, "producer_cs": zeros, "producer_ce": zeros, "producer_p": zeros}
    return record, arrays


def test_the_preregistered_sustained_thresholds() -> None:
    assert an.sustained_thresholds(ob.PREREGISTERED) == (1.6, 16.0)


@pytest.mark.parametrize(("slope_ms_per_s", "offset_ms", "expected"), [
    (0.0, 1.0, True),      # flat and small
    (1.5, 1.0, True),      # growing, below the slope limit; final median 15.25 ms
    (1.7, -10.0, False),   # the slope alone is over the limit; final median 6.15 ms
    (0.0, 17.0, False),    # flat, but the final median is over one interval
    (-2.0, 5.0, True),     # shrinking
])
def test_sustained_follows_the_lag_slope_and_the_final_median(slope_ms_per_s: float, offset_ms: float,
                                                              expected: bool) -> None:
    c = ob.PREREGISTERED
    times = [(k + 1) * c.batch_period_ns / 1e9 for k in range(round(c.stream_ns / c.batch_period_ns))]
    record, arrays = lag_record([offset_ms + slope_ms_per_s * (t - 5.0) for t in times])
    metrics = an.producer_metrics(record, arrays)
    assert math.isclose(metrics["lag_slope_ms_per_s"], slope_ms_per_s, abs_tol=0.01)
    assert metrics["sustained"] is expected


def test_batches_never_released_in_the_window_mean_not_sustained() -> None:
    c = ob.PREREGISTERED
    released = round((c.stream_ns - 2 * 10**9) / c.batch_period_ns)
    record, arrays = lag_record([1.0] * released, unreleased=400)
    metrics = an.producer_metrics(record, arrays)
    assert metrics["lag_last_second_median_ms"] == math.inf
    assert metrics["sustained"] is False
    assert metrics["source_backlog_batches"] == 400


def test_the_memory_ceiling_means_not_sustained() -> None:
    c = ob.PREREGISTERED
    record, arrays = lag_record([1.0] * round(c.stream_ns / c.batch_period_ns))
    record["memory_ceiling"] = True
    assert an.producer_metrics(record, arrays)["sustained"] is False


# ---------------------------------------------------------------- the outcome classifier (14.1)


CLEAN = {"power": {"ac": True, "low_power_mode": False}, "thermal": {"ok": True},
         "swap": {"swapouts": 7}}


def good_record(**changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"errors": [], "alive_after_grace": [], "integrity": {"harness": [], "problems": []},
                              "power": {"slept": False, "end": {"full_wake": True}}}
    record.update(changes)
    return record


def classify(record: dict[str, Any] | None, *, rc: int | None = 0, pre: dict[str, Any] = CLEAN,
             post: dict[str, Any] = CLEAN, load: dict[str, Any] | None = None, sustained: bool | None = True,
             gate_env: bool = True, timed_out: bool = False) -> tuple[str, list[str], list[str]]:
    return drv.classify(rc, record, pre, post, load or {"violations": []}, sustained, gate_env=gate_env,
                        timed_out=timed_out)


def test_outcome_classes() -> None:
    assert classify(good_record())[0:2] == ("VALID", [])
    assert classify(None)[0] == "HARNESS_FAILURE"
    assert classify(None, rc=-11)[0] == "CRASH"
    assert classify(None, rc=-9, timed_out=True)[0] == "HARNESS_FAILURE"
    assert classify(good_record(refused=["wrong numpy"]))[0] == "REFUSED"
    assert classify(good_record(fatal="Traceback\nValueError: x"))[0] == "HARNESS_FAILURE"
    assert classify(good_record(errors=[{"thread": "consumer 0", "type": "KeyError"}]))[0] == "HARNESS_FAILURE"
    assert classify(good_record(alive_after_grace=["consumer-0"]))[0] == "SHUTDOWN_TIMEOUT"
    assert classify(good_record(integrity={"harness": ["x"], "problems": []}))[0] == "HARNESS_FAILURE"
    assert classify(good_record(integrity={"harness": [], "problems": ["lost"]}))[0] == "INVALID_INTEGRITY"
    assert classify(good_record(readback={"ok": False}))[0] == "INVALID_INTEGRITY"
    assert classify(good_record(power_refused="DarkWake"))[0] == "INVALID_ENV"


def test_expected_losses_and_limits_are_flags_on_valid_runs() -> None:
    outcome, flags, _ = classify(good_record(errors=[{"thread": "consumer 0", "type": "MemoryError"}],
                                             memory_ceiling=True, drain_timeout=True), sustained=False)
    assert outcome == "VALID"
    assert set(flags) == {"ARM_MEMORY_ERROR", "MEMORY_CEILING", "DRAIN_TIMEOUT", "NOT_SUSTAINED"}


@pytest.mark.parametrize("change", [
    {"record": {"power": {"slept": True, "end": {"full_wake": True}}}},
    {"record": {"power": {"slept": False, "end": {"full_wake": False}}}},
    {"pre": {**CLEAN, "power": {"ac": False, "low_power_mode": False}}},
    {"post": {**CLEAN, "power": {"ac": True, "low_power_mode": True}}},
    {"post": {**CLEAN, "thermal": {"ok": False}}},
    {"post": {**CLEAN, "swap": {"swapouts": 8}}},
    {"load": {"violations": ["x at 30%"]}},
])
def test_environment_problems_invalidate_unless_not_gating(change: dict[str, Any]) -> None:
    record = good_record(**change.get("record", {}))
    kwargs = {k: v for k, v in change.items() if k != "record"}
    assert classify(record, **kwargs)[0] == "INVALID_ENV"
    outcome, _, reasons = classify(record, gate_env=False, **kwargs)
    assert outcome == "VALID" and reasons and all(r.startswith("not gating") for r in reasons)


# ---------------------------------------------------------------- integrity accounting (21.4)


ARMS_AND_WORKLOADS = [("A", "none", 0), ("A", "W1", 2), ("B", "W1", 2), ("B", "W3", 1), ("C", "W3", 2),
                      ("E", "W3", 1), ("F", "none", 0), ("F", "W1", 2), ("G", "W3", 2), ("H", "W1", 2),
                      ("H'", "none", 0), ("RB", "W1", 2), ("RB", "W3", 1)]


@pytest.mark.parametrize(("arm", "workload", "n"), ARMS_AND_WORKLOADS)
def test_every_arm_accounts_for_every_batch_publication_and_item(tmp_path: Path, mac: FakeMac, arm: str,
                                                                 workload: str, n: int) -> None:
    record = run_tiny(tmp_path, arm, workload, n)
    assert record["errors"] == [] and record["alive_after_grace"] == []
    assert record["integrity"] == {"harness": [], "problems": []}
    assert record["batches"] > 0
    metrics = an.run_metrics(*an.load(tmp_path / f"{record['run_id']}.json"))
    if n:
        assert metrics["freshness_ms"]["n"] > 0 and metrics["freshness_ms"]["p50"] >= 0
    if arm in ("A", "B", "E") and n:
        assert metrics["coverage"] == 1.0
    assert mac.held == set()


@pytest.mark.parametrize(("arm", "workload", "n"), [("A", "none", 0), ("G", "none", 0), ("H", "none", 0),
                                                   ("H", "W1", 1), ("B", "W1", 1), ("RB", "W1", 1)])
def test_the_accounting_holds_with_aggregate_only_instrumentation(tmp_path: Path, mac: FakeMac, arm: str,
                                                                   workload: str, n: int) -> None:
    request = {"run_id": f"agg_{ob.arm_slug(arm)}_{n}", "experiment": "test", "arm": arm, "workload": workload,
               "n": n, "runtime": "A", "condition": TINY.to_record(), "out_dir": str(tmp_path),
               "tmp_dir": str(tmp_path / "tmp"), "check_runtime": False, "instrumentation": "aggregate"}
    record = ob.run(request)
    assert record["errors"] == [] and record["alive_after_grace"] == []
    assert record["integrity"] == {"harness": [], "problems": []}


@pytest.mark.parametrize("n_so", [0, 1])
@pytest.mark.parametrize("arm", ["EP-H", "EP-QB", "EP-QE"])
def test_the_preservation_arms_record_every_batch_fed(tmp_path: Path, mac: FakeMac, arm: str, n_so: int) -> None:
    require_backend("h5py", "hdf5plugin")
    record = run_tiny(tmp_path, arm, "W1" if n_so else "none", n_so)
    assert record["integrity"] == {"harness": [], "problems": []}
    assert record["readback"]["ok"] and record["readback"]["events"] == record["batches"] * TINY.batch_size
    assert not (tmp_path / "tmp" / f"{record['run_id']}.h5").exists()


def test_the_accounting_catches_a_lost_item_and_an_unresolvable_watermark(tmp_path: Path, mac: FakeMac) -> None:
    record = run_tiny(tmp_path, "B", "W1", 1)
    _, arrays = an.load(tmp_path / f"{record['run_id']}.json")
    source = ob.Source(TINY)
    lost = {**record, "queues": [{**record["queues"][0], "dequeued": record["queues"][0]["dequeued"] - 1}]}
    assert ob.integrity("B", 1, TINY, source, lost, arrays, True)["problems"]
    wrong = dict(arrays)
    wrong["c0_wm"] = arrays["c0_wm"] + 1
    assert any("resolve" in p for p in ob.integrity("B", 1, TINY, source, record, wrong, True)["problems"])


# ---------------------------------------------------------------- V1


SMALL_V1 = ob.Condition(sensor_size=(64, 48), batch_size=5_000, rate_hz=1_000_000, pool_events=320_000)


def test_v1_finds_the_arms_equivalent_to_the_engine_and_to_the_reference() -> None:
    result = val.v1(SMALL_V1, batches=60)
    assert result["ok"], result
    assert result["intervals"]["16.0"]["publications"] == 15
    assert result["intervals"]["0.0"]["publications"] == 60


def test_v1_detects_a_baseline_that_publishes_differently(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_window_closure(self: ob.Publisher) -> ob.Item:
        self.sequence += 1
        watermark = self.acc.watermark
        assert watermark is not None
        return self.acc.read(), watermark, self.sequence

    monkeypatch.setattr(ob.Publisher, "publish", no_window_closure)
    result = val.v1(SMALL_V1, batches=20)
    assert not result["ok"]
    assert any(p.startswith("A:") for p in result["intervals"]["16.0"]["problems"])


# ---------------------------------------------------------------- the driver's parsers and load check


def test_power_thermal_and_swap_parsers() -> None:
    batt = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true"
    assert drv.parse_batt(batt) == {"source": "AC Power", "ac": True, "percent": 100, "state": "charged"}
    assert drv.parse_batt("Now drawing from 'Battery Power'\n -x\t80%; discharging; 3:00")["ac"] is False
    assert drv.parse_low_power(" lowpowermode         0\n sleep 1") is False
    assert drv.parse_low_power(" lowpowermode         1") is True
    assert drv.parse_low_power("") is None
    clear = "\n".join(f"Note: {n}" for n in drv.THERMAL_CLEAR)
    assert drv.parse_thermal(clear)["ok"]
    assert not drv.parse_thermal(clear.replace("No thermal warning level has been recorded", "Thermal Warning: 1"))["ok"]
    assert drv.parse_swapouts("Pageouts: 3.\nSwapouts:   12.\n") == 12


@pytest.mark.parametrize(("comm", "expected"), [
    ("/Library/PrivilegedHelperTools/com.docker.vmnetd", "Docker"),
    ("/Applications/Docker.app/Contents/MacOS/com.docker.backend", "Docker"),
    ("/usr/local/bin/qemu-system-aarch64", "qemu"),
    ("/opt/homebrew/bin/codex", "codex"),
    ("/Applications/Cursor.app/Contents/MacOS/Cursor", "Cursor"),
    ("/Applications/Visual Studio Code.app/Contents/Frameworks/Code Helper.app/Contents/MacOS/Code Helper",
     "VS Code"),
    ("/System/Library/CoreServices/CursorUIViewService.app/Contents/MacOS/CursorUIViewService", None),
    ("claude", None),
    ("/usr/libexec/syspolicyd", None),
])
def test_the_denylist(comm: str, expected: str | None) -> None:
    assert drv.denylisted(comm) == expected


def ps(*rows: tuple[int, int, float, str]) -> str:
    return "\n".join(f"{pid} {ppid} {pcpu} 1000 {comm}" for pid, ppid, pcpu, comm in rows)


def test_background_load_invalidates_only_when_it_persists_for_two_samples() -> None:
    monitor = drv.LoadMonitor(child_pid=100)
    driver = monitor.driver_pid
    monitor.observe(ps((100, driver, 300.0, "python"), (7, 1, 12.0, "mds"), (driver + 1, driver, 50.0, "ps")))
    monitor.observe(ps((100, driver, 300.0, "python"), (8, 1, 12.0, "mdworker")))
    assert monitor.violations == []
    monitor.observe(ps((8, 1, 11.0, "mdworker")))
    assert len(monitor.violations) == 1 and "mdworker" in monitor.violations[0]
    total = drv.LoadMonitor(child_pid=100)
    for _ in range(2):
        total.observe(ps(*[(10 + i, 1, 9.0, f"p{i}") for i in range(3)]))
    assert any("together" in v for v in total.violations)


def test_the_driver_lineage_excludes_its_own_family_only() -> None:
    parents = {10: 1, 20: 10, 30: 20, 40: 30, 41: 30, 50: 10, 60: 1}
    assert drv.lineage(30, parents) == {10, 20, 30, 40, 41}


# ---------------------------------------------------------------- analysis


def cell(values: list[float]) -> dict[str, Any]:
    return {"m": {"median": float(np.median(values)), "min": min(values), "max": max(values)}}


def test_distinguishable_needs_both_the_band_and_disjoint_ranges() -> None:
    assert an.distinguishable(cell([10, 11, 12]), cell([20, 21, 22]), "m", 1.2) is True
    assert an.distinguishable(cell([10, 11, 12]), cell([12.5, 13, 14]), "m", 1.2) is False  # inside the band
    assert an.distinguishable(cell([10, 11, 30]), cell([20, 21, 22]), "m", 1.2) is False  # ranges overlap
    assert an.distinguishable(cell([10, 11, 12]), cell([20, 21, 22]), "m", None) is None


def test_the_band_is_the_largest_a_a_ratio_without_a_floor() -> None:
    def full(v: float) -> dict[str, Any]:
        return {"n_valid": 5, **{m: {"median": v, "min": v, "max": v} for m in an.BAND_METRICS}}

    cells = {("H", "W1", 1, "A"): full(100.0), ("H'", "W1", 1, "A"): full(102.0),
             ("H", "W3", 1, "A"): full(50.0), ("H'", "W3", 1, "A"): full(49.0)}
    band = an.band(cells)
    assert band["step_p99_us"] == pytest.approx(50 / 49)
    cells[("H'", "W3", 1, "A")]["n_valid"] = 4
    assert an.band(cells)["excluded"] == [["H", "W3", 1, "A"]]


@given(st.lists(st.integers(-5, 5), min_size=3, max_size=20))
def test_ranks_average_ties(values: list[int]) -> None:
    ranked = an.ranks(np.array(values))
    for i, v in enumerate(values):
        below = sum(x < v for x in values)
        equal = sum(x == v for x in values)
        assert ranked[i] == below + (equal + 1) / 2
