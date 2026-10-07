"""The ``wait_for_newer`` producer-cost harness (``benchmarks/wait*.py``): the preregistered
specs and order, the section 8 rule, outcome classification, resume planning, and both
experiments end to end on the installed build. No timing assertions: nothing here is a
measurement.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from benchmarks import wait
from benchmarks import wait_analysis as an
from benchmarks import wait_driver as drv
from tests.contract.helpers import SUPPORTED_RUNTIME


def test_specs_are_the_preregistered_set() -> None:
    specs = wait.specs()
    assert len(specs) == 144 and len({s.id for s in specs}) == 144
    per = Counter((s.runtime, s.experiment) for s in specs)
    assert per == {("A", "M1"): 12, ("A", "M2"): 54, ("A", "M3"): 6, ("B", "M1"): 12, ("B", "M2"): 54, ("B", "M3"): 6}
    m2 = {(s.width, s.height, s.batch_size, s.interval_ms) for s in specs if s.experiment == "M2"}
    assert m2 == {(w, h, n, i) for w, h in ((346, 260), (640, 480), (1280, 720))
                  for n, i in ((10_000, 0.0), (100_000, 0.0), (100_000, 16.0))}
    assert {(s.label, s.build, s.waiters) for s in specs if s.experiment != "M3"} == {
        ("B", "baseline", 0), ("B'", "baseline", 0), ("F0", "feature", 0),
        ("F1", "feature", 1), ("F4", "feature", 4), ("F8", "feature", 8)}
    m3 = {(s.label, s.build, s.waiters, s.width, s.height, s.batch_size, s.interval_ms)
          for s in specs if s.experiment == "M3"}
    assert m3 == {(f"{kind}{w}", "feature", w, 1280, 720, 100_000, 16.0) for kind in ("WAIT", "POLL") for w in (1, 4, 8)}


def test_each_pass_is_a_fixed_permutation() -> None:
    orders = [wait.pass_order(p) for p in range(1, 6)]
    assert all(sorted(s.id for s in o) == sorted(s.id for s in wait.specs()) for o in orders)
    assert [s.id for s in orders[0]] == [s.id for s in wait.pass_order(1)]
    assert len({tuple(s.id for s in o) for o in orders}) == 5
    with pytest.raises(ValueError):
        wait.pass_order(6)


# ------------------------------------------------------------------ the rule


def test_compare_needs_both_conditions() -> None:
    inside = an.compare([1.05] * 5, [1.0] * 5, band=1.10)
    assert not inside["outside_band"] and inside["ranges_disjoint"] and not inside["distinguishable"]
    overlapping = an.compare([1.2, 1.2, 1.2, 1.2, 0.99], [1.0] * 5, band=1.10)
    assert overlapping["outside_band"] and not overlapping["ranges_disjoint"]
    assert not overlapping["distinguishable"]
    both = an.compare([1.2] * 5, [1.0] * 5, band=1.10)
    assert both["distinguishable"] and both["ratio"] == pytest.approx(1.2)


@pytest.mark.parametrize(("direction", "ratio", "slower"),
                         [("lower", 1.2, True), ("lower", 0.8, False), ("higher", 0.8, True), ("higher", 1.2, False)])
def test_slower_follows_the_metric_direction(direction: str, ratio: float, slower: bool) -> None:
    assert an.is_slower(direction, ratio) is slower


def _records(value: Any) -> dict[str, list[dict[str, Any]]]:
    """Five runs per spec; *value(spec, run)* gives every metric's value."""
    out = {}
    for s in wait.specs():
        if s.experiment == "M3":
            continue
        metrics = an.METRICS[s.experiment]
        out[s.id] = [{m: value(s, k) for m in metrics} for k in range(5)]
    return out


def _flat(spec: wait.Spec, k: int) -> float:
    # B' differs from B by up to 2% in one cell; everything else is B's value plus a run spread.
    base = 100.0 + k
    if spec.label == "B'" and spec.width == 640:
        return base * 1.02
    return base


def test_bands_are_the_largest_aa_ratio() -> None:
    summary = an.analyse(_records(_flat))
    assert summary["bands"]["M2 events_per_s"]["band"] == pytest.approx(1.02)
    assert summary["bands"]["M2 events_per_s"]["aa_cells"] == 18
    assert summary["bands"]["M1 median_ns"]["band"] == pytest.approx(1.0)
    assert summary["bands"]["M1 median_ns"]["aa_cells"] == 4


def test_no_distinguishable_cell_meets_the_criterion() -> None:
    summary = an.analyse(_records(_flat))
    assert summary["verdict"] == "MET" and summary["compared"] == summary["expected"] == 2 * 9 * 4
    assert len(summary["w0_m1"]) == 2 * 2 * 3


def test_m1_at_no_waiters_is_described_not_judged() -> None:
    def slower_m1(spec: wait.Spec, k: int) -> float:
        value = _flat(spec, k)
        return value * 1.5 if spec.label == "F0" and spec.experiment == "M1" else value

    summary = an.analyse(_records(slower_m1))
    assert summary["verdict"] == "MET" and not summary["distinguishably_slower"]
    assert all(r["distinguishable"] and r["direction"] == "slower" for r in summary["w0_m1"])


def test_a_slower_cell_fails_and_a_faster_one_is_only_reported() -> None:
    def slower(spec: wait.Spec, k: int) -> float:
        value = _flat(spec, k)
        if spec.label == "F0" and spec.experiment == "M2" and spec.width == 1280 and spec.runtime == "B":
            return value * 1.5  # a time metric 1.5x: slower; events_per_s 1.5x: faster
        return value

    summary = an.analyse(_records(slower))
    assert summary["verdict"] == "NOT MET"
    assert {r["metric"] for r in summary["distinguishably_slower"]} == {"p50", "p95", "p99"}
    assert {r["metric"] for r in summary["distinguishably_faster"]} == {"events_per_s"}


def test_a_cell_short_of_runs_leaves_the_criterion_not_established() -> None:
    records = _records(_flat)
    short = wait.Spec("M2", "A", 346, 260, 10_000, 0.0, "F0").id
    records[short] = records[short][:4]
    summary = an.analyse(records)
    assert summary["verdict"] == "NOT ESTABLISHED"
    assert {e["cell"] for e in summary["excluded"]} == {"3.11.14 346x260 10000 @ 0 ms"}


def test_characterisation_reports_the_per_waiter_increment() -> None:
    def waiters(spec: wait.Spec, k: int) -> float:
        return _flat(spec, k) + 10.0 * spec.waiters

    rows = an.analyse(_records(waiters))["characterisation"]
    m1 = [r for r in rows if r["experiment"] == "M1" and r["metric"] == "median_ns"]
    assert m1 and all(r["per_waiter_ns"] == pytest.approx(10.0) for r in m1)


# ------------------------------------------------------------------ the driver


def _ok_record(**result: Any) -> dict[str, Any]:
    return {"result": {"checks": {"valid": True}, "waiter_errors": [], "waiters_alive": [], **result},
            "power": {"slept": False, "end": {"full_wake": True}}}


STATE = {"power": {"ac": True, "low_power_mode": False}, "thermal": {"ok": True}, "swap": {"swapouts": 3}}
QUIET = {"violations": []}


@pytest.mark.parametrize(("record", "outcome"), [
    (None, "HARNESS_FAILURE"),
    ({"refused": ["NumPy 2.5.3, expected 2.4.6"]}, "REFUSED"),
    ({"fatal": "Traceback\nRuntimeError: boom"}, "HARNESS_FAILURE"),
    ({"power_refused": "not in full wake"}, "INVALID_ENV"),
    (_ok_record(waiter_errors=["Traceback\nTypeError: x"]), "INVALID_INTEGRITY"),
    (_ok_record(waiters_alive=["waiter-0"]), "SHUTDOWN_TIMEOUT"),
    (_ok_record(checks={"valid": False, "failures": ["nothing published"]}), "INVALID_INTEGRITY"),
    (_ok_record(), "VALID"),
])
def test_classify(record: Any, outcome: str) -> None:
    assert drv.classify(0, record, STATE, STATE, QUIET, timed_out=False)[0] == outcome


def test_a_crash_and_environment_problems() -> None:
    assert drv.classify(-11, None, STATE, STATE, QUIET, timed_out=False)[0] == "CRASH"
    battery = {**STATE, "power": {"ac": False, "low_power_mode": False}}
    assert drv.classify(0, _ok_record(), battery, STATE, QUIET, timed_out=False)[0] == "INVALID_ENV"
    assert drv.classify(0, _ok_record(), STATE, STATE, {"violations": ["x at 50%"]}, timed_out=False)[0] == "INVALID_ENV"
    outcome, reasons = drv.classify(0, _ok_record(), battery, STATE, QUIET, timed_out=False, gate_env=False)
    assert outcome == "VALID" and reasons and all(r.startswith("not gating") for r in reasons)


def test_a_pass_resumes_and_retries_environment_failures_last() -> None:
    order = wait.pass_order(2)
    entries = [{"spec": order[0].to_record(), "pass": 2, "outcome": "VALID", "ended_at": "1"},
               {"spec": order[1].to_record(), "pass": 2, "outcome": "INVALID_ENV", "ended_at": "2"},
               {"spec": order[2].to_record(), "pass": 2, "outcome": "INVALID_ENV", "ended_at": "3"},
               {"spec": order[2].to_record(), "pass": 2, "outcome": "INVALID_ENV", "ended_at": "4"},
               {"spec": order[2].to_record(), "pass": 2, "outcome": "INVALID_ENV", "ended_at": "5"},
               {"spec": order[3].to_record(), "pass": 1, "outcome": "VALID", "ended_at": "6"}]
    todo, attempts = drv.pass_plan(2, entries)
    assert [s.id for s in todo] == [s.id for s in order[3:]] + [order[1].id]
    assert attempts == {order[0].id: 1, order[1].id: 1, order[2].id: 3}


# ------------------------------------------------------------------ both experiments, end to end


requires_wait = pytest.mark.skipif(not SUPPORTED_RUNTIME, reason="the Engine refuses this free-threaded runtime")


@requires_wait
@pytest.mark.timeout(120)
@pytest.mark.parametrize("label", ["F0", "F4"])
def test_m1_runs_and_checks_itself(label: str) -> None:
    result = wait.measure_m1(wait.Spec("M1", "A", 346, 260, 1, 0.0, label))
    assert result["checks"] == {"valid": True, "failures": []}
    assert len(result["call_ns"]) == wait.M1_PUBLICATIONS
    assert result["waiters_alive"] == [] and result["waiter_errors"] == []


@requires_wait
@pytest.mark.timeout(120)
@pytest.mark.parametrize(("batch", "interval"), [(10_000, 0.0), (100_000, 16.0)])
def test_m2_runs_and_checks_itself(batch: int, interval: float) -> None:
    result = wait.measure_m2(wait.Spec("M2", "A", 346, 260, batch, interval, "F8"))
    assert result["checks"]["valid"], result["checks"]
    assert result["registered_at_start"] == 8
    assert result["waiters_alive"] == [] and result["waiter_errors"] == []


def test_m3_table_pairs_waiters_with_pollers() -> None:
    def run(step: float) -> dict[str, Any]:
        return {"step_p99_us": step, "busy_ns_per_event": 10.0, "freshness_ms": {"p50": 2.0, "p95": 4.0},
                "post_step_ms": {"p50": 1.0, "p95": 3.0}}

    records = {wait.Spec("M3", "A", 1280, 720, 100_000, 16.0, "WAIT4").id: [run(100.0 + k) for k in range(5)],
               wait.Spec("M3", "A", 1280, 720, 100_000, 16.0, "POLL4").id: [run(200.0 + k) for k in range(5)]}
    rows = {(r["runtime"], r["consumers"], r["metric"]): r for r in an.m3_table(records)}
    row = rows[("3.11.14", 4, "step_p99_us")]
    assert row["wait_median"] == 102.0 and row["poll_median"] == 202.0 and row["wait_range"] == [100.0, 104.0]
    assert row["wait_over_poll"] == pytest.approx(102 / 202)
    assert rows[("3.11.14", 1, "step_p99_us")]["valid"] == {"wait": 0, "poll": 0}


@requires_wait
@pytest.mark.timeout(120)
@pytest.mark.parametrize("label", ["WAIT4", "POLL4"])
def test_m3_runs_and_checks_itself(label: str) -> None:
    from benchmarks import observation as ob

    short = ob.Condition(warmup_s=0.3, window_s=1.0, pool_events=400_000)
    result = wait.measure_m3(wait.Spec("M3", "A", 1280, 720, 100_000, 16.0, label), None, short)
    assert result["checks"] == {"valid": True, "failures": []}
    assert result["consumer_kind"] == ("wait" if label.startswith("WAIT") else "poll")
    assert result["waiters_alive"] == [] and result["waiter_errors"] == []


def test_the_driver_does_not_flag_its_own_caffeinate_wrapper() -> None:
    # The tree `caffeinate -dimsu uv run ... -m benchmarks.wait_driver` makes on macOS: uv (100) runs the
    # driver (102) and caffeinate (101) as its children. Another benchmark under its own caffeinate is flagged.
    ps = "\n".join([
        "  90     1 -zsh",
        " 100    90 uv run --no-sync python -m benchmarks.wait_driver campaign --unattended",
        " 101   100 caffeinate -dimsu uv run --no-sync python -m benchmarks.wait_driver campaign --unattended",
        " 102   100 /repo/.venv/bin/python -m benchmarks.wait_driver campaign --unattended",
        " 103   102 /repo/env/bin/python -m benchmarks.wait worker",
        " 200     1 caffeinate -i python -m benchmarks run --suite gate",
        " 201   200 python -m benchmarks run --suite gate",
    ])
    assert drv.other_benchmarks(102, ps) == ["200 caffeinate -i python -m benchmarks run --suite gate",
                                              "201 python -m benchmarks run --suite gate"]


# ------------------------------------------------------------------ AC pause and revision 1 (amendments 5 and 6)


def _power(*ac: bool) -> Any:
    states = iter(ac)
    return lambda: {"ac": next(states), "source": "AC Power"}


def test_ac_present_means_no_wait() -> None:
    slept: list[float] = []
    state, waited = drv.wait_ac(_power(True), slept.append, wait_s=90, poll_s=30)
    assert state["ac"] and waited == 0 and slept == []


def test_ac_lost_then_returned_waits_until_it_is_back() -> None:
    slept: list[float] = []
    state, waited = drv.wait_ac(_power(False, False, False, True), slept.append, wait_s=1800, poll_s=30)
    assert state["ac"] and waited == 90 and slept == [30, 30, 30]


def test_ac_that_never_returns_ends_the_wait_at_its_bound() -> None:
    slept: list[float] = []
    state, waited = drv.wait_ac(lambda: {"ac": False}, slept.append, wait_s=90, poll_s=30)
    assert not state["ac"] and waited == 90 and slept == [30, 30, 30]


def _fake_campaign(monkeypatch: pytest.MonkeyPatch, ac: Any) -> list[str]:
    """Run campaign() with no workers: *ac()* gives each start check's (state, waited)."""
    made: list[str] = []

    def attempt(spec: wait.Spec, *, pass_index: int, attempt_index: int, session: str, out_dir: Any,
                gate_env: bool) -> dict[str, Any]:
        made.append(spec.id)
        return {"run_id": f"{spec.id}_p{pass_index}_a{attempt_index}", "spec": spec.to_record(), "pass": pass_index,
                "attempt": attempt_index, "session": session, "outcome": "VALID", "reasons": [], "ended_at": "t"}

    monkeypatch.setattr(drv, "session_checks", lambda unattended, gate_env: ({"session": "s"}, []))
    monkeypatch.setattr(drv, "attempt", attempt)
    monkeypatch.setattr(drv, "wait_ac", ac)
    monkeypatch.setattr(drv, "IDLE_GAP_S", 0.0)
    return made


def test_a_session_stops_cleanly_when_ac_does_not_return_and_resumes_where_it_stopped(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = iter([({"ac": True}, 0.0)] * 10 + [({"ac": False}, drv.AC_WAIT_S)])
    first = _fake_campaign(monkeypatch, lambda: next(calls))
    assert drv.campaign([1], directory=tmp_path) == 4
    entries = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    order = wait.pass_order(1)
    assert [e["spec"] for e in entries] == [s.to_record() for s in order[:10]] and len(first) == 10
    assert "AC power not restored" in (tmp_path / "progress.json").read_text()

    second = _fake_campaign(monkeypatch, lambda: ({"ac": True}, 0.0))
    assert drv.campaign([1], directory=tmp_path) == 0
    assert second == [s.id for s in order[10:]]
    entries = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert len(entries) == 144 and all(e["attempt"] == 1 and e["revision"] == 1 for e in entries)


def test_a_wait_for_ac_is_recorded_with_the_run_it_delayed(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter([({"ac": True}, 120.0)] + [({"ac": True}, 0.0)] * 143)
    _fake_campaign(monkeypatch, lambda: next(calls))
    assert drv.campaign([1], directory=tmp_path) == 0
    entries = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert entries[0]["ac_wait"]["waited_s"] == 120.0 and all("ac_wait" not in e for e in entries[1:])


def test_a_campaign_leaves_another_revision_untouched(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = tmp_path / "campaign", tmp_path / "campaign-r1"
    old.mkdir()
    (old / "attempts.jsonl").write_text('{"run_id": "x"}\n')
    (old / "progress.log").write_text("old\n")
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    _fake_campaign(monkeypatch, lambda: ({"ac": True}, 0.0))
    assert drv.campaign([1], directory=new) == 0
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before
    assert drv.CAMPAIGN_DIR != drv.SUPERSEDED_DIRS[0]
