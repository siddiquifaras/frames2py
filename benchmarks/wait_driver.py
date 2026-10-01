"""The driver of the ``wait_for_newer`` producer-cost measurement (``benchmarks/wait_preregistration.md``).

It runs outside the workers. Before a session it checks the repository (clean tree, the
preregistration as committed), both builds (each worktree clean at its commit), the
environments, and the machine (AC power, Low Power Mode, Docker, the observation study's
process denylist), and refuses to start if any check fails. Around every run it records power,
thermal state and swap, samples ``ps`` every 5 s while the worker runs, and classifies the
attempt. Every attempt is appended to ``attempts.jsonl``; nothing is overwritten. Progress is
written after every run, an interrupted pass resumes from its last completed run, and every
session ends with a summary.

    uv run python -m benchmarks.wait_driver campaign      # the evidentiary measurement
    uv run python -m benchmarks.wait_driver validate      # harness check, never evidence

The machine checks are the observation study's (``benchmarks/observation_driver.py``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from benchmarks import observation_driver as od
from benchmarks.wait import (
    BUILD_DIRS,
    BUILDS,
    ENVIRONMENTS,
    MEASURE_DIR,
    PREREGISTRATION,
    REPETITIONS,
    REPO,
    Spec,
    pass_order,
    sha256,
)

CAMPAIGN_DIR: Final = MEASURE_DIR / "campaign"
VALIDATION_DIR: Final = MEASURE_DIR / "validation"
MAX_ATTEMPTS: Final = 3
IDLE_GAP_S: Final = 2.0
CHILD_TIMEOUT_S: Final = 300.0

VALID, REFUSED, INVALID_ENV, INVALID_INTEGRITY = "VALID", "REFUSED", "INVALID_ENV", "INVALID_INTEGRITY"
HARNESS_FAILURE, SHUTDOWN_TIMEOUT, CRASH = "HARNESS_FAILURE", "SHUTDOWN_TIMEOUT", "CRASH"
STOPPING: Final = frozenset({REFUSED, INVALID_INTEGRITY, HARNESS_FAILURE, SHUTDOWN_TIMEOUT, CRASH})


def _git(*args: str, cwd: Path = REPO) -> tuple[int | None, str]:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=20, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    return result.returncode, result.stdout


def repository_state() -> dict[str, Any]:
    _, status = _git("status", "--porcelain", "--untracked-files=normal")
    _, head = _git("rev-parse", "HEAD")
    code, committed = _git("show", f"HEAD:{PREREGISTRATION.relative_to(REPO)}")
    builds = {}
    for name, commit in BUILDS.items():
        directory = BUILD_DIRS[name]
        _, build_head = _git("rev-parse", "HEAD", cwd=directory) if directory.exists() else (None, "")
        _, build_status = _git("status", "--porcelain", cwd=directory) if directory.exists() else (None, "missing")
        builds[name] = {"dir": str(directory), "expected": commit, "head": build_head.strip(),
                        "clean": build_status.strip() == ""}
    return {"head": head.strip(), "clean": status.strip() == "", "status": status.strip().splitlines()[:20],
            "preregistration_as_committed": code == 0 and committed == PREREGISTRATION.read_text(),
            "preregistration_sha256": sha256(PREREGISTRATION), "builds": builds}


def repository_problems(state: dict[str, Any]) -> list[str]:
    problems = []
    if not state["clean"]:
        problems.append(f"the working tree is not clean: {state['status']}")
    if not state["preregistration_as_committed"]:
        problems.append("the preregistration file differs from its content at HEAD")
    for name, build in state["builds"].items():
        if build["head"] != build["expected"]:
            problems.append(f"build {name} at {build['dir']} is at {build['head'] or 'nothing'}, "
                            f"expected {build['expected']}")
        if not build["clean"]:
            problems.append(f"build {name} at {build['dir']} is not clean")
    return problems


def session_checks(unattended: bool, gate_env: bool) -> tuple[dict[str, Any], list[str]]:
    record = od.session_record(unattended)
    repo = repository_state()
    machine = od.machine_state(MEASURE_DIR)
    procs = od.parse_ps(od._run("ps", "-A", "-o", "pid=,ppid=,pcpu=,rss=,comm=")[1])
    docker_code, _ = od._run("docker", "info", timeout=15.0)
    others = od.other_benchmarks(os.getpid())
    gating = repository_problems(repo) + od.machine_problems(machine, procs, docker_code, others)
    missing = [f"runtime {runtime}: no environment at {python}" for runtime, python in ENVIRONMENTS.items()
               if not python.exists()]
    record.update(repository=repo, machine=machine, docker_info_code=docker_code, gate_env=gate_env,
                  not_gating=[] if gate_env else gating)
    return record, missing + (gating if gate_env else [])


def child_environment(build: str) -> dict[str, str]:
    env = od.child_environment()
    env["PYTHONPATH"] = str(BUILD_DIRS[build] / "src")
    return env


def classify(returncode: int | None, record: dict[str, Any] | None, pre: dict[str, Any], post: dict[str, Any],
             load: dict[str, Any], timed_out: bool, gate_env: bool = True) -> tuple[str, list[str]]:
    """``(outcome, reasons)`` for one attempt (preregistration section 9)."""
    if record is None:
        if returncode is not None and returncode < 0 and not timed_out:
            return CRASH, [f"the worker died with signal {-returncode}"]
        return HARNESS_FAILURE, ["the worker exited without a result" + (" (timed out)" if timed_out else "")]
    if returncode is not None and returncode < 0 and not timed_out:
        return CRASH, [f"the worker died with signal {-returncode}"]
    if record.get("refused"):
        return REFUSED, list(record["refused"])
    if record.get("fatal"):
        return HARNESS_FAILURE, [record["fatal"].strip().splitlines()[-1]]
    if record.get("power_refused"):
        return INVALID_ENV, [record["power_refused"]]
    result = record.get("result") or {}
    if result.get("waiter_errors"):
        return INVALID_INTEGRITY, [f"a waiter raised: {e.strip().splitlines()[-1]}" for e in result["waiter_errors"]]
    if result.get("waiters_alive"):
        return SHUTDOWN_TIMEOUT, [f"waiters alive after shutdown: {result['waiters_alive']}"]
    checks = result.get("checks") or {}
    if not checks.get("valid", False):
        return INVALID_INTEGRITY, list(checks.get("failures", ["the run's checks did not pass"]))
    reasons = od.environment_reasons(record, pre, post, load)
    if reasons and gate_env:
        return INVALID_ENV, reasons
    return VALID, [f"not gating: {r}" for r in reasons]


def attempt(spec: Spec, *, pass_index: int, attempt_index: int, session: str, out_dir: Path,
            gate_env: bool) -> dict[str, Any]:
    """One attempt at one spec, with the checks around it. Returns the ledger entry."""
    run_id = f"{spec.id}_p{pass_index}_a{attempt_index}"
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if (runs / f"{run_id}.json").exists():
        raise FileExistsError(f"{run_id} exists; results are never overwritten")
    entry: dict[str, Any] = {"run_id": run_id, "spec": spec.to_record(), "pass": pass_index,
                             "attempt": attempt_index, "session": session, "started_at": od._now(),
                             "gate_env": gate_env}
    repo = repository_state()
    entry["commit"] = repo["head"]
    if gate_env and repository_problems(repo):
        entry.update(outcome=REFUSED, reasons=repository_problems(repo), ended_at=od._now())
        return entry
    thermal_state, waited = od.wait_thermal() if gate_env else (od.thermal(), 0.0)
    pre = od.machine_state(MEASURE_DIR)
    pre.update(thermal=thermal_state, thermal_waited_s=waited)
    entry["pre"] = pre
    if gate_env and not thermal_state["ok"]:
        entry.update(outcome=INVALID_ENV, reasons=["thermal warning persisted for 10 min at the start"],
                     ended_at=od._now(), pause=True)
        return entry
    request = {"run_id": run_id, "spec": spec.to_record(), "out_dir": str(runs),
               "build_commit": BUILDS[spec.build]}
    proc = subprocess.Popen([str(ENVIRONMENTS[spec.runtime]), "-m", "benchmarks.wait", "worker"], cwd=REPO,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=child_environment(spec.build))
    monitor = od.LoadMonitor(proc.pid)
    monitor.start()
    timed_out = False
    try:
        _, stderr = proc.communicate(json.dumps(request), timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        _, stderr = proc.communicate()
    load = monitor.stop()
    path = runs / f"{run_id}.json"
    record = json.loads(path.read_text()) if path.exists() else None
    post = od.machine_state(MEASURE_DIR)
    outcome, reasons = classify(proc.returncode, record, pre, post, load, timed_out, gate_env)
    entry.update(post=post, load={"violations": load["violations"], "samples": len(load["samples"]),
                                  "max_total_pct": max((s["total_pct"] for s in load["samples"]), default=None)},
                 returncode=proc.returncode, timed_out=timed_out, child_pid=proc.pid,
                 stderr_tail=stderr[-4000:] if outcome != VALID else "", outcome=outcome, reasons=reasons,
                 ended_at=od._now())
    return entry


def pass_plan(pass_index: int, entries: Sequence[dict[str, Any]]) -> tuple[list[Spec], dict[str, int]]:
    """The specs still to do in a pass, in order, and the attempts made so far per spec: specs
    never attempted first, in the seeded order; then those whose last attempt was INVALID_ENV
    with attempts left, in the order they failed."""
    mine = [e for e in entries if e["pass"] == pass_index]
    attempts: Counter[str] = Counter(Spec.from_record(e["spec"]).id for e in mine)
    last = {Spec.from_record(e["spec"]).id: e for e in mine}
    todo = [s for s in pass_order(pass_index) if s.id not in attempts]
    retry = sorted((e for e in last.values() if e["outcome"] == INVALID_ENV
                    and attempts[Spec.from_record(e["spec"]).id] < MAX_ATTEMPTS and not e.get("pause")),
                   key=lambda e: str(e["ended_at"]))
    todo += [Spec.from_record(e["spec"]) for e in retry]
    return todo, dict(attempts)


def campaign(passes: Sequence[int], *, directory: Path = CAMPAIGN_DIR, unattended: bool = False,
             gate_env: bool = True) -> int:
    """Run passes. Returns 0 when every requested pass is complete."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sessions").mkdir(exist_ok=True)
    record, problems = session_checks(unattended, gate_env)
    session = record["session"]
    if problems:
        record.update(stopped="REFUSED at session start", problems=problems, ended_at=od._now())
        (directory / "sessions" / f"{session}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
        od.write_progress(directory, session, {"line": f"REFUSED at session start: {problems}", "stopped": True})
        print(f"refusing to start: {problems}", file=sys.stderr)
        return 3
    ledger = od.Ledger(directory)
    counts: Counter[str] = Counter()
    stopped: str | None = None
    completed: list[int] = []
    for pass_index in passes:
        todo, attempts = pass_plan(pass_index, ledger.entries())
        total = len(pass_order(pass_index))
        while todo:
            spec = todo.pop(0)
            n = attempts.get(spec.id, 0) + 1
            entry = attempt(spec, pass_index=pass_index, attempt_index=n, session=session, out_dir=directory,
                            gate_env=gate_env)
            ledger.append(entry)
            attempts[spec.id] = n
            counts[entry["outcome"]] += 1
            if entry["outcome"] == INVALID_ENV and n < MAX_ATTEMPTS and not entry.get("pause"):
                todo.append(spec)
            done = total - len({s.id for s in todo})
            od.write_progress(directory, session, {
                "line": f"pass {pass_index}: {done}/{total} {entry['run_id']} {entry['outcome']}",
                "pass": pass_index, "done": done, "total": total,
                "last": {k: entry.get(k) for k in ("run_id", "outcome", "reasons")}, "counts": dict(counts)})
            if entry["outcome"] in STOPPING or entry.get("pause"):
                stopped = f"{entry['outcome']} at {entry['run_id']}: {entry['reasons']}"
                break
            time.sleep(IDLE_GAP_S)
        if stopped:
            break
        completed.append(pass_index)
    record.update(ended_at=od._now(), passes_completed=completed, counts=dict(counts), stopped=stopped)
    (directory / "sessions" / f"{session}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    od.write_progress(directory, session, {"line": f"session ended: {stopped or 'all requested passes complete'}",
                                           "stopped": bool(stopped), "counts": dict(counts),
                                           "completed": completed})
    return 0 if not stopped else 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.wait_driver")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("campaign", help="the preregistered measurement")
    run.add_argument("--pass", dest="passes", type=int, action="append",
                     help=f"repeatable; default all of 1..{REPETITIONS}")
    run.add_argument("--unattended", action="store_true")
    check = sub.add_parser("validate", help="exercise the harness once per spec; never evidence")
    check.add_argument("--pass", dest="passes", type=int, action="append", help="default 1")
    args = parser.parse_args(argv)
    if args.command == "campaign":
        return campaign(args.passes or list(range(1, REPETITIONS + 1)), unattended=args.unattended)
    stamp = f"{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%SZ}"
    return campaign(args.passes or [1], directory=VALIDATION_DIR / stamp, gate_env=False)


if __name__ == "__main__":
    sys.exit(main())
