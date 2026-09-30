"""The observation study's driver (preregistration 11.4, 13, 14): sessions, runs and passes.

The driver runs outside the children. Before a session it checks the repository (clean tree,
the preregistration commit an ancestor of HEAD, the protocol file as committed) and the
machine (AC power, Low Power Mode, Docker, the denylist), and refuses to start if any check
fails. Around every run it records power, thermal state, swap and load, samples ``ps`` every
5 s while the child runs, and classifies the attempt (14.1). Every attempt is appended to
``attempts.jsonl``; nothing is ever overwritten. Progress is written after every run, an
interrupted pass resumes from its last completed run, and every session ends with a summary.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

from benchmarks.observation import (
    ENVIRONMENTS,
    POOL_SHA256,
    PREREGISTERED,
    PREREGISTRATION,
    PREREGISTRATION_COMMIT,
    REPO,
    STUDY_DIR,
    Cell,
    Condition,
    pass_order,
)

CAMPAIGN_DIR: Final = STUDY_DIR / "campaign"
VALIDATION_DIR: Final = STUDY_DIR / "validation"
CALIBRATION_RECORD: Final = Path(__file__).with_name("observation_calibration.json")
CAMPAIGN_REVISION: Final = 0
"""Raised by an amendment (24) whose re-run passes replace earlier ones; both are kept."""

MAX_ATTEMPTS: Final = 3
"""An INVALID_ENV attempt is re-queued at the end of its pass at most twice (14.1)."""
IDLE_GAP_S: Final = 5.0
PS_PERIOD_S: Final = 5.0
PROCESS_LIMIT_PCT: Final = 10.0
TOTAL_LIMIT_PCT: Final = 25.0
THERMAL_WAIT_S: Final = 600.0
THERMAL_POLL_S: Final = 30.0
P2_MIN_FREE_BYTES: Final = 20 * 2**30
CHILD_TIMEOUT_S: Final = 900.0

VALID: Final = "VALID"
REFUSED: Final = "REFUSED"
INVALID_ENV: Final = "INVALID_ENV"
INVALID_INTEGRITY: Final = "INVALID_INTEGRITY"
HARNESS_FAILURE: Final = "HARNESS_FAILURE"
SHUTDOWN_TIMEOUT: Final = "SHUTDOWN_TIMEOUT"
CRASH: Final = "CRASH"
STOPPING: Final = frozenset({REFUSED, INVALID_INTEGRITY, HARNESS_FAILURE, SHUTDOWN_TIMEOUT, CRASH})
"""Outcomes after which the campaign stops (or pauses) for review (14.1, 14.2)."""


# ---------------------------------------------------------------- shell


def _run(*args: str, timeout: float = 20.0) -> tuple[int | None, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    return result.returncode, result.stdout


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- machine state


def parse_batt(text: str) -> dict[str, Any]:
    """``pmset -g batt``: power source, battery percentage and charging state."""
    first = text.splitlines()[0] if text else ""
    source = first.split("'")[1] if first.count("'") >= 2 else None
    match = re.search(r"(\d+)%;\s*([^;]+);", text)
    return {"source": source, "ac": source == "AC Power",
            "percent": int(match.group(1)) if match else None,
            "state": match.group(2).strip() if match else None}


def parse_low_power(text: str) -> bool | None:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "lowpowermode" and parts[1] in ("0", "1"):
            return parts[1] == "1"
    return None


THERMAL_CLEAR: Final = ("No thermal warning level has been recorded",
                        "No performance warning level has been recorded",
                        "No CPU power status has been recorded")


def parse_thermal(text: str) -> dict[str, Any]:
    """``pmset -g therm``: clear only when all three "nothing recorded" notes are present."""
    return {"ok": all(note in text for note in THERMAL_CLEAR), "text": text.strip()}


def parse_swapouts(text: str) -> int | None:
    match = re.search(r"Swapouts:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def power() -> dict[str, Any]:
    _, batt = _run("pmset", "-g", "batt")
    _, settings = _run("pmset", "-g")
    return {**parse_batt(batt), "low_power_mode": parse_low_power(settings)}


def thermal() -> dict[str, Any]:
    _, text = _run("pmset", "-g", "therm")
    return parse_thermal(text)


def swap() -> dict[str, Any]:
    _, usage = _run("sysctl", "-n", "vm.swapusage")
    _, stat = _run("vm_stat")
    return {"swapusage": usage.strip(), "swapouts": parse_swapouts(stat)}


def machine_state(path: Path) -> dict[str, Any]:
    free = shutil.disk_usage(path).free
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {"at": _now(), "power": power(), "thermal": thermal(), "swap": swap(), "disk_free_bytes": free,
            "load_average": load}


# ---------------------------------------------------------------- processes


def parse_ps(text: str) -> list[dict[str, Any]]:
    """``ps -A -o pid=,ppid=,pcpu=,rss=,comm=`` rows."""
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "pcpu": float(parts[2]),
                         "rss": int(parts[3]), "comm": parts[4].strip()})
        except ValueError:
            continue
    return rows


def denylisted(comm: str) -> str | None:
    """Which denylist entry (13.3) a process's command matches, or ``None``. ``claude`` is not
    on it (13.4)."""
    base = os.path.basename(comm).lower()
    path = comm.lower()
    if "/applications/docker.app/" in path or base.startswith("com.docker.") or base in ("docker", "dockerd"):
        return "Docker"
    if base.startswith("qemu"):
        return "qemu"
    if base == "codex" or "/codex/" in path:
        return "codex"
    if "/applications/cursor.app/" in path:
        return "Cursor"
    if "/applications/visual studio code.app/" in path or base.startswith("code helper"):
        return "VS Code"
    return None


def lineage(pid: int, parents: dict[int, int]) -> set[int]:
    """*pid*, its ancestors and its descendants, from a pid-to-parent map."""
    descendants = {pid}
    grew = True
    while grew:
        grew = False
        for child, parent in parents.items():
            if parent in descendants and child not in descendants:
                descendants.add(child)
                grew = True
    ancestors = set()
    here = pid
    while here in parents and parents[here] > 1 and parents[here] not in ancestors:
        here = parents[here]
        ancestors.add(here)
    return descendants | ancestors


def other_benchmarks(pid: int) -> list[str]:
    """Other ``python -m benchmarks`` processes, outside the driver's own lineage (whatever
    launched it, and its children)."""
    _, text = _run("ps", "-A", "-o", "pid=,ppid=,args=")
    parents: dict[int, int] = {}
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            parents[int(parts[0])] = int(parts[1])
            rows.append((int(parts[0]), parts[2]))
    own = lineage(pid, parents)
    return [f"{p} {args}" for p, args in rows if p not in own and re.search(r"-m\s+benchmarks\b", args)]


class LoadMonitor:
    """``ps`` every 5 s during a run (13.3). Invalid if a process other than the child, the
    driver and ``ps`` is at or above 10% of a core in two consecutive samples, or all of them
    together at or above 25% in two consecutive samples."""

    def __init__(self, child_pid: int, sampler: Callable[[], str] | None = None) -> None:
        self.child_pid = child_pid
        self.driver_pid = os.getpid()
        self._sampler = sampler or (lambda: _run("ps", "-A", "-o", "pid=,ppid=,pcpu=,rss=,comm=")[1])
        self.samples: list[dict[str, Any]] = []
        self.violations: list[str] = []
        self._previous_high: set[int] = set()
        self._previous_total_high = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="load-monitor", daemon=True)

    def observe(self, text: str) -> None:
        rows = [r for r in parse_ps(text)
                if r["pid"] not in (self.child_pid, self.driver_pid)
                and not (os.path.basename(r["comm"]) == "ps" and r["ppid"] == self.driver_pid)]
        high = {r["pid"] for r in rows if r["pcpu"] >= PROCESS_LIMIT_PCT}
        total = sum(r["pcpu"] for r in rows)
        top = sorted(rows, key=lambda r: -r["pcpu"])[:5]
        self.samples.append({"total_pct": total, "top": [(r["pid"], r["pcpu"], r["comm"]) for r in top]})
        for pid in high & self._previous_high:
            comm = next(r["comm"] for r in rows if r["pid"] == pid)
            self.violations.append(f"{comm} (pid {pid}) at or above {PROCESS_LIMIT_PCT:g}% in two samples")
        if total >= TOTAL_LIMIT_PCT and self._previous_total_high:
            self.violations.append(f"other processes together at {total:.1f}% in two samples")
        self._previous_high = high
        self._previous_total_high = total >= TOTAL_LIMIT_PCT

    def _loop(self) -> None:
        while not self._stop.wait(PS_PERIOD_S):
            self.observe(self._sampler())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        return {"samples": self.samples, "violations": sorted(set(self.violations))}


# ---------------------------------------------------------------- repository and session checks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_state() -> dict[str, Any]:
    _, status = _run("git", "status", "--porcelain", "--untracked-files=normal")
    _, head = _run("git", "rev-parse", "HEAD")
    ancestor, _ = _run("git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD")
    code, committed = _run("git", "show", "HEAD:benchmarks/observation_preregistration.md")
    on_disk = PREREGISTRATION.read_text() if PREREGISTRATION.exists() else None
    _, branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    return {"head": head.strip(), "branch": branch.strip(), "clean": status.strip() == "",
            "status": status.strip().splitlines()[:20], "preregistration_ancestor": ancestor == 0,
            "preregistration_as_committed": code == 0 and committed == on_disk,
            "preregistration_sha256": _sha256(PREREGISTRATION)}


def repository_problems(state: dict[str, Any]) -> list[str]:
    problems = []
    if not state["clean"]:
        problems.append(f"the working tree is not clean: {state['status']}")
    if not state["preregistration_ancestor"]:
        problems.append(f"the preregistration commit {PREREGISTRATION_COMMIT} is not an ancestor of HEAD")
    if not state["preregistration_as_committed"]:
        problems.append("the preregistration file differs from its content at HEAD")
    return problems


def machine_problems(state: dict[str, Any], processes: list[dict[str, Any]], docker_info_code: int | None,
                     benchmarks: list[str]) -> list[str]:
    problems = []
    p = state["power"]
    if not p["ac"]:
        problems.append(f"not on AC power ({p['source']})")
    if p["low_power_mode"] is not False:
        problems.append(f"Low Power Mode is {p['low_power_mode']}")
    if docker_info_code == 0:
        problems.append("the Docker daemon is up (`docker info` succeeds)")
    for r in processes:
        hit = denylisted(r["comm"])
        if hit:
            problems.append(f"denylisted process: {hit}: {r['comm']} (pid {r['pid']})")
    for line in benchmarks:
        problems.append(f"another benchmark process: {line}")
    return problems


def session_record(unattended: bool) -> dict[str, Any]:
    def sysctl(name: str) -> str:
        return _run("sysctl", "-n", name)[1].strip()

    _, sw = _run("sw_vers")
    _, pmset = _run("pmset", "-g")
    _, uptime = _run("uptime")
    return {
        "session": f"{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
        "started_at": _now(), "sw_vers": sw.strip(), "hw.model": sysctl("hw.model"),
        "cpu": sysctl("machdep.cpu.brand_string"), "perflevel0": sysctl("hw.perflevel0.logicalcpu"),
        "perflevel1": sysctl("hw.perflevel1.logicalcpu"), "memsize": sysctl("hw.memsize"), "pmset": pmset,
        "uptime": uptime.strip(),
        "operator_confirmation": ("none: unattended session; the checklist items the driver cannot verify "
                                  "(lid, applications other than those denylisted, Time Machine, software "
                                  "updates, Spotlight exclusion) are unverified") if unattended else None,
        "driver_python": sys.version, "driver_pid": os.getpid(),
    }


def session_checks(unattended: bool) -> tuple[dict[str, Any], list[str]]:
    record = session_record(unattended)
    repo = repository_state()
    state = machine_state(STUDY_DIR)
    procs = parse_ps(_run("ps", "-A", "-o", "pid=,ppid=,pcpu=,rss=,comm=")[1])
    docker_code, _ = _run("docker", "info", timeout=15.0)
    benchmarks = other_benchmarks(os.getpid())
    problems = repository_problems(repo) + machine_problems(state, procs, docker_code, benchmarks)
    for rt, python in ENVIRONMENTS.items():
        if not python.exists():
            problems.append(f"runtime {rt}: no study environment at {python}")
    record.update({"repository": repo, "machine": state, "docker_info_code": docker_code,
                   "problems": problems})
    return record, problems


# ---------------------------------------------------------------- one run


def calibration() -> dict[str, Any] | None:
    if not CALIBRATION_RECORD.exists():
        return None
    record: dict[str, Any] = json.loads(CALIBRATION_RECORD.read_text())
    return record


def child_environment() -> dict[str, str]:
    """The driver's environment without ``PYTHON*`` or thread-count variables (12)."""
    drop = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLOSC_NTHREADS"}
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in drop}


def classify(returncode: int | None, record: dict[str, Any] | None, pre: dict[str, Any], post: dict[str, Any],
             load: dict[str, Any], sustained: bool | None, *, gate_env: bool = True,
             timed_out: bool = False) -> tuple[str, list[str], list[str]]:
    """``(outcome, flags, reasons)`` for one attempt (14.1)."""
    reasons: list[str] = []
    flags: list[str] = []
    if record is None:
        if returncode is not None and returncode < 0 and not timed_out:
            return CRASH, flags, [f"the child died with {_signal_name(returncode)}"]
        return HARNESS_FAILURE, flags, ["the child exited without a result" + (" (timed out)" if timed_out else "")]
    if returncode is not None and returncode < 0 and not timed_out:
        return CRASH, flags, [f"the child died with {_signal_name(returncode)}"]
    if record.get("refused"):
        return REFUSED, flags, list(record["refused"])
    if record.get("fatal"):
        return HARNESS_FAILURE, flags, [record["fatal"].strip().splitlines()[-1]]
    if record.get("power_refused"):
        return INVALID_ENV, flags, [record["power_refused"]]
    errors = record.get("errors", [])
    other = [e for e in errors if e["type"] != "MemoryError"]
    if other:
        return HARNESS_FAILURE, flags, [f"{e['thread']}: {e['type']}" for e in other]
    if record.get("alive_after_grace"):
        return SHUTDOWN_TIMEOUT, flags, [f"alive after the grace period: {record['alive_after_grace']}"]
    integrity = record.get("integrity") or {}
    if integrity.get("harness"):
        return HARNESS_FAILURE, flags, list(integrity["harness"])
    problems = list(integrity.get("problems", []))
    readback = record.get("readback")
    if readback is not None and not readback.get("ok"):
        problems.append("the P2 read-back differs from the batches fed")
    if problems:
        return INVALID_INTEGRITY, flags, problems
    env = environment_reasons(record, pre, post, load)
    if env and gate_env:
        return INVALID_ENV, flags, env
    reasons += [f"not gating: {r}" for r in env]
    if any(e["type"] == "MemoryError" for e in errors):
        flags.append("ARM_MEMORY_ERROR")
    if record.get("memory_ceiling"):
        flags.append("MEMORY_CEILING")
    if record.get("drain_timeout"):
        flags.append("DRAIN_TIMEOUT")
    if sustained is False:
        flags.append("NOT_SUSTAINED")
    return VALID, flags, reasons


def environment_reasons(record: dict[str, Any], pre: dict[str, Any], post: dict[str, Any],
                        load: dict[str, Any]) -> list[str]:
    from benchmarks import power as power_module

    reasons = []
    awake = record.get("power")
    if power_module.slept_during(awake):
        reasons.append("the machine slept during the run")
    if power_module.ended_outside_full_wake(awake):
        reasons.append("the run ended outside full wake")
    for when, state in (("start", pre), ("end", post)):
        p = state.get("power", {})
        if not p.get("ac"):
            reasons.append(f"not on AC power at the {when}")
        if p.get("low_power_mode") is not False:
            reasons.append(f"Low Power Mode not off at the {when}")
    if not post.get("thermal", {}).get("ok"):
        reasons.append("a thermal, performance or CPU power warning at the end")
    a, b = pre.get("swap", {}).get("swapouts"), post.get("swap", {}).get("swapouts")
    if a is None or b is None:
        reasons.append("swap-outs could not be read")
    elif b > a:
        reasons.append(f"{b - a} swap-outs during the run")
    reasons += load.get("violations", [])
    return reasons


def wait_thermal() -> tuple[dict[str, Any], float]:
    """Wait up to 10 min, polling every 30 s, for a clear thermal state (13.3)."""
    waited = 0.0
    state = thermal()
    while not state["ok"] and waited < THERMAL_WAIT_S:
        time.sleep(THERMAL_POLL_S)
        waited += THERMAL_POLL_S
        state = thermal()
    return state, waited


def run_child(python: Path, request: dict[str, Any], child_pid_out: list[int], monitor_load: bool = True
              ) -> tuple[int | None, dict[str, Any] | None, dict[str, Any], bool, str]:
    """Run one child; ``(returncode, record, load, timed_out, stderr)``."""
    proc = subprocess.Popen([str(python), "-m", "benchmarks", "observation", "worker"], cwd=REPO,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=child_environment())
    child_pid_out.append(proc.pid)
    monitor = LoadMonitor(proc.pid)
    if monitor_load:
        monitor.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(json.dumps(request), timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    load = monitor.stop() if monitor_load else {"samples": [], "violations": [], "not_monitored": True}
    path = Path(request["out_dir"]) / f"{request['run_id']}.json"
    record = json.loads(path.read_text()) if path.exists() else None
    return proc.returncode, record, load, timed_out, stderr[-4000:]


def sustained_of(record: dict[str, Any] | None) -> bool | None:
    if record is None or "t0" not in record or not record.get("npz"):
        return None
    from benchmarks.observation_analysis import load, producer_metrics

    try:
        rec, arrays = load(Path(record["request"]["out_dir"]) / f"{record['run_id']}.json")
        return bool(producer_metrics(rec, arrays)["sustained"])
    except (KeyError, ValueError, OSError):
        return None


def attempt(cell: Cell, *, experiment: str, pass_index: int, attempt_index: int, revision: int, session: str,
            out_dir: Path, condition: Condition = PREREGISTERED, k5: int | None, gate_env: bool = True,
            instrumentation: str = "full", run_id: str | None = None) -> dict[str, Any]:
    """One attempt at one cell, with the checks around it. Returns the ledger entry."""
    run_id = run_id or f"{cell.id}_r{revision}_p{pass_index}_a{attempt_index}"
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {"run_id": run_id, "experiment": experiment, "pass": pass_index,
                             "revision": revision, "attempt": attempt_index, "cell": cell.to_record(),
                             "session": session, "started_at": _now(), "gate_env": gate_env,
                             "instrumentation": instrumentation}
    if (runs / f"{run_id}.json").exists():
        raise FileExistsError(f"{run_id} exists; results are never overwritten")
    repo = repository_state()
    entry["commit"] = repo["head"]
    if gate_env and repository_problems(repo):
        entry.update(outcome=REFUSED, flags=[], reasons=repository_problems(repo), ended_at=_now())
        return entry
    thermal_state, waited = wait_thermal() if gate_env else (thermal(), 0.0)
    pre = machine_state(STUDY_DIR)
    pre["thermal"] = thermal_state
    pre["thermal_waited_s"] = waited
    entry["pre"] = pre
    if gate_env and not thermal_state["ok"]:
        entry.update(outcome=INVALID_ENV, flags=[], reasons=["thermal warning persisted for 10 min at the start"],
                     ended_at=_now(), pause=True)
        return entry
    if gate_env and cell.experiment == "P2" and pre["disk_free_bytes"] < P2_MIN_FREE_BYTES:
        entry.update(outcome=REFUSED, flags=[], reasons=["less than 20 GiB free for a P2 run"], ended_at=_now())
        return entry
    request: dict[str, Any] = {
        "run_id": run_id, "experiment": experiment, "arm": cell.arm, "workload": cell.workload, "n": cell.n,
        "runtime": cell.runtime, "condition": condition.to_record(), "k5": k5, "out_dir": str(runs),
        "tmp_dir": str(STUDY_DIR / "tmp"), "instrumentation": instrumentation,
        "expected_pool_sha256": POOL_SHA256 if condition.pool_events == PREREGISTERED.pool_events else None,
        "check_runtime": True,
    }
    pids: list[int] = []
    returncode, record, load, timed_out, stderr = run_child(ENVIRONMENTS[cell.runtime], request, pids)
    post = machine_state(STUDY_DIR)
    sustained = sustained_of(record) if instrumentation == "full" else None
    outcome, flags, reasons = classify(returncode, record, pre, post, load, sustained, gate_env=gate_env,
                                       timed_out=timed_out)
    entry.update(post=post, load={"violations": load["violations"], "samples": len(load["samples"]),
                                  "max_total_pct": max((s["total_pct"] for s in load["samples"]), default=None)},
                 returncode=returncode, timed_out=timed_out, child_pid=pids[0] if pids else None,
                 stderr_tail=stderr if outcome != VALID else "", outcome=outcome, flags=flags, reasons=reasons,
                 sustained=sustained, ended_at=_now())
    return entry


# ---------------------------------------------------------------- the ledger and passes


class Ledger:
    """``attempts.jsonl``: one line per attempt, appended, never rewritten."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "attempts.jsonl"
        directory.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, entry: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


def pass_plan(experiment: str, pass_index: int, revision: int, entries: Sequence[dict[str, Any]]
              ) -> tuple[list[Cell], dict[str, int]]:
    """The runs still to do in one pass, in order, and the attempts made so far per cell.

    Cells never attempted come first, in the seeded order; then cells whose last attempt was
    INVALID_ENV with attempts left, in the order they failed. This is how an interrupted pass
    resumes from its last completed run (13.4).
    """
    mine = [e for e in entries if e["experiment"] == experiment and e["pass"] == pass_index
            and e["revision"] == revision]
    attempts: dict[str, int] = Counter(Cell.from_record(e["cell"]).id for e in mine)
    last: dict[str, dict[str, Any]] = {}
    for e in mine:
        last[Cell.from_record(e["cell"]).id] = e
    order = pass_order(experiment, pass_index)
    todo = [c for c in order if c.id not in attempts]
    retry = sorted((e for e in last.values() if e["outcome"] == INVALID_ENV and attempts[Cell.from_record(e["cell"]).id]
                    < MAX_ATTEMPTS), key=lambda e: str(e["ended_at"]))
    todo += [Cell.from_record(e["cell"]) for e in retry]
    return todo, dict(attempts)


K = TypeVar("K")


def run_pass_with_retries(order: Sequence[K], run: Callable[[K, int], str]) -> dict[K, str]:
    """One pass in *order*, with 14.1's rule: an INVALID_ENV run is re-queued at the end of the
    pass, at most twice. Stops at the first outcome in ``STOPPING``. Returns each run's last
    outcome; ``run(key, attempt)`` makes one attempt."""
    todo = [(key, 1) for key in order]
    final: dict[K, str] = {}
    while todo:
        key, n = todo.pop(0)
        outcome = run(key, n)
        final[key] = outcome
        if outcome in STOPPING:
            break
        if outcome == INVALID_ENV and n < MAX_ATTEMPTS:
            todo.append((key, n + 1))
    return final


RUN_ID: Final = re.compile(r"_r(\d+)_p(\d+)_a(\d+)$")
INTERRUPTED: Final = "the driver was interrupted during this attempt; the checks around it were not recorded"


def reconcile(directory: Path, ledger: Ledger, session: str) -> list[dict[str, Any]]:
    """Record every run that has a result file but no ledger entry: a child that finished after
    its driver stopped. It becomes an attempt of its pass (every attempt is kept, 14.1), so its
    id is never reused. With no environment record around it, it is at best INVALID_ENV; a
    harness, shutdown or integrity outcome in its own record still takes precedence."""
    known = {e["run_id"] for e in ledger.entries()}
    added = []
    for path in sorted((directory / "runs").glob("*.json")):
        run_id = path.stem
        match = RUN_ID.search(run_id)
        if run_id in known or match is None:
            continue
        record: dict[str, Any] = json.loads(path.read_text())
        request = record.get("request", {})
        cell = Cell(request["experiment"], request["arm"], request["workload"], int(request["n"]), request["runtime"])
        outcome, flags, reasons = classify(None if record.get("t0") is None else 0, record, {}, {},
                                           {"violations": []}, sustained_of(record))
        if outcome in (VALID, INVALID_ENV):
            # Only what the run's own record shows: the checks around it were never made.
            from benchmarks import power as power_module

            awake = record.get("power")
            reasons = ([r for r, hit in (("the machine slept during the run", power_module.slept_during(awake)),
                                         ("the run ended outside full wake",
                                          power_module.ended_outside_full_wake(awake))) if hit]
                       + ([record["power_refused"]] if record.get("power_refused") else []))
            outcome, flags = INVALID_ENV, []
        entry = {"run_id": run_id, "experiment": request["experiment"], "pass": int(match.group(2)),
                 "revision": int(match.group(1)), "attempt": int(match.group(3)), "cell": cell.to_record(),
                 "session": session, "started_at": None,
                 "ended_at": datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.UTC).isoformat(
                     timespec="seconds"),
                 "gate_env": True, "instrumentation": request.get("instrumentation", "full"), "outcome": outcome,
                 "flags": flags, "reasons": [INTERRUPTED, *reasons], "reconciled": True}
        ledger.append(entry)
        added.append(entry)
    return added


def write_progress(directory: Path, session: str, state: dict[str, Any]) -> None:
    state = {**state, "session": session, "at": _now()}
    tmp = directory / "progress.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, default=str) + "\n")
    tmp.replace(directory / "progress.json")
    with (directory / "progress.log").open("a") as f:
        f.write(f"{state['at']} {state.get('line', '')}\n")


def campaign(experiments: Sequence[str], passes: Sequence[int], *, directory: Path = CAMPAIGN_DIR,
             revision: int = CAMPAIGN_REVISION, unattended: bool = False, session_retries: int = 0,
             retry_wait_s: float = 600.0) -> int:
    """Run passes of the campaign. Returns 0 when every requested pass is complete."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sessions").mkdir(exist_ok=True)
    record, problems = session_checks(unattended)
    cal = calibration()
    if cal is None or not cal.get("k5"):
        problems.append(f"no calibration record with K5 at {CALIBRATION_RECORD}")
    tries = 0
    while problems and tries < session_retries:
        tries += 1
        print(f"session checks failed ({problems}); retry {tries} of {session_retries} in {retry_wait_s:g} s",
              flush=True)
        time.sleep(retry_wait_s)
        record, problems = session_checks(unattended)
        if cal is None or not cal.get("k5"):
            problems.append(f"no calibration record with K5 at {CALIBRATION_RECORD}")
    session = record["session"]
    if problems:
        record.update(stopped="REFUSED at session start", ended_at=_now())
        (directory / "sessions" / f"{session}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
        write_progress(directory, session, {"line": f"REFUSED at session start: {problems}", "stopped": True})
        print(f"refusing to start: {problems}", file=sys.stderr)
        return 3
    assert cal is not None
    k5 = int(cal["k5"])
    ledger = Ledger(directory)
    counts: Counter[str] = Counter()
    stopped: str | None = None
    completed: list[str] = []
    for entry in reconcile(directory, ledger, session):
        counts[entry["outcome"]] += 1
        write_progress(directory, session, {"line": f"recorded interrupted attempt {entry['run_id']}: "
                                                    f"{entry['outcome']}", "last": entry})
        if entry["outcome"] in STOPPING:
            stopped = f"{entry['outcome']} at {entry['run_id']} (interrupted attempt): {entry['reasons']}"
    if stopped:
        experiments = []
    for experiment in experiments:
        for pass_index in passes:
            todo, attempts = pass_plan(experiment, pass_index, revision, ledger.entries())
            total = len(pass_order(experiment, pass_index))
            while todo:
                cell = todo.pop(0)
                n = attempts.get(cell.id, 0) + 1
                entry = attempt(cell, experiment=experiment, pass_index=pass_index, attempt_index=n,
                                revision=revision, session=session, out_dir=directory, k5=k5)
                ledger.append(entry)
                attempts[cell.id] = n
                counts[entry["outcome"]] += 1
                for flag in entry["flags"]:
                    counts[flag] += 1
                if entry["outcome"] == INVALID_ENV and n < MAX_ATTEMPTS and not entry.get("pause"):
                    todo.append(cell)
                done = total - len({c.id for c in todo})
                write_progress(directory, session, {
                    "line": f"{experiment} pass {pass_index}: {done}/{total} {entry['run_id']} {entry['outcome']} "
                            f"{','.join(entry['flags'])}",
                    "experiment": experiment, "pass": pass_index, "done": done, "total": total,
                    "last": {k: entry.get(k) for k in ("run_id", "outcome", "flags", "reasons")},
                    "counts": dict(counts)})
                if entry["outcome"] in STOPPING or entry.get("pause"):
                    stopped = f"{entry['outcome']} at {entry['run_id']}: {entry['reasons']}"
                    break
                time.sleep(IDLE_GAP_S)
            if stopped:
                break
            completed.append(f"{experiment} pass {pass_index}")
        if stopped:
            break
    record.update(ended_at=_now(), passes_completed=completed, counts=dict(counts), stopped=stopped)
    (directory / "sessions" / f"{session}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    write_progress(directory, session, {"line": f"session ended: {stopped or 'all requested passes complete'}",
                                        "stopped": bool(stopped), "counts": dict(counts), "completed": completed})
    return 0 if not stopped else 4


# ---------------------------------------------------------------- validation stages


def _stage_dir(stage: str) -> Path:
    path = VALIDATION_DIR / f"{stage}-{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%SZ}"
    path.mkdir(parents=True)
    return path


def validate_child(stage: str, runtime: str, out: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    request = {"stage": stage, "runtime": runtime, "tmp_dir": str(STUDY_DIR / "tmp"), **(extra or {})}
    proc = subprocess.run([str(ENVIRONMENTS[runtime]), "-m", "benchmarks", "observation", "validate-worker"],
                          cwd=REPO, input=json.dumps(request), capture_output=True, text=True,
                          env=child_environment(), timeout=3600)
    record: dict[str, Any] = {"request": request, "returncode": proc.returncode, "stderr_tail": proc.stderr[-4000:]}
    try:
        record["result"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        record["result"] = None
        record["stdout_tail"] = proc.stdout[-4000:]
    (out / f"{stage}_{runtime}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    return record


V4_ARMS: Final = ("A", "B", "C", "E", "F", "G", "H", "H'", "RB", "EP-H", "EP-QB", "EP-QE")
V3_CELLS: Final = (("A", "none", 0), ("G", "none", 0), ("H", "none", 0), ("H", "W1", 1), ("B", "W1", 1),
                   ("RB", "W1", 1))
V3_REPETITIONS: Final = 3


def validate(stage: str, *, gate_env: bool) -> tuple[Path, bool]:
    """Run one validation stage; ``(output directory, passed)``."""
    out = _stage_dir(stage)
    ok = True
    if stage in ("V1", "V6"):
        for rt in ("A", "B"):
            record = validate_child(stage, rt, out)
            result = record["result"] or {}
            ok = ok and record["returncode"] == 0 and not result.get("refused") and (
                result.get("ok", True) if stage == "V1" else bool(result.get("arms")))
    elif stage == "V2":
        proc = subprocess.run(["uv", "run", "--quiet", "pytest", "-q", "-p", "no:cacheprovider",
                               "tests/test_observation_benchmarks.py"], cwd=REPO, capture_output=True, text=True)
        (out / "V2_pytest.txt").write_text(proc.stdout + proc.stderr)
        ok = proc.returncode == 0
    elif stage == "V4":
        condition = Condition(window_s=5.0)
        session = session_record(unattended=True)["session"]
        ledger = Ledger(out)
        for rt in ("A", "B"):
            for arm in V4_ARMS:
                experiment = "P2" if arm.startswith("EP-") else "P1"
                cell = Cell(experiment, arm, "W1", 1, rt)
                entry = attempt(cell, experiment="V4", pass_index=1, attempt_index=1, revision=0, session=session,
                                out_dir=out, condition=condition, k5=None, gate_env=gate_env)
                ledger.append(entry)
                ok = ok and entry["outcome"] == VALID
                print(f"V4 {cell.id}: {entry['outcome']} {entry['flags']} {entry['reasons']}", flush=True)
    elif stage == "V3":
        import random

        session = session_record(unattended=True)["session"]
        ledger = Ledger(out)
        runs = [(arm, w, n, rt, mode) for arm, w, n in V3_CELLS for rt in ("A", "B") for mode in ("full", "aggregate")]
        for rep in range(1, V3_REPETITIONS + 1):
            order = list(runs)
            random.Random(20261001 + 300 + rep).shuffle(order)

            def run_v3(key: tuple[str, str, int, str, str], n_attempt: int, rep: int = rep) -> str:
                arm, w, n, rt, mode = key
                cell = Cell("V3", arm, w, n, rt)
                entry = attempt(cell, experiment="V3", pass_index=rep, attempt_index=n_attempt, revision=0,
                                session=session, out_dir=out, k5=None, gate_env=gate_env, instrumentation=mode,
                                run_id=f"{cell.id}_{mode}_rep{rep}_a{n_attempt}")
                ledger.append(entry)
                print(f"V3 {entry['run_id']}: {entry['outcome']} {entry['reasons']}", flush=True)
                time.sleep(IDLE_GAP_S)
                return str(entry["outcome"])

            final = run_pass_with_retries(order, run_v3)
            if any(outcome in STOPPING for outcome in final.values()):
                return out, False
            ok = ok and all(outcome == VALID for outcome in final.values())
    else:
        raise ValueError(f"unknown stage {stage!r}")
    (out / "passed.json").write_text(json.dumps({"stage": stage, "passed": ok, "at": _now(),
                                                 "gate_env": gate_env}) + "\n")
    return out, ok


def calibrate() -> tuple[Path, bool]:
    """V5 on runtime A, then W5's cost at K5 on runtime B; writes the calibration record."""
    out = _stage_dir("V5")
    record = validate_child("V5", "A", out)
    result = record["result"] or {}
    if not result.get("ok"):
        return out, False
    k5 = int(result["k5"])
    cost_b = validate_child("W5_COST", "B", out, {"k5": k5})
    calibration_record = {"k5": k5, "target_ns": result["target_ns"], "v5": result,
                          "w5_cost_runtime_b": cost_b["result"], "at": _now(),
                          "preregistration_commit": PREREGISTRATION_COMMIT}
    if CALIBRATION_RECORD.exists():
        raise FileExistsError(f"{CALIBRATION_RECORD} exists; the calibration is committed once")
    CALIBRATION_RECORD.write_text(json.dumps(calibration_record, indent=1) + "\n")
    return out, True


def _signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return str(-returncode)
