# `wait_for_newer` producer cost: preregistration

The measurement `decisions.md` 80 requires in Phase 1 of `planv1.1.md` (section 5). This file is committed before the
first evidentiary datapoint. After that, it changes only by numbered, dated amendments (section 11), each committed
before the runs it affects.

Code: `benchmarks/wait.py` (definitions and worker), `benchmarks/wait_driver.py` (sessions, checks, ledger),
`benchmarks/wait_analysis.py` (the rule of section 8). Results go to `scratch/v11_phase1/measure/campaign/`.

## 1. Questions

- **Q0, the criterion (decisions.md 80).** With no waiter, does `wait_for_newer` make the producer slower? The feature
  adds, per publication, one `object()` allocation, one `deque.append` and one `deque.popleft` (each in the deque's
  critical section on free-threaded builds), and a loop test. The criterion is met if no preregistered cell is
  distinguishably *slower* with the feature than with the baseline, by section 8. A distinguishably *faster* cell is not
  a failure; it is reported and investigated before any claim (section 8.4).
- **Q1, characterisation (decisions.md 80: "W = 1, 4, 8: characterisation only").** What does the producer's
  `ingest()` cost with 1, 4 and 8 waiters, at the gate's Engine-level cells (M2), and per publication with exactly W
  waiters registered (M1)? No pass/fail.

Not asked, and not claimed from this data: throughput on any other machine, workload or runtime; "zero overhead"; any
consumer-side latency; the comparison with polling (planv1.1.md section 14 docs).

## 2. Builds and labels

| build | commit | what it is |
|---|---|---|
| baseline | `e63150e1da26d1177d0f7246ab6123d0140d214a` | `v1.1-pre-release` before Phase 1 (= `main`) |
| feature | `c0b6d9a4df7d87eafb27ea6b791e01b0418f3904` | the commit that adds `wait_for_newer` |

Each build is a clean git worktree under `scratch/v11_phase1/measure/builds/<build>/`; the worker imports `frames2py`
from that worktree's `src/` through `PYTHONPATH` and refuses to run if it got anything else (section 7). The harness
itself (`benchmarks/`) is the measured commit's, identical for every label.

| label | build | waiters W | role |
|---|---|---|---|
| B | baseline | 0 | the baseline |
| B′ (`B'`) | baseline | 0 | A/A control: the baseline again under a second label, interleaved with everything else |
| F0 | feature | 0 | Q0 |
| F1, F4, F8 | feature | 1, 4, 8 | Q1 |

**Waiters.** Each is a thread looping `s = engine.wait_for_newer(seq); seq = s.meta.sequence`, starting with
`seq = None`, with no timeout and no other work, so it re-registers as soon as it returns. The threads start before
the warmup call, and every waiter is registered before it. They stop after the timed calls: a flag is set, one more
publication (untimed) wakes the registered ones, and all are joined within 10 s.

## 3. Runtimes and environments

| runtime | interpreter | NumPy | environment |
|---|---|---|---|
| A | CPython 3.11.14 | 2.4.6 | `scratch/v11_phase1/measure/envs/py311` |
| B | CPython 3.14.2, free-threaded, GIL disabled | 2.4.6 | `scratch/v11_phase1/measure/envs/py314t` |

These are the performance-reference environments of the v1 gate (decisions.md 17, benchmarks.md "v1 gate"). Each
environment holds NumPy only; `frames2py` comes from the build. The worker refuses to run on any other Python or NumPy
version, with the GIL enabled on runtime B, or disabled on runtime A.

Runtime B is 3.14.2t, which has the `_PyRawMutex_LockSlow` race first fixed in 3.14.5 (CPython gh-148820;
`scratch/v11_phase1/REVERIFICATION.md` section 9). It stays the reference runtime here; a crash is a CRASH outcome
(section 9).

Machine: the Apple M4 MacBook Pro of benchmarks.md "Machine and method" (Mac16,1, 4P + 6E cores, 16 GiB), macOS 15.7.7.

## 4. Experiments

### M2: the gate's Engine-level cells (Q0 and Q1)

The `v1-engine` target of `benchmarks/targets/v1.py`, unchanged: `Engine.ingest()` on the virtual 20M events/s arrival
clock, one warmup call, distinct pre-generated batches (`benchmarks.workloads`, seed `benchmarks.matrix.default_seed`),
GC disabled while timing, one `perf_counter_ns` interval per call, and the target's own result checks.

- kernel `event_count`; uniform events;
- resolutions 346x260, 640x480, 1280x720;
- (events per call, interval): (10k, 0 ms), (100k, 0 ms), (100k, 16 ms);
- timed calls: 50 at 10k and 20 at 100k (`DEFAULT_TIMED_CALLS`), raised for the 16 ms cells until the timed calls
  hold at least 10 publications (`engine_timed_calls`).

9 cells per runtime. At 0 ms every call publishes, which makes the per-publication cost as visible as the gate's grid
allows; (100k, 16 ms) is a gate cell with occasional publications.

### M1: one publication per call, with exactly W waiters registered (Q0 and Q1)

`Engine.ingest()` of one in-bounds event at interval 0 on an `event_count` Engine, at 346x260 and 1280x720 (2 cells per
runtime). 50 warmup calls, then 2,000 timed calls. Before every call, outside the timed interval, the harness spins
until exactly W waiters are registered, so every publication releases W waiters (for W = 0, it doesn't wait). The run
checks that every call published, that each waiter returned once per publication, and that the last snapshot holds
the last call's single event.

## 5. Metrics

Per run, from that run's timed calls:

| experiment | metric | faster is |
|---|---|---|
| M2 | `events_per_s`: events in the timed calls / the sum of their call times (the gate's sustained statistic) | higher |
| M2 | `p50`, `p95`, `p99`: nearest-rank percentiles of the call times | lower |
| M1 | `median_ns`: median call time | lower |
| M1 | `p95`, `p99`: nearest-rank percentiles of the call times | lower |

Also recorded per run, not part of the rule: every call time; waiter returns; the process's `proc_pid_rusage` over the
timed region (CPU time, P-core CPU time, cycles, instructions); thread counts; the power guard's record; the runtime.

Per cell and label: the median of the per-run values, with their min and max. Every per-run value is published.

## 6. Repetitions and order

- 5 passes. A pass runs every (runtime, experiment, cell, label) once: 2 × (2 + 9) × 6 = 132 runs, so 660 runs in all.
- Within a pass the order is a seeded shuffle (`benchmarks.wait.pass_order`, seed `20261002 * 100 + pass`), so labels,
  cells and runtimes are interleaved and drift spreads across them.
- One process per run. 2 s idle between runs.

## 7. Environment controls

`planv1.1.md` section 1 and the observation study's 13.2-13.3, as follows. **The human prepares the machine; nothing is
measured on a machine in normal use.**

Operator checklist, before the session:
1. AC power connected; Low Power Mode off; lid open.
2. Every application closed except one terminal (Terminal.app, not an IDE's): no VS Code, Cursor or browser; Docker
   Desktop quit, the daemon down (`docker info` fails) and `com.docker.vmnetd` not running; no virtual machines.
3. No development agent running other than, at most, the Claude Code session that launched the driver, idle while it
   runs.
4. No other benchmark or build. Time Machine backup paused; no software update downloading.
5. The repository clean at the measured commit; both build worktrees clean at their commits.
6. Launch the driver detached under `caffeinate -dimsu` (command in section 10).

Checked by the driver, refusing to start the session if any fails: a clean tree; this file as committed at HEAD; both
builds at their commits and clean; both environments present; AC power; Low Power Mode off; the Docker daemon down; no
denylisted process (Docker and `com.docker.*`, `qemu*`, `codex`, Cursor, VS Code); no other `python -m benchmarks`
process.

Around every run (as `benchmarks/observation_driver.py`): the power guard (`benchmarks.power.hold_awake()`) inside the
worker, refusing outside full wake and recording sleep and the end state; `pmset -g batt` and Low Power Mode at start and
end; `pmset -g therm` (wait up to 10 min for a clear state at the start); swap-outs; `ps` every 5 s, flagging any other
process at or above 10% of a core, or all of them together at or above 25%, in two consecutive samples.

## 8. Analysis rule (Q0)

This is the observation study's 16.3 rule, as decisions.md 80 (corrected 2026-10-01) applies it. It is not the M3/M3f
10% rule.

1. **A/A band.** For each experiment and metric: `band = the largest max(r, 1/r)` over the A/A cells, where `r` is the
   ratio of the medians of B′ and B in that cell. All cells of both runtimes are A/A cells (18 for M2, 4 for M1). No
   floor is applied. A cell where B or B′ has fewer than 5 valid runs is left out of the band, and the analysis says so.
2. **Distinguishable.** F0 is distinguishable from B in a cell, on a metric, only if the ratio of their medians lies
   outside `[1/band, band]` **and** their per-run min-max ranges don't overlap. Otherwise they are not distinguishable at
   this measurement's resolution. Results never say "equivalent", "the same" or "no difference".
3. **Verdict.**
   - **MET** if every preregistered cell-metric comparison could be made (B and F0 each with 5 valid runs, and a band)
     and none is distinguishably slower: a time metric higher, or `events_per_s` lower.
   - **NOT MET** if any comparison is distinguishably slower.
   - **NOT ESTABLISHED** otherwise: some comparisons could not be made; they are listed.
4. **A distinguishably faster comparison** is reported with its cell and metric, and investigated before any claim,
   starting from the recorded cycles per CPU nanosecond and P-core share (the frequency-scaling lead,
   `.claude/hill-climbing.md` "Paced-mode producer placement"). The feature only adds work, so a speed-up suggests a
   confound.
5. **Q1** is described, not tested: for each cell, metric and W, the ratio of F_W's median to F0's, with ranges; for
   M1, also `(median(F_W) - median(F0)) / W` in nanoseconds. No other comparison is tested.
6. **No outlier removal,** trimming or winsorising. Invalid runs are excluded only by section 9.

## 9. Run outcomes

| outcome | cause | effect |
|---|---|---|
| VALID | every check passed | used |
| REFUSED | wrong Python, NumPy, GIL state or build; an unclean tree or build at a run's start | the session stops; nothing measured |
| INVALID_ENV | the power guard refused or recorded sleep or a non-full-wake end; not on AC, or Low Power Mode on, at start or end; a thermal warning at the end; swap-outs; competing load | re-queued at the end of its pass, at most 2 retries; every attempt kept |
| INVALID_INTEGRITY | the target's result checks failed; a waiter raised | the session stops for review; reported whatever the cause |
| HARNESS_FAILURE | an exception in the harness; a worker with no result | the session stops; the harness is fixed by an amendment, and the affected pass is re-run whole |
| SHUTDOWN_TIMEOUT | waiters still alive 10 s after shutdown | the session stops for review |
| CRASH | the worker died from a signal | the session stops for review |

A cell with fewer than 5 valid runs is reported as "n of 5 valid" and left out of section 8's comparisons. An interrupted
session resumes at its last completed run; a pass re-run whole after an amendment keeps the earlier attempts.

## 10. Harness validation and launch

**Harness validation (not evidence).** Before this file was committed, `uv run python -m benchmarks.wait_driver
validate` ran every spec once on a machine in normal use, with the environment checks recorded but not gating, to show
that each spec runs, its checks pass and its waiters shut down. Its outcomes were looked at; its timings were not, and
they are never used. It is kept in `scratch/v11_phase1/measure/validation/`:
- `20261001T204330Z` stopped at its 16th run, INVALID_INTEGRITY: M1 read the waiters' return count before every waiter
  had counted its return from the last publication (8,197 of 8,200 at W = 4). The harness now waits for every waiter
  to register again, which each does only after counting, before it reads the count.
- `20261001T204422Z`, with that fix: 132 of 132 VALID.

**Launch** (the operator, after section 7's checklist, from Terminal.app at the repository root):

```sh
mkdir -p scratch/v11_phase1/measure/campaign
nohup caffeinate -dimsu uv run --no-sync python -m benchmarks.wait_driver campaign --unattended \
    > scratch/v11_phase1/measure/campaign/driver.log 2>&1 &
tail -f scratch/v11_phase1/measure/campaign/progress.log
```

Then `uv run --no-sync python -m benchmarks.wait_analysis` writes `summary.md` and `summary.json` in the campaign
directory.

## 11. Amendments

Numbered and dated, each naming what changed and why, committed before any run it affects. In an unattended session only
a harness defect may be amended (as the observation study's 14.2); anything else stops the session for the user. No
amendment changes section 8 after the first evidentiary run.
