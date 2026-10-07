# Benchmark methodology

How the figures on [Performance](performance.md) were measured, what the published data
file contains, and how to measure your own machine. The benchmark suite lives in the
repository's `benchmarks/` directory. It is not part of the installed package, so every
command below runs from a checkout.

**Code measured.** The v1 gate, the `wait_for_newer` measurement and the temporal-kernel
gate were each taken at the commit recorded in its section below. The ingest path changed
after each of them, and none was re-measured on the released code:

- **the v1 gate** (`6a0fa27`, before 1.0.0): since then, each publication also takes a
  step in `wait_for_newer()`'s waiter registry, with or without waiters; the Accumulator
  reuses the range check's timestamp maximum; `TimestampDecay` checks whether its division
  can overflow (from 1.0.0) and computes its exponentials in place; and the producer-thread
  check changed (below);
- **the `wait_for_newer` measurement** (`dcc1fa1`): the Accumulator's reuse of the
  timestamp maximum, `TimestampDecay`'s exponentials in place, and the producer-thread
  check;
- **the temporal-kernel gate** (its second run, `8131aca`, whose figures the kernels page
  gives): the producer-thread check.

The producer-thread check: `ingest()` now looks up the calling thread's object instead of
its thread ident, one lookup and one comparison per call.

## The gate, precisely

The gate's method was written down in `benchmarks/gate_preregistration.md` before the first
measurement and not changed afterwards. In summary:

**Workload.** Events are generated before timing, all in bounds, one event per microsecond of
event time, `t` increasing across the whole stream; each call gets distinct events. Per call,
`x` and `y` are drawn uniformly over the sensor, or, for the clustered distribution, around 8
fixed centres with a Gaussian spread of 2% of the smaller sensor side; `p` is uniform over
{0, 1}. Seeds are fixed per resolution, batch size and distribution, and every run records a
SHA-256 of the bytes it fed: the runs of a cell must agree.

**Kernel level** times `kernel.begin_call(state)` plus `kernel.accumulate(events, state,
watermark)` per call, through the public kernel classes. Validation, range and bounds
checks, reading and result checks are outside the timed region. Statistic per run: events
per call divided by the median call time.

**Engine level** times `Engine.ingest(events)` per call, publication included when the Engine
decides on one. The Engine's clock is replaced, for its construction and each call, by a
virtual clock that reads the arrival time of the call's events at 20M events/s (0.5 ms per
10k call, 5 ms per 100k call, 50 ms per 1M call), so the Engine's own interval check makes
exactly the publications a live 20M events/s stream would cause. Statistic per run: events
divided by the sum of the call times, in wall-clock time.

**Calls.** One warm-up call per run, discarded. Timed calls per run: 50 / 20 / 7 at 10k / 100k
/ 1M events at kernel level; at Engine level the same at 0 ms, and at 16 ms enough calls for
at least 10 publications (320 / 40 / 10). The garbage collector is disabled during timed
calls; each call is one `time.perf_counter_ns()` interval.

**Runs and classification.** 5 runs per cell and level, each a separate Python process. The
cell's value is the median of the 5 per-run values. At or above 22M events/s is a clear pass,
below 18M a clear miss; in between, 20 more runs decide (median at or above 20M and at least
18 of the 20 at or above 20M). No outlier removal; no run discarded or repeated. A cell
passes only if both levels pass; a runtime meets the gate only if all 150 cells pass on it.

**Validity.** A result counts only if every run passed its result checks (the final state
equals a reference built with other NumPy primitives; at Engine level also the counters, the
publication schedule, the sequence numbers and each publication's watermark), the runtime
matched, the fed bytes matched, and the document came from a clean working tree on the
reference machine.

**Characterisation, not classification.** Latency percentiles (nearest rank over the pooled
calls) and memory (`tracemalloc` in a separate untimed pass) are recorded for every cell but
never change a verdict.

**Conditions of the published run.** Commit `6a0fa27`, clean tree. Apple M4 (4P + 6E), 16 GB,
macOS 15.7.7, mains power, Low Power Mode off. Runtime A: CPython 3.11.14, NumPy 2.4.6.
Runtime B: CPython 3.14.2 free-threaded with the GIL disabled, NumPy 2.4.6 (pinned by hand so
NumPy is the same on both runtimes).

The gate run predates the later sleep guard (below) and entered a low-power state near the end
of the final 3.14t Engine-level run. A low-power state can only slow a run, so no pass verdict
depends on it; in that run's last ten cells, median call times were 0.89-1.08x those of the
other four runs. The recorded background environment was not controlled.

## The data file

[`benchmarks/results/gate_v1.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/gate_v1.csv)
holds the published gate run: **600 rows**, one per cell, level and runtime (150 cells × 2
levels × 2 runtimes). It was derived from the run's result and verdict documents, with local
paths and machine-specific details left out.

| column | meaning |
|---|---|
| `runtime` | `cpython-3.11.14` or `cpython-3.14.2t-gil-disabled` |
| `python`, `free_threaded_build`, `gil_enabled`, `numpy` | as recorded inside every run's own process |
| `commit` | the measured commit, `6a0fa27` |
| `kernel`, `kernel_parameters` | the kernel, and its parameter if it has one |
| `width`, `height` | the sensor size |
| `events_per_call`, `interval_ms` | the batch size and `snapshot_interval_ms` (at kernel level the interval doesn't apply; the 0 and 16 ms rows are separate measurements of the same work) |
| `distribution` | `uniform` or `clustered` |
| `level` | `kernel` or `engine` |
| `statistic` | `median_call` (kernel) or `sustained` (engine), as defined above |
| `events_per_s` | the cell's value: the median of the per-run statistics, events per second |
| `run_min_events_per_s`, `run_max_events_per_s` | the range of the 5 per-run values |
| `runs`, `timed_calls_per_run` | 5, and the timed calls in each run |
| `latency_samples`, `latency_p50_us`, `latency_p95_us`, `latency_p99_us`, `latency_max_us` | per-call time over the pooled timed calls of all runs, microseconds |
| `peak_temporary_mib` | the largest `tracemalloc` peak temporary allocation of one call, MiB |
| `stage1` | this level's classification: `clear_pass`, `borderline` or `clear_miss` |
| `level_result`, `cell_verdict` | this level's result, and the cell's verdict over both levels |

## The temporal-kernel gate

`StackedHistogram` and `VoxelGrid`, added in 1.1, have a gate of their own, written down in
`benchmarks/temporal_gate_preregistration.md` before the kernels existed. Its results are
summarised under [Throughput](../core/kernels.md#throughput) on the kernels page. It uses the
gate above, with these differences:

- **Cells.** 150 cells: five kernel configurations (`StackedHistogram` 5 x 10,000, 15 x 3,333
  and 10 x 5,000 µs; `VoxelGrid` 5 x 12,500 and 15 x 3,571 µs, each about 50 ms of event time
  per frame) x three resolutions (346x260, 640x480, 1280x720) x five batch and interval
  conditions (10k @ 16 ms; 100k and 1M @ 0 and 16 ms) x two distributions.
- **Event time.** Events advance at 20M events/s of event time (20 per µs), not one per µs,
  so event time and the Engine level's virtual arrival clock advance together and bins close
  as they would in a live 20M events/s stream.
- **Result checks.** At kernel level the state, read at the final watermark, must equal bit
  for bit a reference built with `np.bincount`; at Engine level the counters, publication
  schedule, sequence numbers and watermarks are checked as above, and the last published
  frame against the reference.
- **Plane clearing.** A separate instrumented pass, never a timed run, measures the share of
  kernel-level time spent clearing planes.

Statistics, classification, calls per run and the 5-run median are the gate's. The
reference machine and runtimes are the same: Apple M4 (4P + 6E), 16 GB, macOS 15.7.7;
CPython 3.11.14 and CPython 3.14.2 free-threaded with the GIL disabled, both with NumPy
2.4.6. Every run was inside the sleep guard (below).

To measure it on your machine, use the commands of
[Measuring your own machine](#measuring-your-own-machine) with `--suite temporal-gate` and the
targets `temporal-kernel` and `temporal-engine`. `benchmarks/temporal_gate.sh` runs the
gate's full sequence for both runtimes.

**The first run** (2026-10-03, commit `4d5a9f3`) is the gate's result for these kernels:
**not met**. On 3.11.14 137 cells passed and 13 missed; on 3.14.2t 135 passed and 15 missed.
Every cell passed at kernel level; every miss was at Engine level. The machine was prepared by
its owner. Recorded checks: every run on AC power with Low Power Mode off at its start and end;
every result document in the sleep guard, with no sleep and full wake at its start and end.

**The second run** (2026-10-03, commit `8131aca`) measured the kernels after performance
changes made following the first run. It is a later measurement recorded
alongside the first, not a replacement for it. On each runtime 138 cells passed and 12
missed, all at Engine level. Its environment control was weaker than the first run's:

- the machine was not prepared by its owner, and the "only one terminal open" condition
  was not verified;
- the power adapter had been unplugged for about an hour before the run started;
- the adapter lost power for 10 seconds during the fourth 3.11.14 kernel-level run. That run's
  measurements of 28 cells were not on mains power at their start or end, and the gate tool's
  own verdict for 3.11.14 classifies 27 cells INVALID (the 28th missed at Engine level). As
  the preregistration allows for an environment failure, those 28 measurements were taken
  again once, with the same code and workload, and merged by hand into the result in place of
  the failed ones. The published 3.11.14 figures include that merge.

**Data files.**
[`temporal_gate_run1.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/temporal_gate_run1.csv)
and
[`temporal_gate_run2.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/temporal_gate_run2.csv)
have 600 rows each and the columns of `gate_v1.csv` above, with `kernel_parameters` as
`bins=...;bin_us=...` and these additions:

| column | meaning |
|---|---|
| `stage2_events_per_s` | the median of the 20 stage-2 runs, for a borderline level; empty otherwise |
| `stage2_runs_at_20m` | how many of those 20 runs reached 20M events/s |
| `qualification` | empty, or what is unusual about the row: in run 2, the 28 rows measured again after the power loss |

## The `wait_for_newer` measurement

What [`wait_for_newer()`](../core/snapshots.md#what-waiting-costs), added in 1.1, costs the
producer. The method was written down in `benchmarks/wait_preregistration.md` before the
first measurement; it changed afterwards only through dated amendments, each committed
before the runs it affected.

- **Builds.** The Engine without `wait_for_newer` (commit `e63150e`, labelled `B`) and with
  it (commit `dcc1fa1`). `B'` is the first build again under a second label: the
  difference between `B` and `B'` measures the method's own variation (an A/A control).
- **Waiters.** Threads that loop on `wait_for_newer()` with no timeout and no other work,
  all registered before the timed calls: 0 (`F0`), 1, 4 or 8 (`F1`, `F4`, `F8`).
- **M2, the gate's Engine-level cells.** `event_count`, uniform events, 346x260, 640x480 and
  1280x720; 10k-event calls at interval 0, 100k-event calls at 0 and 16 ms. Timed as
  [the gate's Engine level](#the-gate-precisely): virtual 20M events/s arrival clock, one
  warm-up call, then events divided by the sum of the call times, and nearest-rank
  percentiles of the call times.
- **M1, one publication per call.** `ingest()` of a single event at interval 0, with exactly
  W waiters registered before each call: 50 warm-up and 2,000 timed calls, at 346x260 and
  1280x720. Statistic: the median and percentiles of the call times.
- **M3, waiting against polling.** The paced condition of the v1 observation study (below):
  1280x720, 100k-event batches released on the real clock at 20M events/s, 16 ms interval,
  5 s warm-up, a 10 s window. 1, 4 or 8 consumers each render every state they get, either
  waiting (`wait_for_newer(last, timeout=0.1)`, `WAIT1` to `WAIT8`) or polling
  (`snapshot()` every 16 ms, `POLL1` to `POLL8`). Its metrics are the study's: freshness and
  post-step delay per observation, the producer's step time and busy time per event.
- **The criterion,** for the M2 cells with no waiter only: a cell with the feature is
  distinguishably slower when the ratio of its median to the baseline's lies outside the
  largest `B'`/`B` ratio over all cells (no floor) **and** the two cells' per-run ranges
  don't overlap. Met if no cell is distinguishably slower. Everything with waiters, M1 and
  M3 is characterisation, with no pass or fail.
- **Runs.** 5 runs per cell and label, each its own process, in a seeded shuffle across
  runtimes, cells and labels; 720 runs. The cell's value is the median of its 5 per-run
  values.

**Conditions.** 2026-10-02, one session, all 720 runs valid at their first attempt. Apple
M4 (4P + 6E), 16 GB, macOS 15.7.7, on mains power throughout with Low Power Mode off, each
run inside the sleep guard. Runtime A: CPython 3.11.14, NumPy 2.4.6. Runtime B: CPython
3.14.2 free-threaded with the GIL disabled, NumPy 2.4.6. An earlier attempt the same day
lost mains power partway through; it is not used.

**Result.** No-waiter criterion met: none of the 72 comparisons (9 cells, 2 runtimes, 4
metrics) was distinguishable in either direction; the ratios of the medians were
0.94-1.08. That is a statement about this measurement's resolution, not a finding that
the feature costs nothing. The figures with waiters are on
[Snapshots and consumers](../core/snapshots.md#what-waiting-costs).

**Data file.**
[`benchmarks/results/wait_for_newer_r1.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/wait_for_newer_r1.csv)
has **600 rows**, one per runtime, experiment, cell, label and metric. M1 and M2 values were
recomputed from each run's recorded call times; M3 values are those the study's metric
functions recorded in each run.

| column | meaning |
|---|---|
| `runtime`, `python`, `free_threaded_build`, `gil_enabled`, `numpy` | as recorded inside every run's own process |
| `build`, `commit` | `baseline` (`e63150e`) or `feature` (`dcc1fa1`) |
| `experiment`, `label` | `M1`, `M2` or `M3`, and the label above |
| `waiters` | M1 and M2: the waiting threads, 0 to 8 |
| `consumers`, `consumer_loop` | M3: the rendering consumers, and whether they wait or poll |
| `kernel`, `width`, `height`, `events_per_call`, `interval_ms` | the cell |
| `metric`, `unit` | M1: `median_ns`, `p95_ns`, `p99_ns`. M2: `events_per_s`, `p50_ns`, `p95_ns`, `p99_ns`. M3: `step_p99_us`, `busy_ns_per_event`, `freshness_p50_ms`, `freshness_p95_ms`, `post_step_p50_ms`, `post_step_p95_ms`, `achieved_over_offered`, `coverage` (the share of publications each consumer saw) |
| `median`, `run_min`, `run_max`, `runs` | the median of the per-run values, their range, and the number of runs (5) |

The driver and analysis are `benchmarks/wait_driver.py` and `benchmarks/wait_analysis.py`.
They expect both builds as git worktrees and both runtimes in environments at the paths the
preregistration names (its sections 2, 3 and 10), so they are a record of how this was
measured rather than a tool for other machines.

## The v1 observation study

The study summarised under
[Consumers under load](performance.md#consumers-under-load-the-v1-observation-study). Its
protocol, `benchmarks/observation_preregistration.md`, was committed alone before any
measurement. It changed afterwards only through four numbered amendments, each fixing a
defect in the measuring harness and committed before the work it affected was re-run;
none changed a rule, parameter, threshold or configuration.

- **What was compared.** Frames2Py 1.0.0 from the PyPI wheel (an `Engine`, consumers
  polling `snapshot()` every 16 ms), the same Engine again under a second label as an A/A
  control, and hand-written designs built on the same `Accumulator` and fed the same
  batches: inline consumers; finished-frame queues per consumer (blocking with 4 slots,
  dropping the oldest, unbounded); a shared frame copied under a lock; a reference swapped
  under a lock; raw batches fanned out to consumers with their own Accumulators.
- **Conditions.** 1280x720 uniform synthetic events, 100k-event batches released on the
  real clock at 20M events/s, `event_count`, a 16 ms interval; 5 s warm-up, then a 10 s
  window. 0, 1 or 4 consumer threads, each rendering every state it got
  (`viewer.render()`), holding it through a 30 ms sleep, or running 250 ms of pure-Python
  arithmetic (calibrated on CPython 3.11.14). A second experiment recorded every event with
  the recorder (Blosc) on the producer's thread or on a thread fed by a queue, and read the
  file back.
- **Metrics.** *Freshness* is the time from the start of the producer step that took in the
  newest event in a consumer's state to the moment the consumer received that state. A run
  is *sustained* when the producer's lag behind the offered schedule grows by at most 1.6
  ms/s over the window and is at most 16 ms over its last second. The protocol's section 10
  defines every metric.
- **Runs.** 130 configurations, 5 valid runs each, in a seeded shuffle; each run its own
  process. A configuration's value is the median of its 5 runs. Comparisons use the A/A
  rule described for `wait_for_newer` above: the largest A/A ratio as the band, and
  non-overlapping per-run ranges.
- **Machine.** 2026-09-30, the Apple M4 (4P + 6E), 16 GB, macOS 15.7.7, Low Power Mode off,
  each run inside the sleep guard. CPython 3.11.14 and CPython 3.14.2 free-threaded with the
  GIL disabled, both with NumPy 2.4.6. Runs failing an environment check (mains power lost
  for part of the session, background load) were classified invalid and repeated; 38 of 688
  attempts.

**Data file.**
[`benchmarks/results/observation_v1_cells.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/observation_v1_cells.csv)
has one row per configuration (130), written by the study's analysis code. `arm` is the
design (`H` Frames2Py, `H'` its A/A control, `A` inline, `B`, `C`, `E` the blocking,
drop-oldest and unbounded queues, `F` copy under a lock, `G` reference swap under a lock,
`RB` raw fan-out, `EP-*` the recording experiment), `workload` the consumer (`W1` render,
`W3` sleep, `W5` pure Python, `none`), `n` the consumers, `runtime` `A` (3.11.14) or `B`
(3.14.2t), and `label` whether the producer was sustained in at least 4 of the 5 runs. Every
metric column holds the median of the 5 runs followed by their range, as `median [min,
max]`.

## Measuring your own machine

From a checkout, with the development environment (`uv sync`), writing results outside the
repository:

```sh
RESULTS="${TMPDIR:-/tmp}/frames2py-gate" && mkdir -p "$RESULTS"
uv run python -m benchmarks list --suite gate          # the 150 cells
uv run python -m benchmarks run --suite gate --target v1-kernel --out "$RESULTS/kernel.json"
uv run python -m benchmarks run --suite gate --target v1-engine --out "$RESULTS/engine.json"
uv run python -m benchmarks stage2 "$RESULTS/kernel.json" --out "$RESULTS/kernel_stage2.json"
uv run python -m benchmarks stage2 "$RESULTS/engine.json" --out "$RESULTS/engine_stage2.json"
uv run python -m benchmarks gate --kernel-level "$RESULTS/kernel.json" \
    --engine-level "$RESULTS/engine.json" \
    --kernel-stage2 "$RESULTS/kernel_stage2.json" \
    --engine-stage2 "$RESULTS/engine_stage2.json" \
    --out "$RESULTS/gate.json"
uv run python -m benchmarks report "$RESULTS/engine.json"   # a table of one document
```

A result document records the commit and the working tree's state, and the gate does not
classify a document from a dirty tree, so keep the results out of the checkout. `stage2`
measures only the borderline cells, and nothing if there are none. An existing result file
is not overwritten unless you pass `run` the `--overwrite` flag.

For the free-threaded runtime, the published run used a separate environment synced from
the lockfile for CPython 3.14.2t, with NumPy then pinned to 2.4.6, and invoked its Python
directly (`uv run` would re-sync NumPy):

```sh
UV_PROJECT_ENVIRONMENT="$RESULTS/venv-314t" uv sync --frozen --python 3.14.2+freethreaded
uv pip install --python "$RESULTS/venv-314t/bin/python" numpy==2.4.6
"$RESULTS/venv-314t/bin/python" -m benchmarks run --suite gate --target v1-kernel --out "$RESULTS/314t_kernel.json"
```

On any machine other than an Apple M4 with 16 GB, `gate` reports `NOT ON THE REFERENCE
MACHINE` instead of a verdict: the published gate is tied to that machine, and a result on
another machine is a measurement of that machine, not a re-run of the gate. The per-cell
numbers are still comparable with the CSV.

To measure a subset, `run` takes `--kernel`, `--resolution WxH`, `--batch-size`, `--interval`
and `--distribution`, each repeatable. The suites `run` and `list` take are `gate`,
`temporal-gate` and `prototype-baseline`. `adapters`, `recorder`, `viewer` and `replay` are
separate characterisation commands, not suites (`uv run python -m benchmarks --help`).

## Hygiene that mattered

On this machine, differences under about 10% between measurements of the same configuration
were not treated as meaningful, and the suite never compares single runs. Conditions that
changed results enough to matter:

- **System sleep.** The Mac used idle-sleeps after a minute and can also run in a low-power
  state; runs that overlapped either were slow. Since the published gate run, the
  runner holds its own idle-sleep assertion on macOS, refuses to start outside full wake,
  records any sleep during the run, and the gate treats cells from a run that slept as
  invalid.
- **Background load.** Other work on the machine competes for CPU. The published gate run
  recorded the machine's load but did not control it, so read small differences between cells
  as noise.
- **Separate processes.** Repeated runs inside one process measured slower than the first, so
  each run is its own process by default.
