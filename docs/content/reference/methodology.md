# Benchmark methodology

How the figures on [Performance](performance.md) were measured, what the published data
file contains, and how to measure your own machine. The benchmark suite lives in the
repository's `benchmarks/` directory. It is not part of the installed package, so every
command below runs from a checkout.

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
macOS 15.7.7, mains power, Low Power Mode off, Docker not running, the benchmark holding its
own idle-sleep assertion. Runtime A: CPython 3.11.14, NumPy 2.4.6. Runtime B: CPython 3.14.2
free-threaded with the GIL disabled, NumPy 2.4.6 (pinned by hand so NumPy is the same on both
runtimes).

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
and `--distribution`, each repeatable. The other suites are `adapters`, `recorder`, `viewer`
and `replay` (`uv run python -m benchmarks --help`).

## Hygiene that mattered

On this machine, differences under about 10% between measurements of the same configuration
were not treated as meaningful, and the suite never compares single runs. Conditions that
changed results enough to matter:

- **System sleep.** The Mac used idle-sleeps after a minute and can run in a low-power
  "dark wake" state; runs that overlapped either were slow. The runner holds its own idle-sleep assertion
  on macOS, refuses to start outside full wake, records any sleep during the run, and the gate
  treats cells from a run that slept as invalid.
- **Background load.** Heavy background processes (container runtimes, editors' helper
  processes) were stopped before measuring.
- **Separate processes.** Repeated runs inside one process measured slower than the first, so
  each run is its own process by default.
