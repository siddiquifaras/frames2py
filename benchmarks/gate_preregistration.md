# 20M events/s gate: preregistration

Written before any gate datapoint was collected. Nothing here changes once the first
gate datapoint exists. If the method turns out to be impossible to follow as written,
measurement stops and the problem is reported; the rules are not edited to fit.

## Base and measured code

- Base commit: `44d1109faf33d70edfd046f2fc27c57db6eacd92` (`feat/core-rebuild`).
- Measured code: the commit that adds this file. It can't name itself; every result
  document records the commit it ran on and the working-tree state, and a document from
  anything but a clean tree is not classified.
- No `src/` change is part of this commit. The kernels and Engine measured are the v1
  code at the base commit.

## Machine and runtimes

- Apple M4 (4 performance + 6 efficiency cores), 16 GiB, macOS 15.7.7 (24G720), arm64.
- Mains power, Low Power Mode off, Docker not running. Power, Low Power Mode and load
  average are recorded at the start and end of every result document.
- Runtime A: CPython 3.11.14 (standard build), NumPy 2.4.6. The project environment from
  `uv.lock`, invoked with `uv run python`.
- Runtime B: CPython 3.14.2 free-threaded build with the GIL disabled, NumPy 2.4.6. A
  separate environment synced from `uv.lock` with
  `UV_PROJECT_ENVIRONMENT=<env> uv sync --frozen --python 3.14.2+freethreaded`, then
  `uv pip install --python <env>/bin/python numpy==2.4.6`. The lockfile resolves NumPy 2.5.3
  for Python 3.12 and later; 2.4.6 holds NumPy constant across the two runtimes. Invoked as
  `<env>/bin/python` directly, because `uv run` would re-sync NumPy. `PYTHON_GIL` is not
  set.
- Every run records, from inside its own process: `sys.version`, whether the build is
  free-threaded (`Py_GIL_DISABLED`), `sys._is_gil_enabled()`, the NumPy version, the
  platform and the architecture. A run whose record differs from its document's
  environment is invalid.
- The two runtimes are separate populations. They are reported separately and never
  combined, and each must meet the gate on its own.

## Cells

`python -m benchmarks list --suite gate`: 150 cells.

- Kernels: `event_count`, `polarity`, `time_surface`, `exp_decay`, `timestamp_decay`.
- Resolutions (width x height): 346x260, 640x480, 1280x720.
- Batch / interval: 10k @ 16 ms; 100k @ 0 ms; 100k @ 16 ms; 1M @ 0 ms; 1M @ 16 ms.
- Distributions: uniform, clustered.

No cell is added, removed, substituted or aggregated.

## Workload (`benchmarks/workloads.py`, unchanged)

- Events are `EVENT_DTYPE`, all in bounds, in timestamp order, 1 event per µs of event
  time: event `i` of a stream has `t = i`. Batches continue the stream, so batch `b` of size
  `n` holds `t = b*n ... b*n + n - 1`.
- Per batch, from one generator seeded with the cell's seed: `x`, then `y`, then `p`
  uniform over {0, 1}.
- Uniform: `x` uniform over `[0, width)`, `y` over `[0, height)`.
- Clustered: 8 centres, drawn once per stream, uniformly within a margin of
  `min(3 sigma, (extent - 1) / 2)` from each edge, from a separate generator seeded with
  `[seed, 1]`. Each event picks a centre with equal probability and adds a Gaussian offset,
  `sigma = 0.02 * min(width, height)`, rounded and clipped to the sensor. The centres
  never change during a stream.
- Seed: `7 * width + batch_size + offset`, offset 0 for uniform and 1 for clustered. The
  seed doesn't depend on the kernel or the interval.

| resolution | 10k uniform / clustered | 100k | 1M |
|---|---|---|---|
| 346x260 | 12422 / 12423 | 102422 / 102423 | 1002422 / 1002423 |
| 640x480 | 14480 / 14481 | 104480 / 104481 | 1004480 / 1004481 |
| 1280x720 | 18960 / 18961 | 108960 / 108961 | 1008960 / 1008961 |

- Batches are generated before timing and are distinct per call. Every run records a
  SHA-256 over the bytes of the batches it fed; the runs of a cell must agree or the cell
  is invalid.

## Kernel parameters

`ExpDecay(decay=0.95)` and `TimestampDecay(tau_us=10000.0)`. The other three kernels have
no parameters.

Expected, not rules: `exp_decay` first folds its scale at about the 12,960th call at this
decay (the smallest `k` with `0.95**k < 2**-959`), so no gate run folds. `timestamp_decay`
rebases once its watermark is more than 665 * tau_us = 6.65 s past its reference time; in
1M-event cells that happens once per run, at the call with index 7 (0-based, warmup
included), which is a timed call. The 10k and 100k cells don't rebase. Both are characterised separately; the gate workload is not changed to
cause or avoid them.

## Timed regions

**Kernel level** (target `v1-kernel`): per call, `kernel.begin_call(state)` then
`kernel.accumulate(events, state, watermark)`, through the public kernel classes. The
watermark passed is the running maximum timestamp up to and including that call,
computed before timing starts. Event generation, structural validation, the range and
bounds checks, `read`, window closure and result checking are outside the timed region.
The interval doesn't apply at this level: the 0 ms and 16 ms cells are separate
measurements of the same kernel work, and neither is merged into the other.

**Engine level** (target `v1-engine`): per call, `Engine.ingest(events)`, on an Engine
constructed with the cell's kernel and `snapshot_interval_ms`. Publication, when the
Engine decides on it, happens inside that call and is timed.

**Virtual clock** (engine level, every cell): the Engine reads `time.monotonic_ns()`
through the `time` module inside `frames2py._engine`. For the Engine's construction and
for each call, the benchmark puts a virtual clock object in that module's place and puts
the real module back straight after. Call `k` (0-based, warmup included) reads
`k * batch_size * 50` ns, the arrival time at 20M events/s: 0.5 ms per 10k call, 5 ms per
100k call, 50 ms per 1M call. The Engine's own interval check decides every publication;
nothing forces or suppresses one. Inside the timed call this costs one Python method call
in place of a C function call per `ingest()`.

## Measurement

- Warmup: 1 call per run, discarded. At engine level it is the first `ingest()` and
  publishes.
- Timed calls per run:
  - kernel level: 50 / 20 / 7 at 10k / 100k / 1M
  - engine level, 0 ms: 20 at 100k, 7 at 1M
  - engine level, 16 ms: at least the kernel-level count, raised to the smallest number
    that gives at least 10 publications during the timed calls under the virtual clock:
    320 at 10k (a publication every 32 calls), 40 at 100k (every 4), 10 at 1M (every call)
- The garbage collector is disabled during the timed calls. Each call is one
  `time.perf_counter_ns()` interval. Per-call hooks (clock advance, publication
  observation) run outside those intervals.
- Primary stage: 5 runs per cell, each run a separate Python process that measures every
  cell in gate order. Kernel level first, then engine level, runtime A then runtime B.
- No outlier removal. No run is discarded, repeated or replaced. If a worker process
  fails, measurement stops and it is reported.

## Statistics

- Kernel level, per run: `batch_size / median(call time)`.
- Engine level, per run (sustained): `(timed calls * batch_size) / sum(call times)`, in
  wall-clock time, publication calls included. Virtual time is never a denominator.
- Per cell and level: the median of the per-run values.
- The gate command recomputes everything from the raw per-call times in the result
  documents.

## Classification

The requirement is >= 20,000,000 events/s at both levels. The band below is uncertainty
around it, not a different requirement.

Stage 1, per cell and level, on the median of the 5 per-run values:

- `>= 22,000,000`: clear pass
- `< 18,000,000`: clear miss
- otherwise: borderline

Stage 2, only for borderline results, on the same level only: 20 more runs, each a
separate process, with identical code, commit, runtime, workload, seeds, kernel
parameters, Engine configuration, timed calls and method. The level passes if the median
of the 20 per-run values is >= 20,000,000 *and* at least 18 of the 20 are >= 20,000,000.
Otherwise it is not met. There is no stage 3. Clear passes and clear misses are never
re-run. Latency percentiles and memory never change a classification.

A cell passes only if both levels pass. A runtime meets the gate only if all 150 cells
pass on it.

## Validity

A cell's level result is invalid, and so not a pass, if any of its runs:

- fails its result checks: the kernel-level state, read after the timed calls, must equal
  a reference built with other NumPy primitives (`bincount`, `lexsort`, an eager decay
  loop): exactly for the count and time kernels, within 3e-7 relative for the decay
  kernels. At engine level: `events_ingested` equals the events fed,
  `events_out_of_bounds` is 0, each call's publication matches the schedule the contract
  gives for the virtual clock, publication sequence numbers run 1, 2, 3, ..., each
  publication's watermark is the maximum timestamp fed up to it, and the last published
  frame equals the reference for its window (windowed kernels) or for everything fed up
  to it (running kernels);
- ran on a different runtime from its document;
- fed different bytes from the other runs of the cell;

or if the cell has the wrong number of runs, or its document isn't from a clean working
tree at the measured commit on the reference machine. Kernel- and engine-level documents
must come from the same commit and runtime.

## Characterisation (not the gate)

For every cell and level: p50, p95 and p99 of the pooled per-call times (nearest rank,
reported with the sample count; with 35 samples, p99 is the maximum). Memory in a
separate, untimed pass in the last run: `tracemalloc` peak temporary bytes over 3 calls
and retained growth, plus the peak RSS of the worker processes. None of these changes a
classification.

## Commands

From the repository root, results written to a gitignored directory `<results>`. `<py>`
is `uv run python` for runtime A and `<env>/bin/python` for runtime B.

```sh
<py> -m benchmarks run --suite gate --target v1-kernel --out <results>/<runtime>_kernel.json
<py> -m benchmarks run --suite gate --target v1-engine --out <results>/<runtime>_engine.json
<py> -m benchmarks stage2 <results>/<runtime>_kernel.json --out <results>/<runtime>_kernel_stage2.json
<py> -m benchmarks stage2 <results>/<runtime>_engine.json --out <results>/<runtime>_engine_stage2.json
<py> -m benchmarks gate --kernel-level <results>/<runtime>_kernel.json \
    --engine-level <results>/<runtime>_engine.json \
    --kernel-stage2 <results>/<runtime>_kernel_stage2.json \
    --engine-stage2 <results>/<runtime>_engine_stage2.json \
    --out <results>/<runtime>_gate.json
```

`stage2` measures exactly the borderline cells of the document it is given, and nothing if
there are none. Result files are never overwritten.
