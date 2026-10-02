# Temporal-kernel gate: preregistration

The 20M events/s target for `StackedHistogram` and `VoxelGrid`, measured by the v1 gate's method
(`benchmarks/gate_preregistration.md`) on a grid of its own. This file is committed before the kernels, the gate's
targets or any datapoint exist. After that it changes only by numbered, dated amendments (section 12), each committed
before the runs it affects. If the method turns out to be impossible to follow as written, measurement stops and the
problem is reported; the rules are not edited to fit.

## 1. Question and target

For every cell of section 4, at both levels of section 6: do the temporal kernels sustain 20,000,000 events/s on the
reference machine?

- 20M events/s is the preregistered target, at kernel level and at `Engine.ingest()` level.
- A cell that misses it neither passes silently nor blocks automatically: every miss goes to the project owner for a
  decision before release.
- The v1 gate's 150 cells are a separate gate and are not re-measured here.

Not asked, and not claimed from this data: throughput on another machine, runtime, workload or parameter set; anything
about the five v1 kernels.

## 2. Measured code

- The kernels and the gate targets don't exist when this file is committed. The measured commit is the one that adds
  them, plus any later commit, with no `src/` change after it, that is needed to run the method.
- Every result document records the commit it ran on and the working-tree state. A document from anything but a clean
  tree at the measured commit is not classified.
- Kernel- and engine-level documents of one runtime come from the same commit.

## 3. Machine, runtimes and environment

- **Machine:** the v1 gate's: Apple M4 (4 performance + 6 efficiency cores), 16 GiB, macOS 15.7.7, arm64.
- **Runtime A:** CPython 3.11.14 standard build, NumPy 2.4.6, the project environment from `uv.lock`.
- **Runtime B:** CPython 3.14.2 free-threaded build with the GIL disabled, NumPy 2.4.6.
  - Its environment is built as in the v1 gate preregistration, "Machine and runtimes". `PYTHON_GIL` is not set.
  - 3.14.2t has the `_PyRawMutex_LockSlow` race first fixed in 3.14.5 (CPython gh-148820). It stays the reference
    runtime, for comparability with the v1 gate. A crash is reported (section 10).
- **Run records:** every run records, from inside its own process, `sys.version`, `Py_GIL_DISABLED`,
  `sys._is_gil_enabled()`, the NumPy version, the platform and the architecture. A run whose record differs from its
  document's environment is invalid.
- **Separate populations:** the two runtimes are reported separately and never combined. Each is judged on its own.

**Environment controls.** The machine owner prepares the machine; nothing is measured on a machine in normal use.
- **Before the session:**
  - AC power, Low Power Mode off, lid open;
  - every application closed except one terminal (not an IDE's terminal);
  - Docker Desktop quit, with its daemon and `com.docker.vmnetd` not running;
  - no other benchmark, build, backup or software update running;
  - no development agent running other than, at most, an idle launching session;
  - the repository clean at the measured commit.
- **Every run is inside the power guard** (`benchmarks.power.hold_awake()`). It refuses to start outside full wake and
  records any sleep and the end state.
- **Recorded at the start and end of every result document:** power source, Low Power Mode and load average.
- **A run is invalid** (section 10) if it was not on AC power or had Low Power Mode on at its start or end, or if the
  power guard refused, recorded sleep, or ended outside full wake.

## 4. Cells

150 cells: 5 kernel configurations × 3 resolutions × 5 batch/interval conditions × 2 distributions.

| kernel | `bins` | `bin_us` | event time shown per frame |
|---|---|---|---|
| `StackedHistogram` | 5 | 10,000 | 50,000 µs (`bins · bin_us`) |
| `StackedHistogram` | 15 | 3,333 | 49,995 µs |
| `StackedHistogram` | 10 | 5,000 | 50,000 µs |
| `VoxelGrid` | 5 | 12,500 | 50,000 µs (`(bins - 1) · bin_us`) |
| `VoxelGrid` | 15 | 3,571 | 49,994 µs |

- **Resolutions** (width x height): 346x260, 640x480, 1280x720.
- **Batch / interval:** 10k @ 16 ms; 100k @ 0 ms; 100k @ 16 ms; 1M @ 0 ms; 1M @ 16 ms.
- **Distributions:** uniform, clustered.

No cell is added, removed, substituted or aggregated. A kernel configuration is part of the cell: the five
configurations are five cells per condition, never summarised as one.

## 5. Workload

`benchmarks/workloads.py`, as in the v1 gate, with one difference: the event-time rate is **20,000,000 events/s**
(`event_rate_hz = 20_000_000`), so event `i` of a stream has `t = i // 20`, 20 events per µs. The v1 gate used one event
per µs.
- **Same as the v1 gate:**
  - all events in bounds, in timestamp order;
  - per batch, `x`, then `y`, then `p` uniform over {0, 1}, from one generator seeded with the cell's seed;
  - the uniform and clustered definitions;
  - the seed rule `benchmarks.matrix.default_seed`, which doesn't depend on the kernel, its parameters or the interval.
- **Batches:** generated before timing and distinct per call. Every run records a SHA-256 over the bytes it fed; the
  runs of a cell must agree.
- **Event time per call:** a 10k call covers 500 µs of event time, a 100k call 5 ms, a 1M call 50 ms. This matches the
  Engine level's virtual arrival clock (section 6), so event time and arrival time advance together, as in live use.

Expected, not rules: a 1M call advances the watermark by 50,000 µs, across 5 bin edges (`bin_us` 10,000), 15 or 16
(3,333), 10 (5,000), 4 (12,500) or 14 or 15 (3,571). A 10k call advances it by 500 µs, so fewer than 1 call in 6 crosses
an edge. The workload is not
changed to cause or avoid crossings.

## 6. Timed regions

**Kernel level:**
- Per call, `kernel.begin_call(state)` then `kernel.accumulate(events, state, watermark)`, through the public kernel
  classes.
- The watermark passed is the running maximum timestamp up to and including that call, computed before timing.
- Outside the timed region: event generation, structural validation, the range and bounds checks, `read`, and result
  checking.
- The 0 ms and 16 ms cells are separate measurements of the same kernel work.

**Engine level:**
- Per call, `Engine.ingest(events)` on an Engine constructed with the cell's kernel configuration and
  `snapshot_interval_ms`.
- Publication happens inside that call when the Engine decides on it, and is timed. It includes reading every bin
  into the published frame.
- **Virtual clock:** as in the v1 gate. Call `k` (0-based, warmup included) reads `k * batch_size * 50` ns, the arrival
  time at 20M events/s.

## 7. Measurement

As the v1 gate's "Measurement":
- one discarded warmup call per run;
- timed calls 50 / 20 / 7 at kernel level (10k / 100k / 1M); at engine level 20 / 7 at 0 ms, and at 16 ms 320 / 40 /
  10, so that at least 10 publications fall in the timed calls;
- GC disabled while timing; one `time.perf_counter_ns()` interval per call; per-call hooks outside the intervals;
- 5 runs per cell, each run a separate Python process that measures every cell in gate order: kernel level first,
  then engine level; runtime A, then runtime B;
- no outlier removal, and no run discarded, repeated or replaced except as section 10 says.

## 8. Statistics and classification

The v1 gate's, unchanged:
- **Kernel level, per run:** `batch_size / median(call time)`.
- **Engine level, per run (sustained):** `(timed calls · batch_size) / sum(call times)`, publication calls included.
- **Per cell and level:** the median of the 5 per-run values.
- **Stage 1:** `>= 22,000,000` is a clear pass; `< 18,000,000` a clear miss; anything else borderline.
- **Stage 2:** for a borderline level only, 20 more runs, otherwise identical. The level passes if the median of the
  20 is `>= 20,000,000` and at least 18 of the 20 are.
- **What passes:** a cell passes only if both levels pass. Latency, memory and the plane-clearing share never change a
  classification.
- **Misses:** every cell that is not a pass is reported to the owner by name (section 1).

## 9. Characterisation (not the gate)

For every cell and level:
- **Latency:** p50, p95 and p99 of the pooled per-call times (nearest rank, with the sample count).
- **Memory:** in a separate, untimed pass in the last run, `tracemalloc` peak temporary bytes over 3 calls and retained
  growth, the kernel state's size in bytes (the arrays `init_state` returns), and the workers' peak RSS.
- **Share of time clearing planes:**
  - Kernel level only, in a separate instrumented pass in its own process, never in a timed gate run.
  - The kernel's plane-clearing step, which the implementation keeps in one private function named in the result
    document, is wrapped with a `perf_counter_ns` hook.
  - The share is the hooked time over the instrumented calls' total time, over the kernel-level call counts of
    section 7.
  - If the implementation has no such separable step, the share is reported as not measured, with the reason.

## 10. Validity and run outcomes

A level result of a cell is invalid, and so not a pass, if any of its runs:
- **fails its result checks:**
  - **Kernel level:** the state, read after the timed calls at the final watermark, must equal bit for bit a reference
    built with `np.bincount` over the events fed. Histogram counts are exact integers. Voxel numerators are float64
    sums of integers below 2^53, which are exact, converted by the kernel's output rule. The reference is checked
    against the test suite's independent per-event oracle on small inputs before measurement.
  - **Engine level:** `events_ingested` equals the events fed and `events_out_of_bounds` is 0. Each call's
    publication matches the schedule the contract gives for the virtual clock. Sequence numbers run 1, 2, 3, ..., and
    each publication's watermark is the maximum timestamp fed up to it. The last published frame equals the reference
    for everything fed up to it, at its watermark.
- ran on a different runtime from its document;
- fed different bytes from the other runs of the cell;
- broke an environment control of section 3.

The cell's result is also invalid if it has the wrong number of runs, or if its document isn't from a clean working
tree at the measured commit on the reference machine.

- **Environment failure:** a run invalid only because of an environment control may be re-run, at most twice per
  run. Every attempt is kept and reported.
- **Other failures:** a failed result check, a harness exception, or a worker that crashes or produces no result
  stops measurement. It is reported whatever the cause, and nothing is classified until the owner has reviewed it.

## 11. Harness validation and commands

Before the first evidentiary run, the targets are validated on a machine in normal use: every cell runs once at both
levels and its result checks pass. Validation timings are never looked at for the gate or used as evidence.

From the repository root, results in a gitignored directory `<results>`. `<py>` is `uv run python` for runtime A and
`<env>/bin/python` for runtime B. Suite and target names are fixed here: suite `temporal-gate`, targets
`temporal-kernel` and `temporal-engine`.

```sh
<py> -m benchmarks run --suite temporal-gate --target temporal-kernel --out <results>/<runtime>_kernel.json
<py> -m benchmarks run --suite temporal-gate --target temporal-engine --out <results>/<runtime>_engine.json
<py> -m benchmarks stage2 <results>/<runtime>_kernel.json --out <results>/<runtime>_kernel_stage2.json
<py> -m benchmarks stage2 <results>/<runtime>_engine.json --out <results>/<runtime>_engine_stage2.json
<py> -m benchmarks gate --kernel-level <results>/<runtime>_kernel.json \
    --engine-level <results>/<runtime>_engine.json \
    --kernel-stage2 <results>/<runtime>_kernel_stage2.json \
    --engine-stage2 <results>/<runtime>_engine_stage2.json \
    --out <results>/<runtime>_gate.json
```

`stage2` measures exactly the borderline cells of the document it is given, and nothing if there are none. Result
files are never overwritten.

## 12. Amendments

Numbered and dated, each naming what changed and why, committed before any run it affects. No amendment changes
sections 1, 4 or 8 after the first evidentiary run.

None yet.
