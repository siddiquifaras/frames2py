# Performance

Every figure on this page was measured on **one machine**: an Apple M4 (4 performance and 6
efficiency cores), 16 GB, macOS 15.7.7, on mains power with Low Power Mode off. No other
hardware has been measured, and CI measures no throughput. Treat the numbers as what this
machine did under these conditions, not as a rate for yours. How the gate was measured is on
[Benchmark methodology](methodology.md); every gate cell is in
[`benchmarks/results/gate_v1.csv`](https://github.com/siddiquifaras/frames2py/blob/main/benchmarks/results/gate_v1.csv).

## The v1 performance gate

The gate asks whether the five kernels and `Engine.ingest()` sustain at least **20 million
events per second** across a fixed matrix of conditions:

| dimension | values |
|---|---|
| kernels | `event_count`, `polarity`, `time_surface`, `exp_decay` (`decay=0.95`), `timestamp_decay` (`tau_us=10000`) |
| resolutions | 346x260, 640x480, 1280x720 |
| events per call @ publication interval | 10k @ 16 ms, 100k @ 0 ms, 100k @ 16 ms, 1M @ 0 ms, 1M @ 16 ms |
| event distributions | uniform over the sensor; clustered (8 Gaussian clusters) |

5 × 3 × 5 × 2 = **150 cells**. A cell passes only if both of its levels reach 20M events/s:

- **kernel level**: the kernel's own accumulation (`begin_call` + `accumulate`) on in-bounds
  events, without validation, checks or publication; statistic: events per call / median call
  time;
- **Engine level**: `Engine.ingest()`, validation, checks and publication included, with the
  Engine's clock driven as if events arrived at exactly 20M events/s, so publications happen
  on the real schedule; statistic: events / total time inside the timed calls.

Measured on commit `6a0fa27` with two runtimes, each gated on its own:

- **CPython 3.11.14**, NumPy 2.4.6;
- **CPython 3.14.2t**, free-threaded, GIL disabled (checked in every run), NumPy 2.4.6.

**Result: all 150 cells passed at both levels on both runtimes.** Every cell was a clear pass
(at or above 22M events/s on the median of 5 runs), so no cell needed the borderline stage.
The lowest result per kernel, in millions of events per second:

| kernel | 3.11.14 kernel level | 3.11.14 Engine level | 3.14.2t kernel level | 3.14.2t Engine level |
|---|---|---|---|---|
| `event_count` | 393.7 (HD 1M @ 16 u) | 176.0 (HD 100k @ 0 c) | 401.1 (HD 1M @ 16 u) | 170.4 (HD 100k @ 0 c) |
| `polarity` | 258.1 (HD 10k @ 16 u) | 110.2 (HD 100k @ 0 c) | 249.2 (HD 10k @ 16 u) | 108.1 (HD 100k @ 0 c) |
| `time_surface` | 252.2 (HD 10k @ 16 u) | 119.0 (HD 100k @ 0 u) | 73.9 (HD 10k @ 16 u) | 104.8 (HD 10k @ 16 u) |
| `exp_decay` | 326.3 (346 10k @ 16 c) | 127.9 (HD 100k @ 0 u) | 351.9 (HD 10k @ 16 u) | 129.5 (HD 100k @ 0 u) |
| `timestamp_decay` | 162.2 (HD 1M @ 0 u) | 97.4 (HD 100k @ 0 u) | 65.6 (HD 10k @ 16 u) | 80.4 (HD 10k @ 16 c) |

HD = 1280x720, 346 = 346x260; u / c = uniform / clustered.

These are **back-to-back** figures: the benchmark calls as fast as it can, so they show how
much headroom there is above 20M events/s, not the rate of a live stream. The live figures
are below.

**On the lockfile's NumPy.** The gate used NumPy 2.4.6 on both runtimes. The five 3.14t cells
with the lowest gate results were re-measured with NumPy 2.5.3 (what the lockfile installs on
Python 3.12 and later) against a same-session 2.4.6 control: 0.93 to 1.01 times the control,
none below 22M events/s. The other cells were not re-measured on 2.5.3.

### Latency and memory

Per-call latency is recorded for every cell and level (p50, p95, p99 and maximum over the
pooled calls of the 5 runs, in the CSV). For example, `event_count` at 1280x720, 10k events
per call, 16 ms, uniform, Engine level: p50 / p95 / p99 of 40 / 63 / 238 µs on 3.11.14 and
41 / 55 / 228 µs on 3.14.2t, over 1,600 calls. The 1M @ 0 ms cells have 35 calls, so their
p99 is their maximum. The largest single calls were isolated outliers: one 53 ms call at
3.11.14 Engine level (the next largest was 10.5 ms) and one 108 ms call at 3.14.2t kernel
level (the next largest was 9.3 ms). Their cause is not established; read single-call
maxima, and the p99 of the 35-call cells, as single calls.

Memory, measured with `tracemalloc` in a separate untimed pass: the peak temporary
allocation of one call was at most 23 MiB (`timestamp_decay` with 1M events); at 1280x720,
Engine cells ranged from 0.1 to 23 MiB. Memory retained across calls grew by at most 7.03 MiB,
one published frame.

## Live: fed at 20M events/s

The same machine and runtimes, 1280x720 uniform events, all five kernels at the five gate
conditions, on the real clock: the producer released each batch at the moment a 20M events/s
stream would deliver it, for 10 s.

- The Engine kept up in every configuration: 20.00M to 20.07M events/s achieved.
- CPU: 43 to 53% of one core at 10k and 100k events per call, 25 to 30% at 1M.
- Memory (physical footprint): 52 to 71 MiB at 10k and 100k, 120 to 128 MiB at 1M, flat within
  each run. Over 60 s (`timestamp_decay`, 100k @ 0 ms) there was no upward trend on 3.11.14;
  on 3.14.2t it rose from 57.4 to 60.6 MiB, flattening.
- **The live margin is smaller than the gate margin.** Time spent inside `ingest()` per event
  was higher when paced than back to back: busy throughput paced was 0.17 to 0.57 times the
  back-to-back figure for the same configuration (paced 43M to 98M events/s, back to back 104M
  to 300M). The lower paced figure went with time on efficiency cores and a lower cycle rate;
  the cause is not established.

A separate paced check (`timestamp_decay`, 1280x720, 100k events per call @ 16 ms, 1% of
events out of bounds, 30 s after a 5 s warm-up) measured how far completion lagged behind
each batch's arrival: final lag 2.80 ms (3.11.14) and 2.69 ms (3.14.2t), maximum 7.39 and
5.99 ms, with no upward trend (slopes of -0.61 and -0.02 µs/s). That check's pass thresholds
were for that check only; they are not a latency guarantee.

### With a consumer

The same live setup with one reader thread that read every new snapshot at the publication
cadence and touched every pixel (`frame.max()`), 3 runs per runtime:

- the producer achieved 19.8M to 20.3M events/s in every condition, with and without the
  reader;
- the producer's busy throughput with the reader was a median 0.99 times (3.11.14) and 1.01
  times (3.14.2t) the figure without it; it was materially lower (0.90) in one cell per
  runtime, both at 10k @ 16 ms.

The reader-free measurement always ran first in each process, so run order and the reader's
effect can't be fully separated. The [viewer's](#viewer) effect on a live producer is below.

## Workload characterisation

Beyond the gate matrix, on both runtimes, at 346x260 and 1280x720, 3 runs per configuration,
with the gate's method. Every run passed its correctness checks, and nothing measured below
20M events/s.

- **Out-of-bounds events** (0.1%, 1% and 10% of each call): counts, watermark and
  representation were correct in all 1,440 runs. Throughput fell to a median 0.36 to 0.42
  times the all-in-bounds figure, about the same at 0.1% as at 10%, because the mask and
  compaction run as soon as any event is out. Lowest: 40.0M (3.11.14) and 30.8M (3.14.2t)
  events/s, both `timestamp_decay` at 1280x720 with 10k events per call and 10% out, at
  Accumulator level.
- **Hot pixels:** 90% of events on one pixel lowered throughput to a median 0.68 times
  (kernel level) and 0.79 to 0.82 times (Engine level) the uniform figure, lowest 0.47 times;
  50% on 16 pixels made no material difference. Lowest hot-pixel result: 76.1M events/s.
- **Arrival order** (the same events chronological, shuffled within calls, calls permuted,
  globally shuffled): throughput within a median 1.00 times chronological, lowest 87.5M
  events/s. Outputs: `event_count`, `polarity` and `time_surface` identical in every order,
  `timestamp_decay` bit-identical, `exp_decay` identical when events are shuffled within
  calls and different when they move between calls, as its per-call decay defines.

## Adapters

Decoding eight real recordings, 5 runs per recording, each a separate process, medians;
files read from the page cache; decode is `open()` plus iteration with the reader's own
boundaries. CPython 3.11.14 with NumPy 2.4.6, and CPython 3.14.2t (GIL disabled) with NumPy
2.5.3; dv-processing 2.0.4, h5py 3.16.0 (HDF5 2.0.0), hdf5plugin 7.1.0. Millions of events per
second, and how many times faster than the recording's own duration:

| recording | events | 3.11.14 decode | real time | 3.14.2t decode | real time |
|---|---|---|---|---|---|
| `sparklers.raw` (EVT 2.0) | 521,252 | 121.8 | 22x | 124.8 | 23x |
| `200_jets_at_200hz.raw` (EVT 2.0) | 407,365 | 36.8 | 163x | 67.7 | 300x |
| `faery_evt3.raw` (EVT 3.0) | 1,218,618 | 18.3 | 161x | 17.2 | 152x |
| `active_marker.raw` (EVT 3.0) | 22,316,758 | 20.7 | 29x | 19.8 | 28x |
| `dvp_sample_data.aedat4` | 9,193 | 5.8 | 2,390x | 5.3 | 2,212x |
| `dvp_test-minimal.aedat4` | 255,283 | 54.9 | 581x | 51.9 | 549x |
| `faery_davis346.aedat4` | 78,830 | 25.3 | 757x | 23.0 | 688x |
| `dsec_thun_01_a_events_left.h5` | 131,482,728 | 88.5 | 6.6x | 87.5 | 6.5x |

- **Decoding costs more than ingesting.** With `event_count` at 16 ms, the Engine ingested the
  decoded EVT and HDF5 batches at 207M to 281M events/s, so the adapter took 64 to 94% of the
  end-to-end time (92 to 94% for EVT 3.0).
- **EVT 3.0** decoded at about 17M to 21M events/s here: below 20M events/s on
  `faery_evt3.raw` (both runtimes) and `active_marker.raw` (3.14.2t), and still 28 or more
  times faster than both recordings were recorded. Adapter throughput is characterisation,
  not part of the 20M events/s gate.
- **Files of one format decode at very different rates** (EVT 2.0: 37M to 125M events/s); why
  was not investigated.
- **`200_jets_at_200hz.raw`** decoded 1.8 times faster on 3.14.2t, and the difference follows
  the NumPy version, not the runtime: 35M to 37M events/s with NumPy 2.4.6 on CPython 3.11.14,
  3.14.2 and 3.14.2t, 70M with NumPy 2.5.3 on both 3.14 builds.
- **Small AEDAT 4.0 packets ingest slowly.** The three AEDAT 4.0 files have packets of a median
  36, 330 and 944 events, and with one array per packet the Engine ingested them at 9M to 98M
  events/s. With `batch_size=10_000` the same events ingested at 58M to 197M events/s.
- **Memory:** at most 17 MiB traced while decoding the EVT files, 35 MiB for the DSEC file.
- **Blosc threads (HDF5).** On the DSEC file, with CPython 3.11.14, `BLOSC_NTHREADS=4` decoded
  at 137M to 139M events/s instead of 89M to 93M: a process that only decoded the file took
  0.91 s of wall time instead of 1.42 s, and 1.61 s of CPU time instead of 1.42 s. One file,
  one machine, one thread count.

Reproduce with `uv run python -m benchmarks adapters --recording NAME --out FILE` after
downloading the recording (`uv run python -m tests.recordings download NAME`), from a
checkout of the repository.

## Recorder

`recorder.open()` to `close()`, the file left in the page cache (not fsynced), the first 20
million events of each recording (all 521,252 of `sparklers.raw`), 100,000 events per
`write()`, 5 runs, medians; CPython 3.11.14, NumPy 2.4.6, h5py 3.16.0, hdf5plugin 7.1.0.
CPython 3.14.2t (GIL disabled, NumPy 2.5.3) was within 2% in every row.

| recording, its own event rate | `None` | `"gzip"` | `"blosc"` | bytes per event (none / gzip / blosc) |
|---|---|---|---|---|
| Prophesee `sparklers`, 5.4M events/s | 112M events/s | 10.1M | 62.3M | 13.1 / 1.9 / 3.1 |
| Prophesee `active_marker`, 0.7M events/s | 134M | 10.2M | 74.6M | 13.0 / 2.1 / 3.0 |
| DSEC `thun_01_a` left, 13.8M events/s | 135M | 9.7M | 64.2M | 13.0 / 2.0 / 3.3 |

- At 10,000 events per call Blosc wrote 54M to 64M events/s, at about a million 72M to 100M.
- `write()` and `Engine.ingest()` (`event_count`) on the same thread, 100,000 events per call:
  51M to 59M events/s with Blosc, 9.3M to 9.8M with gzip, 82M to 90M uncompressed.
- gzip did not keep up with the DSEC recording's own rate (about 0.7 times); Blosc and no
  compression kept up with all three.
- A `write()` that completes chunks compresses them before returning: at 100,000 events per
  call its 99th-percentile time was about 2 ms with Blosc and 13 ms with gzip, and 12 to 13
  ms and 103 to 107 ms at about a million.
- Peak traced memory while recording: 1.4 MiB.

Reproduce with `uv run python -m benchmarks recorder`.

## Viewer

`render()` at 1280x720 on one Engine snapshot (one 16 ms window of a 20M events/s stream for
the windowed kernels, 80 ms of it for the running ones), 5 runs, medians, CPython 3.11.14 and
NumPy 2.4.6; 3.14.2t (NumPy 2.5.3) was 3 to 13% slower. The time of the automatic scale is in
parentheses:

| kernel | uniform events | clustered events |
|---|---|---|
| `event_count` | 6.8 ms (2.8) | 4.4 ms (0.6) |
| `polarity` | 9.5 ms (3.4) | 7.2 ms (1.1) |
| `time_surface` | 7.6 ms | 6.4 ms |
| `exp_decay` | 12.1 ms (8.3) | 4.5 ms (0.8) |
| `timestamp_decay` | 12.0 ms (8.3) | 4.6 ms (0.8) |

The automatic scale takes the 99th percentile of the nonzero values, so it costs in
proportion to how many there are; passing `scale=` skips it. At 640x480 a render took 1.4 to
4.1 ms, at 346x260 0.4 to 1.8 ms. A render allocates up to 28 MiB of temporaries at 1280x720.

**Effect on a live producer.** With the viewer's loop rendering every publication at 16 ms on
the main thread (no window), and a producer thread feeding the Engine at 20M events/s
(1280x720, `event_count`, `polarity`, `time_surface`, `timestamp_decay`, 10,000 and 100,000
events per call), the producer kept 20M events/s in every run and the viewer presented 61 to
62 frames a second. The producer's busy throughput was *higher* with the viewer running,
1.2 to 1.9 times on 3.11.14 and 1.5 to 3.6 times on 3.14.2t. The likely reason is the machine
keeping a busier process on faster cores, as in the paced measurement above; that was not
measured. What the measurement shows is that the viewer didn't slow the producer in these
runs, not that it speeds producers up.

Reproduce with `uv run python -m benchmarks viewer`.

## Paced replay

`paced()` over 10,000-event batches of the first 5 s of `active_marker.raw` (at 1x and 4x)
and of `sparklers.raw` (at 0.1x), real clock, 5 runs, medians: no batch was early; the replay
took 1.0006 to 1.0017 times the requested time; batches were late by 1.1 to 3.3 ms at the
median and at most 5.1 ms. The lateness is `time.sleep()` overshooting, which on this machine
grew with the requested sleep (about 7 ms over a 14 ms sleep). With a simulated clock, every
batch was yielded exactly at its due time, including across an inserted backward jump and
forward spike.

Reproduce with `uv run python -m benchmarks replay`.

## Caveats

- **One machine.** Everything here is one Apple M4. No x86_64, Linux or edge-device
  throughput has been measured. CI runs on Linux and macOS but measures no throughput.
- **Synthetic core workloads.** The gate and its characterisation use generated events. Real
  recordings were measured through the adapters, recorder, viewer and replay.
- **An unexplained low mode on 3.14t.** On 3.14.2t, some kernel-level runs of the short 10k
  uniform cells for `timestamp_decay` and `time_surface` were bimodal: about 49M to 77M events/s
  against 207M to 276M within one cell. The gate's lowest 3.14t figures come from that low
  mode. It did not recur in 30 later runs at default priority, and recurred once in a later
  characterisation run; forcing the process onto the efficiency cores gives 24M to 40M
  events/s. The cause is not established.
- **Live versus back to back.** The live busy throughput was 0.17 to 0.57 times the
  back-to-back figure, cause not established; the live margin above 20M events/s is smaller
  than the gate margin.
- **Small calls.** Every `ingest()` call has a fixed cost, so throughput falls with very small
  calls; the gate's smallest condition is 10k events per call. Adapters with small native
  batches (AEDAT 4.0) should be re-batched.
