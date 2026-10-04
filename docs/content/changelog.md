# Changelog

Each release has a section headed by its version, exactly as `pyproject.toml` declares it.
The release workflow publishes that section, unchanged, as the release's notes on GitHub.

<!--
Adding a release: put its section above the newest one, headed "## <version>" and nothing
else on the line (for example "## 1.0.0"). Write links as absolute URLs, because the section
is also the GitHub Release text. The tests fail when pyproject.toml's version has no section.
-->

## 1.1.0

A release that adds to 1.0 without changing what 1.0 code does: waiting for a newer
snapshot, two temporal representations for event-vision models, frames at fixed steps of
event time, and a tested recipe for handing snapshots to PyTorch.

```sh
pip install --upgrade frames2py
```

### New

- **`Engine.wait_for_newer(sequence, *, timeout=None)`** blocks until a snapshot newer than
  `sequence` is published and returns the latest one, or `None` on timeout. It returns the
  latest state, not every publication; `stop()` and `reset()` wake nobody; calling it on the
  producer's thread raises `RuntimeError`. Each publication releases the waiters registered
  before it and never waits for one. With no waiter, a preregistered measurement on one
  machine found no distinguishable cost to `ingest()`; waiters do add work to each
  publication. Details and figures:
  [Waiting for a newer snapshot](https://siddiquifaras.github.io/frames2py/core/snapshots/#waiting-for-a-newer-snapshot).
- **Two temporal kernels**, exported from `frames2py` and `frames2py.kernels`:
  - `StackedHistogram(bins=..., bin_us=...)`: `(2, bins, H, W)` uint32, events per polarity
    and time bin, as RVT consumes before its clip;
  - `VoxelGrid(bins=..., bin_us=...)`: `(bins, H, W)` float32, signed events split linearly
    between time knots, as in E2VID and E-RAFT.

  Both use bins on an absolute event-time grid and show only completed bins; they count
  exactly in integers, give the same frame bit for bit whatever the order and batching of
  the events, and apply no normalisation. They did not reach 20M events/s through the Engine
  in every cell of their performance gate. Details:
  [Temporal kernels](https://siddiquifaras.github.io/frames2py/core/kernels/#temporal-kernels), the
  [semantics table](https://siddiquifaras.github.io/frames2py/core/temporal-semantics/) and
  [Throughput](https://siddiquifaras.github.io/frames2py/core/kernels/#throughput).
- **`frames2py.replay.windows(batches, sensor_size, kernel, *, every_us)`** yields a frame
  every `every_us` µs of event time from a recording's batches, deterministically and
  independently of how the events are batched. It refuses `ExpDecay`, whose result depends
  on call boundaries. Details: [Frames in event time](https://siddiquifaras.github.io/frames2py/data/replay/#frames-in-event-time).
- **A PyTorch recipe**: copy a snapshot, then `torch.from_numpy`, with explicit dtype and
  device handling, for every kernel. It is documentation tested in its own CI workflow;
  Frames2Py has no PyTorch dependency, extra or code. Details:
  [Handing snapshots to PyTorch](https://siddiquifaras.github.io/frames2py/consumers/pytorch/).

### Changed

- **Custom kernels: `read()` may be given a later time.** `Kernel.read(state, out,
  watermark)` may now receive a time later than the accumulated watermark, and must evaluate
  the representation at it without changing the state. Only `replay.windows()` does this;
  `Accumulator` and `Engine` still pass the accumulated watermark, so a custom kernel used
  through them behaves as in 1.0. The seven built-in kernels follow the rule. Details:
  [Custom kernels](https://siddiquifaras.github.io/frames2py/core/kernels/#custom-kernels).

### Documentation

- The consumer pattern now waits with `wait_for_newer()`, with polling as the alternative
  for consumers on their own clock.
- New: the temporal kernel semantics table, the measured cost of waiting, the results of the
  v1 observation study (with its scope: one machine, one workload, threads in one process),
  and a [Known limitations](https://siddiquifaras.github.io/frames2py/reference/support/#known-limitations) section.
- New data files in `benchmarks/results/`: the temporal-kernel gate's two runs, the
  `wait_for_newer` measurement and the observation study's per-configuration table.

### Maintenance

- CI also runs once a month on `main`, including a job that installs the newest NumPy,
  extras and CPython builds instead of the lockfile's.

### Upgrading from 1.0

Nothing needs to change. The event contract, the five 1.0 kernels and their outputs,
`Accumulator`, `Engine` (`ingest()`, `snapshot()`, `stats`, the cadence and the
lifecycle), `Snapshot` and the publisher, the adapters, the recorder, `paced()` and the
viewer behave as in 1.0. The supported Python versions and platforms and the NumPy floor are
unchanged; [Supported Python and platforms](https://siddiquifaras.github.io/frames2py/reference/support/).

- A custom kernel you want to use with `replay.windows()` must handle a `read()` time later
  than its own watermark (above).
- Two internal changes speed up 1.0 paths without changing their results: the Accumulator
  reuses the timestamp maximum of the range check, and `TimestampDecay` computes its
  exponentials in place. The v1 performance figures were measured on 1.0's code and not
  re-measured.
- Cross-process snapshots are not part of 1.1.

## 1.0.0

The first stable release. The code is that of 1.0.0rc1, which was installed from PyPI and
checked before this release. What changed: the version, the `Development Status` classifier
(now `5 - Production/Stable`), and the documentation, which drops its release-candidate
install notes. From 1.0.0 on, the public API is stable: changing it incompatibly needs a 2.0.

```sh
pip install frames2py
```

What the release contains, from the core and its five kernels to the adapters, recorder,
replay and viewer, is listed under
[1.0.0rc1](https://siddiquifaras.github.io/frames2py/changelog/#100rc1).

## 1.0.0rc1

The first release of Frames2Py, published to PyPI as a release candidate for 1.0.0. Its API
is the one intended for 1.0.0; 1.0.0 follows once this candidate has been checked as
installed from PyPI.

**Installing the release candidate.** pip and uv skip pre-releases such as 1.0.0rc1 when a
stable release exists, and install one only when none does. So while 1.0.0rc1 is the only
release, `pip install frames2py` installs it; once 1.0.0 is published, it installs 1.0.0. To
ask for the release candidate explicitly:

```sh
pip install frames2py==1.0.0rc1
pip install --pre frames2py   # the newest release, pre-releases included
```

Everything below is new in this release.

### Core

- `Engine`: the live runtime. One producer thread calls `ingest()`; any number of consumers
  call `snapshot()` and read `stats` from other threads, and the producer never waits for
  them. Publication happens at most once per `snapshot_interval_ms` (16 ms by default; 0
  publishes on every call), only inside `ingest()` and `stop()`. `start()`, `stop()` and
  `reset()` manage the lifecycle.
- `Accumulator`: the same accumulation, synchronous, with no publication or threads.
- Snapshots (`frames2py.publish.Snapshot`): a frame and its metadata (`watermark`,
  `sequence`) from one publication. The frame is shared by every consumer and marked
  read-only; `copy()` returns an independent writable copy.
- The event contract: `EVENT_DTYPE` (`t` uint64 µs, `x` and `y` uint16, `p` uint8),
  structural validation (`TypeError`), whole-call rejection of any `t >= 2**63`
  (`ValueError`), and out-of-bounds events counted instead of accumulated.
- The `Kernel` protocol (`frames2py.kernels.Kernel`) and the `SnapshotPublisher` protocol
  (`frames2py.publish`) are public.

Details: [Engine](https://siddiquifaras.github.io/frames2py/core/engine/),
[Event contract](https://siddiquifaras.github.io/frames2py/core/event-contract/),
[Snapshots and consumers](https://siddiquifaras.github.io/frames2py/core/snapshots/).

### Kernels

| kernel | output | mode |
|---|---|---|
| `event_count` | (H, W) uint32 | windowed; counts wrap modulo 2^32 |
| `polarity` | (H, W, 2) uint32 | windowed; channel 0 OFF, channel 1 ON |
| `time_surface` | (H, W) uint64 | running; the latest timestamp per pixel |
| `ExpDecay(decay)` | (H, W) float32 | running; decays once per call, so it depends on batching |
| `TimestampDecay(tau_us)` | (H, W) float32 | running; decays with event time, independent of batching |

Details: [Kernels](https://siddiquifaras.github.io/frames2py/core/kernels/).

### Data and consumers

Each of these is an optional extra; `import frames2py` needs NumPy only.

- `frames2py.adapters.evt` (`frames2py[evt]`): EVT 2.0 and 3.0 (Prophesee RAW), decoded by
  Frames2Py's own NumPy decoder, with no further dependency.
- `frames2py.adapters.aedat4` (`frames2py[aedat4]`): AEDAT 4.0 through dv-processing.
- `frames2py.adapters.hdf5` (`frames2py[hdf5]`): HDF5 files with 1-D `t`, `x`, `y`, `p`
  datasets, through h5py and hdf5plugin.
- `frames2py.recorder` (`frames2py[recorder]`): writes events to HDF5, called next to
  `ingest()` by your own loop; the Engine never calls it.
- `frames2py.replay.paced()`: yields a recording's batches at their recorded pace.
- `frames2py.viewer` (`frames2py[viewer]`, pyglet): `render()` turns a snapshot into an RGB
  image; `run()` shows an Engine in a window.

There are no vendor SDK adapters: SDK output enters through `EVENT_DTYPE`. Details:
[Adapters](https://siddiquifaras.github.io/frames2py/data/adapters/).

### Python and platforms

CPython 3.11 to 3.14, and free-threaded CPython 3.14t with the GIL disabled, on Linux
x86_64, Linux ARM64 and macOS ARM64, with NumPy 2.4 or newer. Other free-threaded minor
versions with the GIL disabled are refused: `Engine(...)` raises `RuntimeError`. The wheel is
pure Python (`py3-none-any`). What CI tests on which platform:
[Supported Python and platforms](https://siddiquifaras.github.io/frames2py/reference/support/).

### Performance

On one Apple M4 (16 GB), the v1 performance gate measured all 150 of its cells above 20M
events/s, on CPython 3.11 and free-threaded 3.14t. No other hardware has been measured.
The cells, the method and the caveats:
[Performance](https://siddiquifaras.github.io/frames2py/reference/performance/).

### Documentation

<https://siddiquifaras.github.io/frames2py/>
