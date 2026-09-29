# Supported Python and platforms

What Frames2Py supports, and what each claim rests on. **Tested** means the full test suite
passed in the repository's CI (`.github/workflows/ci.yml`) against the built wheel,
installed into a fresh environment; a job fails if the wheel wasn't the code under test.
**Supported and fast are separate claims**: nothing on this page says anything about
throughput. Throughput has been measured on one machine only ([Performance](performance.md)).

## Core package

The core needs NumPy and nothing else. It is pure Python (a `py3-none-any` wheel).

| | Linux x86_64 | Linux ARM64 | macOS ARM64 |
|---|---|---|---|
| CPython 3.11 | tested | tested | tested |
| CPython 3.12 | tested | not in CI | not in CI |
| CPython 3.13 | tested | not in CI | not in CI |
| CPython 3.14 | tested | tested | tested |
| CPython 3.14t, GIL disabled | tested | tested | tested |

- **CI runners:** GitHub-hosted `ubuntu-24.04`, `ubuntu-24.04-arm` (native ARM64, not
  emulated) and `macos-15` (Apple silicon), with uv-managed CPython builds.
- **Python:** 3.11 is the lowest supported version; it follows from the NumPy floor. 3.12 and
  3.13 run in CI on Linux x86_64 only.
- **Other platforms:** Windows and Intel macOS are not supported and not tested. The wheel
  installs anywhere pip accepts it, but that is not a support claim.
- **Other interpreters:** PyPy and other Python implementations are not supported. Neither is
  WebAssembly (Pyodide).
- **Package index:** releases are published on PyPI as `frames2py`; see
  [Installation](../getting-started/installation.md).

## Free-threaded CPython

| build | status |
|---|---|
| CPython 3.14t, GIL disabled | **supported**: tested on all three platforms above |
| CPython 3.13t, GIL disabled | **refused**: `Engine(...)` raises `RuntimeError` |
| CPython 3.15t and later, GIL disabled | **refused** until each minor version is verified |
| any free-threaded build with the GIL enabled (`PYTHON_GIL=1`) | behaves as a standard build |

- **What "supported" means on 3.14t:** the full test suite passes, including the
  concurrency tests of the documented model (one producer thread, any number of consumer
  threads, lifecycle calls from any thread; see
  [Lifecycle and threads](../core/lifecycle.md)). CI checks that the GIL stays disabled
  once NumPy and every optional backend are imported, and that the concurrency test which
  only runs without the GIL (parallel `stats` reads during `ingest()`) ran and passed.
- **Why other minors are refused:** the snapshot hand-off relies on CPython source-level
  behaviour that is checked for each minor version before it is enabled
  ([Architecture](../core/architecture.md#how-the-hand-off-works)).
- **No classifier:** the package metadata carries no free-threading Trove classifier,
  because the classifiers can't say "3.14t only".
- **Throughput** on 3.14t has been measured on one Apple M4; see
  [Performance](performance.md).

## CPython 3.15

- **Not supported:** there is no 3.15 classifier and no support claim.
- **Informational CI job:** runs the suite without extras on the current 3.15 pre-release
  (3.15.0rc2 on 2026-09-29). Its result is not a support claim.
- **Extras on 3.15:** on 2026-09-29, h5py 3.16.0 and dv-processing 2.0.4 published no CPython
  3.15 wheels.

## NumPy

- **Floor:** `numpy>=2.4`. NumPy 2.4.0 is yanked, so the lowest release a resolver installs
  is 2.4.1.
- **Floor tested:** CI runs the full suite with NumPy 2.4.1 and every optional backend at its
  declared minimum, on CPython 3.11, on Linux x86_64 and macOS ARM64. The NumPy version is
  checked inside the test process.
- **Current releases:** the other jobs use the locked releases: NumPy 2.4.6 on CPython 3.11
  (NumPy 2.5 needs 3.12 or newer), and 2.5.3 on 3.12 and later.

## Constrained resources

- **Tested:** the suite without extras, contract tests included, passes from the wheel in a
  Linux ARM64 container limited to 1 CPU and 2 GiB of memory.
- **Scope:** a compatibility result under CPU and memory limits on a hosted runner. It is not
  a memory requirement, not a test on a small device and not a throughput measurement; no
  throughput has been measured on any edge device.

## Optional extras

Each extra was tested in every "tested" cell of the core table: those jobs have every extra
installed, and a missing backend fails the run instead of skipping its tests.

| extra | installs | notes |
|---|---|---|
| `evt` | nothing beyond NumPy | the EVT 2.0 / 3.0 decoder is part of Frames2Py |
| `aedat4` | dv-processing >= 2.0.4 | see the platform limits below |
| `hdf5` | h5py >= 3.16, hdf5plugin >= 7.1 | |
| `recorder` | h5py >= 3.16, hdf5plugin >= 7.1 | |
| `viewer` | pyglet >= 2.1.16 | window tests on Linux only, see below |

The minimum versions above were tested too, on CPython 3.11, in the NumPy floor jobs.

### Platform limits

- **AEDAT4 on macOS:** dv-processing 2.0.4 publishes macOS ARM64 wheels for macOS 15 and
  later only (`macosx_15_0_arm64`). On older macOS, `frames2py[aedat4]` has no wheel to
  install. CI's macOS runner is macOS 15.
- **AEDAT4 on Linux:** dv-processing's Linux wheels need the system `libatomic1` library.
- **Viewer windows:** real window tests run in CI on Linux x86_64 under Xvfb, with CPython
  3.11 and 3.14t. On macOS, CI tests the renderer, which needs no window, but opens no
  window. `viewer.run()` must be called on the main thread on every platform; it raises
  `RuntimeError` otherwise.
