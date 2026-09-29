# Supported Python versions and platforms

What Frames2Py supports, and what each claim rests on. "Tested" means the full test suite
passed in this repository's CI (`.github/workflows/ci.yml`) against the built wheel,
installed into a fresh environment. A job fails if the wheel wasn't the code under test.
Supported and fast are separate claims: nothing here says anything about throughput.

## Core package

`pip install frames2py` needs NumPy and nothing else.

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
  3.13 run in CI on Linux x86_64 only. The package itself is pure Python
  (`py3-none-any`).
- **Other platforms:** Windows and Intel macOS are not supported and not tested. The wheel
  installs anywhere pip accepts it, but that is not a support claim.
- **Other interpreters:** PyPy and other Python implementations are not supported.

### Free-threaded CPython

- **Supported:** CPython 3.14t with the GIL disabled. CI checks that the GIL stays off once
  NumPy and every optional backend are loaded. It also checks that the concurrency test
  which only runs without the GIL (parallel `stats` reads during `ingest()`) ran and
  passed.
- **Refused:** every other free-threaded minor version with the GIL disabled. `Engine(...)`
  raises `RuntimeError` there. That includes 3.13t, and it includes 3.15t and later until each
  has been checked in its own right. A free-threaded build running with the GIL enabled
  behaves as a standard build.
- **No classifier:** the package metadata carries no free-threading classifier, because the
  classifiers can't say "3.14t only".

### CPython 3.15

- **Not supported:** there is no 3.15 classifier and no support claim.
- **Informational CI job:** runs the suite without extras on the current 3.15 pre-release
  (3.15.0rc2 on 2026-09-29, before the 3.15.0 release). Its result is not a support claim.
- **Extras on 3.15:** on 2026-09-29, h5py 3.16.0 and dv-processing 2.0.4 published no CPython
  3.15 wheels.

### NumPy

- **Floor:** the package requires `numpy>=2.4`. NumPy 2.4.0 is yanked, so the lowest release
  a resolver installs is 2.4.1.
- **Floor tested:** CI runs the full suite with NumPy 2.4.1 and every optional backend at its
  declared minimum, on CPython 3.11, on Linux x86_64 and macOS ARM64. The NumPy version is
  checked inside the test process.
- **Current releases:** the other jobs use the locked releases: NumPy 2.4.6 on CPython 3.11
  (NumPy 2.5 needs 3.12 or newer), and 2.5.3 on 3.12 and later.

### Constrained resources

- **Tested:** the suite without extras, contract tests included, passes from the wheel in a
  Linux ARM64 container limited to 1 CPU and 2 GiB of memory.
- **Scope:** this is a compatibility result, not a memory requirement and not a throughput
  measurement.

## Optional extras

Each extra was tested in every "tested" cell of the core table. These jobs have every extra
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
  later only (`macosx_15_0_arm64`). On older macOS, `pip install "frames2py[aedat4]"` has no
  wheel to install. CI's macOS runner is macOS 15.
- **AEDAT4 on Linux:** dv-processing's Linux wheels need the system `libatomic1` library.
- **Viewer windows:** real window tests run in CI on Linux x86_64 under Xvfb, with CPython
  3.11 and 3.14t. On macOS, CI tests the renderer, which needs no window, but opens no
  window. `viewer.run()` must be called on the main thread on every platform; it raises
  `RuntimeError` otherwise.
