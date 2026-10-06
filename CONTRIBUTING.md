# Contributing to Frames2Py

Frames2Py is developed on GitHub at
[siddiquifaras/frames2py](https://github.com/siddiquifaras/frames2py). Issues and pull requests
are welcome there. Security problems are not reported in public issues: see
[SECURITY.md](https://github.com/siddiquifaras/frames2py/blob/main/SECURITY.md).

This page is the practical path from a fresh clone to a pull request. The documentation it
links to is at <https://siddiquifaras.github.io/frames2py/>.

## Before you start

- **Scope.** Frames2Py is live, decoupled observation of event-camera state: the core, seven
  kernels, file adapters at the `EVENT_DTYPE` boundary, a recorder, replay and a small viewer.
  It is not a camera SDK, a format-conversion toolkit, an ML framework, a general stream
  processor or a visualisation package. Features outside that scope are better proposed
  first as an issue.
- **The contract comes first.** The behaviour described in the documentation (the event
  contract, kernel semantics, snapshot and lifecycle rules) is what the tests check. A
  change that alters it is an API change: 1.x changes the public API only compatibly, so
  discuss it in an issue before writing code.
- **Performance changes need measurements,** taken as described under
  [Benchmarks](#benchmarks), with their conditions. Intuition about NumPy performance is not
  evidence.

## Setting up a checkout

Install [uv](https://docs.astral.sh/uv/). CI uses uv 0.12.20, and the build backend is
pinned to `uv_build` 0.12.20; using the same uv locally keeps `uv.lock` unchanged.

```sh
git clone https://github.com/siddiquifaras/frames2py.git
cd frames2py
uv sync                  # Python 3.11 (.python-version), the package from src/, NumPy, the dev tools
uv sync --all-extras     # also dv-processing, h5py, hdf5plugin and pyglet
```

`uv sync` installs the `dev` dependency group: pytest, Hypothesis, pytest-timeout, mypy and
pyflakes. The `docs` group is separate and only installed when asked for.
None of the groups is a dependency of the package: `import frames2py` needs NumPy only.

### Python versions

Frames2Py supports CPython 3.11 to 3.14 and free-threaded CPython 3.14t with the GIL
disabled ([Supported Python and platforms](https://siddiquifaras.github.io/frames2py/reference/support/)).
The checkout's environment is 3.11, the floor. To run the suite on another version without
touching it, ask uv for a temporary environment, naming the build explicitly:

```sh
uv run --isolated --python 3.14+gil --all-extras pytest   # standard CPython 3.14
uv run --isolated --python 3.14t --all-extras pytest      # free-threaded CPython 3.14
```

On a free-threaded build, the `Engine` refuses every minor version other than 3.14 with the
GIL disabled (`RuntimeError`), and 3.14.5 or later is recommended. Check that the GIL really
is disabled with all extras loaded, as CI does:
`python -c "import sys; print(sys._is_gil_enabled())"` after importing them.

## Running the tests

```sh
uv run pytest                                         # the suite; tests needing a missing extra skip
FRAMES2PY_REQUIRE_EXTRAS=1 uv run --all-extras pytest # as CI: a missing backend fails instead
FRAMES2PY_SLOW_TESTS=1 uv run pytest tests/contract/test_kernels.py tests/contract/test_temporal_kernels.py
uv run pytest --hypothesis-profile explore            # up to 3,000 fresh random examples per property
uv run pytest --display tests/viewer                  # tests that open a window; need a display
uv run python -m tests.recordings download            # about 400 MB of public recordings, then:
uv run pytest tests/adapters --recordings
```

- The slow tests accumulate 2^32 events or more, to check the documented count and numerator
  wrap-around. The voxel-grid one takes several minutes. CI runs them on Linux x86_64 with
  CPython 3.11.
- The default property run uses a fixed set of examples, so a failure reproduces; the
  `explore` profile searches randomly and prints the smallest failing example it finds.
- [Testing](https://siddiquifaras.github.io/frames2py/development/testing/) lists what each
  part of the suite covers.

## Static checks

The same commands as CI's `lint` job:

```sh
uv lock --check
uv run pyflakes src tests benchmarks examples docs .github/scripts
uv run mypy src/frames2py
uv run mypy benchmarks
uv run mypy --strict .github/scripts
```

mypy runs in strict mode and is clean. There is no formatter, and formatting is not
checked.

## Documentation

```sh
uv run --only-group docs zensical build -f docs/mkdocs.yml --strict
uv run --only-group docs python docs/check_docstrings.py
uv run --only-group docs python docs/check_site.py
uv run pytest tests/test_docs.py
```

The build fails on broken links, anchors and API references; `check_docstrings.py` fails on
any docstring the API reference can't parse; `check_site.py` serves the built site under
`/frames2py/`, as GitHub Pages does, and checks every page, link, anchor and asset, and the
absolute links in the README, the changelog and this file. `tests/test_docs.py` runs every
Python example in the README and the documentation and compares its output with the page.

- Every Python block is an included example (`docs/snippets/name.py`, with its output in
  `name.out`), an inline example followed by its `Output` block, or a sketch whose first line
  starts `# Sketch (not runnable)`. After changing an example, regenerate its output, check
  the diff and commit both.
- The API reference documents every name in every public module's `__all__`, one
  `:::` directive each, and nothing private; the docs tests fail otherwise.
- Diagrams are SVG files in `docs/content/assets/`, written by hand: no diagramming
  dependency. Each draws its own background for light and dark schemes, has a `<title>` and
  `<desc>`, and is placed with alt text that says what it shows. A diagram shows only what
  the text and the tests support.
- Write as one engineer to another: direct, precise, understated. A performance number
  always comes with its conditions (hardware, Python, NumPy, resolution, events per call,
  kernel, method), and one configuration never stands for all. Don't describe the publisher
  as free of synchronisation (CPython locks the list during the store) or NumPy's read-only
  flag as making a frame immutable, and say what happens for unusual input rather than
  calling it undefined.
- Update the [changelog](https://siddiquifaras.github.io/frames2py/changelog/) for any
  change a user would notice.

## Testing philosophy

The rules are short, and reviews hold to them:

- **A test establishes behaviour from the contract,** against an independent oracle where
  one fits. Exercising code is not enough. Kernels are checked against plain per-event Python
  loops (`tests/oracle.py`, `tests/temporal_oracle.py`), never against the kernel's own
  vectorised formula typed out again.
- **No tautological tests.** A test that recomputes the result with the implementation's
  calculation, or restates an implementation detail, can't fail when the implementation is
  wrong.
- **No change-detector tests.** A test that fails when the implementation changes but the
  contract doesn't protects code, not behaviour. Don't assert private fields or call order
  unless the contract says so. Such tests are deleted, not maintained.
- **Prefer properties** where they are stronger than examples: results independent of
  arrival order and of how events are split into calls, rejected calls changing nothing
  (`tests/contract/test_properties.py`).
- **Regression tests** only where a bug shows that some contract behaviour wasn't covered;
  the test then covers that behaviour, not the bug.
- **Concurrency tests** check the documented guarantees with deterministic synchronisation,
  not sleeps, and never assert more than the documentation promises. They run on standard
  builds and on the supported free-threaded version, and skip where the `Engine` refuses the
  runtime.

Before writing a test, ask: if this failed in six months, would it tell us something
important? And: does it test behaviour, or today's implementation?

### Showing that a test can fail

There is no mutation-testing tool in the repository or in CI. Sensitivity is shown with
deliberately broken implementations:

- committed negative controls, which must fail the checks they target: the broken Engines and
  publishers in `tests/contract/test_interleavings.py`, `tests/contract/test_wait_interleavings.py`,
  `tests/contract/test_wait_for_newer.py` and `tests/contract/test_concurrency.py`;
- for a change to a kernel or the core, throwaway mutants: break the implementation on purpose
  in a copy (a wrong bin edge, a dropped polarity case, saturation instead of wrap-around),
  run the relevant tests against it, and confirm a named test fails. Say in the pull request
  what you broke and which test caught it. Mutant copies and scripts stay out of the tree;
  `scratch/` is ignored by git.

## Adding or changing a kernel

The [Kernel protocol](https://siddiquifaras.github.io/frames2py/core/kernels/#custom-kernels)
(`frames2py.kernels.Kernel`) is public: a kernel of your own can live in your code and be
passed to `Accumulator`, `Engine` and `replay.windows()` without changing Frames2Py. The
built-in set is a decision of the project: propose a new built-in kernel in an issue first,
with the event-vision use it serves.

A kernel, built-in or not, keeps to the contract:

- **Output.** A fixed, documented shape and dtype: `(H, W)` or `(H, W, channels)` for 2-D
  kernels, time-first (`(bins, H, W)`, `(2, bins, H, W)`) for temporal ones. The output
  dtype is public API; the internal storage is the kernel's own.
- **Polarity** is `p != 0`, folded into the index, so no `p` value reaches another pixel.
- **Windowed or running.** A windowed kernel starts a new window at each publication; a
  running kernel keeps its state across publications. Neither depends on how Frames2Py splits
  a call internally.
- **`read()`** changes no state, and since 1.1 may be given a time later than the
  accumulated watermark (`replay.windows()` does this).
- **Integer counts wrap** and the wrap is documented (uint32 counts wrap modulo 2^32).
- **Implementation pitfalls.** Index arithmetic in `np.intp` (`uint16 * int` wraps silently
  under NumPy 2), timestamp differences in int64 before converting to float, and no work
  proportional to the frame (H x W) on the accumulation path without benchmark evidence.

A change to a built-in kernel comes with: tests against an independent per-event oracle,
property tests for order and partition invariance where the kernel promises them, a broken
control that fails them, the kernels page (and the temporal semantics table for a temporal
kernel), the API reference, the changelog, and benchmark results if the change touches
performance.

## Benchmarks

Benchmarks live in `benchmarks/`, apart from the tests; the test suite only checks the
harness (`tests/test_benchmarks.py` and its siblings), never performance. CI measures no
throughput.

```sh
uv run python -m benchmarks --help
uv run python -m benchmarks list --suite gate
```

[Benchmark methodology](https://siddiquifaras.github.io/frames2py/reference/methodology/#measuring-your-own-machine)
has the full procedure. What matters for a result anyone else can use:

- **A clean tree at a commit.** A result document records the commit and the tree's state,
  and the gate doesn't classify a document from a dirty tree. Write results outside the
  checkout.
- **A quiet machine.** On AC power, with no low-power mode, no sleep, and no competing work:
  close other applications, stop background services such as container runtimes, and leave
  only the terminal that launches the run. On macOS the runner holds an idle-sleep assertion
  and refuses to start outside full wake, and a run that slept is invalid.
- **Separate processes and repeated runs.** Each run is its own process, and the suite never
  compares single runs.
- **Conditions with every number:** hardware, Python, NumPy, resolution, events per call,
  kernel, method. A measurement on one machine is a measurement of that machine.
- **For an optimisation,** outputs bit-identical to the current implementation (or within the
  documented tolerance), the full suite passing, and before and after measured in the same
  session.

## Pull requests

1. Fork the repository (or, for maintainers, branch from `main`). `main` is protected:
   changes reach it through pull requests, and the CI checks must pass.
2. Keep each commit one coherent change, with a subject of the form `<type>: <description>`
   (`feat`, `bug`, `test`, `docs`, `perf`, `refactor`, `chore`, `updated`), for example
   `bug: reject out of range timestamps atomically`. A body, if useful, is two or three
   short sentences on what changed and why.
3. Update the documentation and the changelog with the behaviour.
4. In the pull request, say what changed and why, which commands you ran (tests, static
   checks, docs build), and, for a performance change, the measurements with their
   conditions.
5. Leave out generated files, build output, scratch material, credentials and `.env` files.

Unless you state otherwise, a contribution you submit for inclusion is licensed under the
[Apache License, Version 2.0](https://github.com/siddiquifaras/frames2py/blob/main/LICENSE),
as its section 5 says.

## What CI runs

On every push and pull request (`.github/workflows/ci.yml`):

- **lint**: pyflakes, mypy (strict) on the package, the benchmarks and the CI scripts,
  `uv lock --check`;
- **package**: the wheel and sdist built twice and compared byte for byte, the wheel rebuilt
  from the sdist, the contents and metadata audited (`.github/scripts/check_dist.py`), the
  sdist installed with uv and with pip;
- **test**: the full suite against the installed wheel, with every extra, on Linux x86_64,
  Linux ARM64 and macOS ARM64 for the supported Python versions, including 3.14t with the
  GIL disabled;
- **core-only**, **floor** (NumPy 2.4.1 and the backends' minimum versions), **display**
  (window tests under Xvfb) and **constrained** (1 CPU, 2 GiB) jobs;
- **docs**: the documentation built strictly, docstrings checked, the built site served
  under `/frames2py/` and every link checked;
- **py315**: an informational run on the CPython 3.15 pre-release, not required.

The same workflow also runs once a month on `main`. That run adds a **latest** job, which
installs the wheel with its dependencies resolved fresh from PyPI and the newest CPython
builds, instead of the lockfile's, so new NumPy and Python releases that break Frames2Py
show up as a failed run.

Separate workflows, none of which the jobs above depend on: `recordings.yml` runs the suite
against the real recordings every week; `torch.yml` runs the
[PyTorch recipe](https://siddiquifaras.github.io/frames2py/consumers/pytorch/)'s tests
against one pinned CPU build of PyTorch on every push. No job in `ci.yml` installs PyTorch.
The documentation is
deployed to GitHub Pages from `main` by `pages.yml`.

## Releases (maintainers only)

Merging into `main`, pushing a version tag, approving the `pypi` deployment and anything
done to a GitHub Release by hand are the maintainer's alone.

A release is published by `release.yml`, and only when a maintainer pushes a version tag
(`v1.1.0`) on a commit of `main`. Before anything is built, it checks that the tag names
exactly the version in `pyproject.toml`, that the tagged commit is on `main`, and that the
[changelog](https://siddiquifaras.github.io/frames2py/changelog/) has a section for that
version. It then builds the wheel and sdist once, audits them, runs the test suite against
them as installed, uploads those same files to PyPI through trusted publishing once a
maintainer approves the deployment, and only after that creates the GitHub Release, with the
changelog section as its notes. There is no other way to publish.

A version bump therefore changes `pyproject.toml`, `__version__` and `uv.lock`, and adds the
version's changelog section; the test suite fails if the section is missing. A published
version number is never reused: a problem found after a release is fixed in the next patch
release.
