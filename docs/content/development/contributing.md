# Contributing

Frames2Py is developed on GitHub at
[siddiquifaras/frames2py](https://github.com/siddiquifaras/frames2py). Issues and pull requests
are welcome there.

## Before you start

- **Scope.** Frames2Py is live, decoupled observation of event-camera state: the core, five
  kernels, file adapters at the `EVENT_DTYPE` boundary, a recorder, replay and a small viewer.
  It is not a camera SDK, a format-conversion toolkit, an ML framework or a general stream
  processor. Features outside that scope are better proposed first as an issue.
- **The contract comes first.** The behaviour described in these docs (the event contract,
  kernel semantics, snapshot and lifecycle rules) is what the tests check. A change that
  alters it is an API change and needs discussing before code.
- **Performance changes need measurements.** Use the benchmark suite
  ([methodology](../reference/methodology.md)) and report the conditions; intuition about
  NumPy performance doesn't count.

## Workflow

1. Branch from `main`. `main` is protected: changes reach it through pull requests, and the
   CI checks must pass.
2. Set up with `uv sync --all-extras` and run the tests ([Testing](testing.md)).
3. Keep each commit one coherent change, with a subject of the form `<type>: <description>`
   (`feat`, `bug`, `test`, `docs`, `perf`, `refactor`, `chore`), for example
   `bug: reject out of range timestamps atomically`.
4. Update the documentation with the behaviour. A runnable example goes in `docs/snippets/`
   with its expected output, so the test suite runs it.

## What CI runs

On every push and pull request (`.github/workflows/ci.yml`):

- **lint**: pyflakes, mypy (strict) on the package and the benchmarks, `uv lock --check`;
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

A weekly workflow (`recordings.yml`) runs the suite against the real recordings. The
documentation is deployed to GitHub Pages from `main` by `pages.yml`.

## Releases

A release is published by `release.yml`, and only when a maintainer pushes a version tag
(`v1.0.0`, `v1.0.0rc1`) on a commit of `main`. Before anything is built, it checks that the
tag names exactly the version in `pyproject.toml`, that the tagged commit is on `main`, and
that the [changelog](../changelog.md) has a section for that version. It then builds the
wheel and sdist once, audits them, runs the test suite against them as installed, uploads
those same files to PyPI through trusted publishing once a maintainer approves the
deployment, and only after that creates the GitHub Release, with the changelog section as its
notes. A version bump therefore changes `pyproject.toml`, `__version__` and `uv.lock`, and
adds the version's changelog section; the test suite fails if the section is missing.
