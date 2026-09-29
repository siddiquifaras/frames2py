# Testing

## Setting up

```sh
git clone https://github.com/siddiquifaras/frames2py.git
cd frames2py
uv sync                  # the package, NumPy and the dev tools (pytest, Hypothesis, mypy, pyflakes)
uv sync --all-extras     # also dv-processing, h5py, hdf5plugin and pyglet
uv run pytest
```

Without the extras, the adapter, recorder and viewer tests that need a backend skip. Set
`FRAMES2PY_REQUIRE_EXTRAS=1` to turn those skips into failures, as CI does.

## What the suite covers

| path | what it checks |
|---|---|
| `tests/contract/` | the behavioural contract of the core: event validation, bounds and watermark, every kernel against an independent reference, publication cadence, snapshots, lifecycle, stats accounting, threads and bounded interleavings |
| `tests/oracle.py` | the independent reference model of the five kernels, a plain per-event Python loop, and its own tests (`tests/test_oracle.py`) |
| `tests/adapters/` | the EVT, AEDAT 4.0 and HDF5 adapters, against committed fixtures, crafted byte streams and OpenEB 5.2.0's recorded output (`tests/data/evt_golden/`) |
| `tests/recorder/`, `tests/viewer/`, `tests/test_replay.py` | the recorder, the renderer and viewer loop, and paced replay |
| `tests/test_docs.py` | the documentation: runnable examples and their output, the README quickstart, the API reference against the public API |
| `tests/test_examples.py` | the programs in `examples/` |
| `tests/test_benchmarks.py`, `tests/test_consumer_benchmarks.py` | the benchmark harness itself, not performance |

The tests check observable behaviour against references that don't reuse the
implementation's own calculation. Some are property tests (Hypothesis) over generated inputs:
kernel results against the reference for any sequence of calls, independence from arrival
order and call partitioning where a kernel promises it, rejected calls changing nothing, and
adapter output independent of how the input is cut (`tests/contract/test_properties.py`,
`tests/adapters/test_chunking_properties.py`). Benchmarks are kept separate from correctness tests and
never run in the test suite.

## Optional test sets

- **Real recordings.** `--recordings` runs the tests against full-size public recordings,
  downloaded on request and checked against a SHA-256:

    ```sh
    uv run python -m tests.recordings list
    uv run python -m tests.recordings download      # about 400 MB into ~/.cache/frames2py/recordings
    uv run pytest tests/adapters --recordings
    ```

    `FRAMES2PY_RECORDINGS_DIR` moves the cache.

- **Windows.** `--display` runs the tests that open a real window. They need a display; on
  Linux CI they run under Xvfb.
- **Slow tests.** `FRAMES2PY_SLOW_TESTS=1` runs the modulo-2^32 wrap tests, which accumulate
  2^32 events.
- **A wider property search.** The property tests run the same examples on every run, so a
  failure reproduces. `uv run pytest --hypothesis-profile explore` draws up to 3,000 fresh random
  examples per property instead; a failure prints the smallest example it found.

## Documentation tests

Every Python block in the README and in `docs/content/` is one of three kinds, and
`tests/test_docs.py` checks each:

1. **Included examples**: a block whose only line is `--8<-- "name.py"` pulls in
   `docs/snippets/name.py`. The test runs every snippet as its own process in a temporary
   directory holding the committed fixtures, and compares its output with
   `docs/snippets/name.out`, which the page includes as its output block.
2. **Inline examples**: every other Python block. The test runs a page's inline blocks in
   order in one namespace, and where a block is followed by a `text` block titled `Output`,
   compares what it printed.
3. **Sketches**: blocks whose first line is `# Sketch (not runnable)...`. They are shown, not
   run, and the label says so.

It also checks that the API reference documents every name in every public module's
`__all__`, and nothing that isn't public. After changing an example, regenerate its output
with the snippet run from a directory holding the fixtures, check the diff, and commit both.

## Building the documentation

```sh
uv run --only-group docs zensical build -f docs/mkdocs.yml --strict
uv run --only-group docs python docs/check_docstrings.py
uv run --only-group docs python docs/check_site.py
```

The site is written to `docs/site/` (ignored by git). `--strict` fails on broken links,
missing anchors and unresolved API references; `check_docstrings.py` fails on any docstring
the API reference can't parse; `check_site.py` serves the built site under `/frames2py/`, as
GitHub Pages will, and checks every page, link, anchor and asset, and the README's links.
`uv run --only-group docs zensical serve -f docs/mkdocs.yml` serves it locally while you
edit.

## Static checks

```sh
uv run pyflakes src tests benchmarks examples docs .github/scripts
uv run mypy src/frames2py
uv run mypy benchmarks
```

There is no formatter; formatting is not checked.
