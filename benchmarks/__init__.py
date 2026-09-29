"""Frames2Py benchmark suite.

Development tooling, not part of the installed package. Run from the repo root:

    uv run python -m benchmarks list --suite gate
    uv run python -m benchmarks run --suite gate --target v1-engine --out results.json
    uv run python -m benchmarks report results.json
    uv run python -m benchmarks gate --kernel-level k.json --engine-level e.json

The pieces are kept apart:

- definition: ``matrix`` (which cells exist) and ``workloads`` (what events they get)
- execution: ``runner`` and ``measure`` against a target from ``targets``
- results: ``results`` (JSON records with the environment they came from)
- interpretation: ``report`` (tables and the 20M events/s threshold comparison)

A harness is not a measurement. Numbers exist only once a suite has been run, and
they hold only for the environment recorded with them.
"""
