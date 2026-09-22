#!/usr/bin/env python3
"""Execute the production E2E notebook and verify all outputs.

This script extracts every code cell from ``production_e2e.ipynb``,
executes them sequentially in a single shared namespace (exactly as
Jupyter does), and reports pass/fail per cell.

It requires **no Jupyter installation**  -  only the stdlib ``json``
module and whatever packages the notebook itself imports.

Usage::

    python run_and_verify.py            # execute and verify
    python run_and_verify.py --verbose  # show each cell's output
    python run_and_verify.py --dry-run  # parse only, don't execute

Exit codes:
    0  -  all cells passed
    1  -  one or more cells failed
    2  -  notebook file not found (run build_notebook.py first)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

NOTEBOOK_NAME = "production_e2e.ipynb"


def load_notebook(path: Path) -> list[tuple[int, str]]:
    """Return (cell_index, source) for every code cell."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells: list[tuple[int, str]] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).strip()
        if source:
            cells.append((i, source))
    return cells


def execute_cells(
    cells: list[tuple[int, str]],
    *,
    verbose: bool = False,
) -> tuple[int, int, list[tuple[int, str, str]]]:
    """Execute cells sequentially. Returns (passed, failed, errors)."""
    ns: dict = {}
    passed = 0
    failed = 0
    errors: list[tuple[int, str, str]] = []

    for cell_idx, source in cells:
        cell_label = f"Cell {cell_idx}"
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        t0 = time.perf_counter()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(source, f"<cell_{cell_idx}>", "exec"), ns)
            elapsed = time.perf_counter() - t0
            passed += 1
            status = "PASS"
        except Exception:
            elapsed = time.perf_counter() - t0
            failed += 1
            tb = traceback.format_exc()
            errors.append((cell_idx, tb, source))
            status = "FAIL"

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        if verbose or status == "FAIL":
            print(f"\n{'='*60}")
            print(f"  {cell_label}  [{status}]  ({elapsed:.2f}s)")
            print(f"{'='*60}")
            if status == "FAIL":
                print(f"\n--- Source (first 10 lines) ---")
                for line in source.splitlines()[:10]:
                    print(f"  {line}")
                if len(source.splitlines()) > 10:
                    print(f"  ... ({len(source.splitlines())} lines total)")
                print(f"\n--- Traceback ---")
                print(errors[-1][1])
            if stdout_text and verbose:
                print(f"--- stdout ---")
                print(stdout_text.rstrip())
            if stderr_text:
                print(f"--- stderr ---")
                print(stderr_text.rstrip())
        else:
            print(f"  {cell_label:>10s}  {status}  ({elapsed:.2f}s)")

    return passed, failed, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show stdout from every cell",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse notebook but don't execute",
    )
    args = parser.parse_args()

    nb_path = Path(__file__).resolve().parent / NOTEBOOK_NAME
    if not nb_path.exists():
        print(f"ERROR: {nb_path} not found.")
        print("Run `python build_notebook.py` first.")
        return 2

    cells = load_notebook(nb_path)
    print(f"Notebook: {nb_path}")
    print(f"Code cells: {len(cells)}")

    if args.dry_run:
        print("Dry run  -  not executing.")
        for idx, src in cells:
            first_line = src.splitlines()[0][:70]
            print(f"  Cell {idx}: {first_line}")
        return 0

    os.chdir(nb_path.parent)
    print(f"Working directory: {os.getcwd()}")
    print()

    t_total = time.perf_counter()
    passed, failed, errors = execute_cells(cells, verbose=args.verbose)
    t_total = time.perf_counter() - t_total

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed  ({t_total:.1f}s)")
    print(f"{'='*60}")

    if errors:
        print(f"\nFailed cells:")
        for cell_idx, tb, _ in errors:
            first_tb_line = tb.strip().splitlines()[-1]
            print(f"  Cell {cell_idx}: {first_tb_line}")
        return 1

    print("\nALL CELLS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
