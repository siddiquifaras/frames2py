"""The end-to-end notebook runs against the installed package, and what is committed is one clean run of it.

The first test needs nothing beyond the standard library and runs everywhere. The second runs the notebook; it
needs the ``notebook`` dependency group and skips without it (``tests/notebook.py``). The notebook's own assertions
check the Frames2Py behaviour its text describes.
"""

from __future__ import annotations

from pathlib import Path

from tests.notebook import execute, load, problems


def test_the_committed_notebook_is_one_clean_top_to_bottom_run() -> None:
    assert problems(load(), committed=True) == []


def test_the_notebook_runs_against_the_installed_package(tmp_path: Path) -> None:
    executed = execute(load(), tmp_path)
    assert problems(executed, committed=False) == []
