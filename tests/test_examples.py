"""The v1 examples run to completion. The windowed ones are opt-in (``--display``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.adapters.backends import require_backend

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(*args: str) -> str:
    done = subprocess.run([sys.executable, str(EXAMPLES / args[0]), *args[1:]], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.parametrize("compression", ["blosc", "gzip", "none"])
def test_record_and_read_back(compression: str) -> None:
    require_backend("h5py", "hdf5plugin")
    assert "identical to the source: True" in run_example("record_and_read_back.py", "--compression", compression)


@pytest.mark.display
@pytest.mark.parametrize("args", [("view_synthetic.py", "--seconds", "1"),
                                  ("view_synthetic.py", "--kernel", "timestamp_decay", "--seconds", "1"),
                                  ("replay_recording.py", "--seconds", "2")])
def test_viewer_examples(args: tuple[str, ...]) -> None:
    require_backend("pyglet")
    assert "EngineStats(" in run_example(*args)
