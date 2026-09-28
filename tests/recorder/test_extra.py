"""The recorder's extra: ``import frames2py.recorder`` needs no backend; ``open()`` names the extra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_importing_the_recorder_imports_no_backend() -> None:
    code = (
        "import sys; import frames2py.recorder; "
        "loaded = sorted(m for m in ('h5py', 'hdf5plugin') if m in sys.modules); assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("blocked", ["h5py", "hdf5plugin"])
def test_a_missing_backend_raises_import_error_naming_the_extra(
    blocked: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from frames2py import recorder

    monkeypatch.setitem(sys.modules, blocked, None)
    with pytest.raises(ImportError, match=r"frames2py\[recorder\]") as info:
        recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720))
    assert isinstance(info.value.__cause__, ImportError)
    assert list(tmp_path.iterdir()) == []
