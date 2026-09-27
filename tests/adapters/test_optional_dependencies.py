"""Optional backends: ``import frames2py`` never needs one, and a missing one fails ``open()`` with its extra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BACKENDS = {"aedat4": ("dv_processing",), "hdf5": ("h5py", "hdf5plugin")}


def test_importing_frames2py_and_every_adapter_imports_no_backend() -> None:
    code = (
        "import sys; import frames2py, frames2py.adapters.evt, frames2py.adapters.aedat4, frames2py.adapters.hdf5; "
        "loaded = sorted(m for m in ('dv_processing', 'h5py', 'hdf5plugin') if m in sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(("adapter", "blocked"), [(a, (m,)) for a, mods in sorted(BACKENDS.items()) for m in mods])
def test_missing_backend_raises_import_error_naming_the_extra(
    adapter: str, blocked: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib

    module = importlib.import_module(f"frames2py.adapters.{adapter}")
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)  # makes ``import name`` raise ImportError
    path = tmp_path / "file"
    path.write_bytes(b"")
    kwargs = {"group": "events"} if adapter == "hdf5" else {}
    with pytest.raises(ImportError, match=rf"frames2py\[{adapter}\]") as info:
        module.open(path, **kwargs)
    assert isinstance(info.value.__cause__, ImportError)


def test_evt_needs_nothing_beyond_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    from frames2py.adapters import evt

    for name in ("dv_processing", "h5py", "hdf5plugin"):
        monkeypatch.setitem(sys.modules, name, None)
    data = Path(__file__).resolve().parent.parent / "data" / "active_marker_head.evt3.raw"
    with evt.open(data) as reader:
        assert sum(len(b) for b in reader) == 46_893
