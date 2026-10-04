"""The PyTorch recipe: what a consumer gets when it hands snapshots to PyTorch.

Frames2Py has no PyTorch code. These tests check, against the installed PyTorch, the behaviour the
recipe page states: a tensor from ``snapshot.copy()`` is the consumer's and leaves the published frame
alone; shapes and layouts arrive unchanged; the dtype conversions it recommends are exact where it says
so; and ``torch.from_numpy`` or ``torch.from_dlpack`` on ``snapshot.frame`` itself writes through to
every consumer. They skip when PyTorch isn't installed; the torch CI job runs them.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import frames2py
from tests.torch_support import require_torch

torch = require_torch()

SENSOR = (6, 4)  # width, height

# Each kernel with the output the kernels page gives it, for SENSOR.
KERNELS: list[tuple[Callable[[], Any], tuple[int, ...], Any]] = [
    (frames2py.EventCount, (4, 6), torch.uint32),
    (frames2py.Polarity, (4, 6, 2), torch.uint32),
    (frames2py.TimeSurface, (4, 6), torch.uint64),
    (lambda: frames2py.ExpDecay(0.5), (4, 6), torch.float32),
    (lambda: frames2py.TimestampDecay(100.0), (4, 6), torch.float32),
    (lambda: frames2py.StackedHistogram(bins=3, bin_us=10), (2, 3, 4, 6), torch.uint32),
    (lambda: frames2py.VoxelGrid(bins=3, bin_us=10), (3, 4, 6), torch.float32),
]
KERNEL_IDS = ["event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay", "stacked_histogram",
              "voxel_grid"]


def events() -> np.ndarray:
    ev = np.zeros(8, dtype=frames2py.EVENT_DTYPE)
    ev["t"] = [3, 8, 14, 21, 27, 33, 38, 45]
    ev["x"] = [0, 1, 2, 3, 4, 5, 1, 2]
    ev["y"] = [0, 1, 2, 3, 0, 1, 1, 2]
    ev["p"] = [1, 0, 1, 1, 0, 1, 0, 1]
    return ev


def published(kernel: Any) -> frames2py.Engine:
    engine = frames2py.Engine(SENSOR, kernel, snapshot_interval_ms=0)
    engine.ingest(events())
    return engine


def shares_memory(tensor: Any, array: np.ndarray) -> bool:
    return bool(np.shares_memory(tensor.numpy(), array))


# ---------------------------------------------------------------- the recipe


@pytest.mark.parametrize("make", [make for make, _, _ in KERNELS], ids=KERNEL_IDS)
def test_a_tensor_from_a_copy_is_the_consumers_and_leaves_the_published_frame_alone(make: Callable[[], Any]) -> None:
    engine = published(make())
    snapshot = engine.snapshot()
    assert snapshot is not None
    original = np.array(snapshot.frame)  # an independent record of the published values

    owned = snapshot.copy()
    tensor = torch.from_numpy(owned)
    assert shares_memory(tensor, owned) and not shares_memory(tensor, snapshot.frame)

    tensor.fill_(7)
    assert (owned == 7).all()  # the tensor is the consumer's copy
    assert np.array_equal(snapshot.frame, original)
    assert engine.snapshot() is snapshot and np.array_equal(engine.snapshot().frame, original)


@pytest.mark.parametrize(("make", "shape", "dtype"), KERNELS, ids=KERNEL_IDS)
def test_the_tensor_has_the_kernels_output_shape_and_dtype_with_nothing_transposed(
    make: Callable[[], Any], shape: tuple[int, ...], dtype: Any
) -> None:
    snapshot = published(make()).snapshot()
    assert snapshot is not None
    tensor = torch.from_numpy(snapshot.copy())
    assert tuple(tensor.shape) == shape and tensor.dtype == dtype and tensor.is_contiguous()
    assert np.array_equal(tensor.numpy(), snapshot.frame)  # element [i, j, ...] is frame[i, j, ...]


def test_the_recipes_channel_layouts_index_polarity_and_time_as_stated() -> None:
    polarity = published(frames2py.Polarity()).snapshot()
    histogram = published(frames2py.StackedHistogram(bins=3, bin_us=10)).snapshot()
    voxels = published(frames2py.VoxelGrid(bins=3, bin_us=10)).snapshot()
    assert polarity is not None and histogram is not None and voxels is not None

    nchw = torch.from_numpy(polarity.copy()).permute(2, 0, 1)
    for channel in (0, 1):  # 0 OFF, 1 ON
        assert np.array_equal(nchw[channel].numpy(), polarity.frame[..., channel])

    merged = torch.from_numpy(histogram.copy()).reshape(1, 2 * 3, 4, 6)
    for p in range(2):
        for b in range(3):
            assert np.array_equal(merged[0, p * 3 + b].numpy(), histogram.frame[p, b])

    batch = torch.from_numpy(voxels.copy()).unsqueeze(0)
    assert tuple(batch.shape) == (1, 3, 4, 6)
    for b in range(3):
        assert np.array_equal(batch[0, b].numpy(), voxels.frame[b])
    with torch.no_grad():
        assert tuple(torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)(batch).shape) == (1, 8, 4, 6)


def test_a_tensor_on_a_reused_buffer_shows_each_refill() -> None:
    engine = frames2py.Engine(SENSOR, "event_count", snapshot_interval_ms=0)
    buffer = np.zeros((4, 6), dtype=np.uint32)
    tensor = torch.from_numpy(buffer)
    ev = events()
    engine.ingest(ev[:3])
    engine.snapshot().copy(out=buffer)
    kept = tensor.clone()
    engine.ingest(ev[3:])
    engine.snapshot().copy(out=buffer)
    assert int(kept.sum()) == 3 and int(tensor.sum()) == 5


# ---------------------------------------------------------------- dtypes


def test_uint32_converts_exactly_to_int64_and_float64_but_not_to_float32_above_2_24() -> None:
    values = [0, 2**24, 2**24 + 1, 2**32 - 1]
    tensor = torch.from_numpy(np.array(values, dtype=np.uint32))  # EventCount, Polarity, StackedHistogram
    assert tensor.to(torch.int64).tolist() == values
    assert [int(v) for v in tensor.to(torch.float64).tolist()] == values
    assert [int(v) for v in tensor.to(torch.float32).tolist()] == [0, 2**24, 2**24, 2**32]
    with pytest.raises(NotImplementedError):  # why the recipe converts before arithmetic
        tensor + 1


def test_time_surface_values_convert_exactly_to_int64() -> None:
    engine = frames2py.Engine((2, 1), "time_surface", snapshot_interval_ms=0)
    ev = np.zeros(2, dtype=frames2py.EVENT_DTYPE)
    ev["t"] = [2**53 + 1, 2**63 - 1]  # the largest timestamp the event contract accepts
    ev["x"] = [0, 1]
    engine.ingest(ev)
    snapshot = engine.snapshot()
    assert snapshot is not None
    assert torch.from_numpy(snapshot.copy()).to(torch.int64).tolist() == [[2**53 + 1, 2**63 - 1]]


# ---------------------------------------------------------------- devices


def test_to_returns_the_same_tensor_unless_the_dtype_or_device_changes() -> None:
    snapshot = published(frames2py.VoxelGrid(bins=3, bin_us=10)).snapshot()
    assert snapshot is not None
    tensor = torch.from_numpy(snapshot.copy())
    assert tensor.to("cpu") is tensor
    assert tensor.to(torch.float64).data_ptr() != tensor.data_ptr()


def other_devices() -> list[str]:
    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    return found


@pytest.mark.parametrize("device", other_devices() or [pytest.param("none", marks=pytest.mark.skip(
    reason="no CUDA or MPS device in this environment"))])
def test_moving_to_another_device_copies(device: str) -> None:
    snapshot = published(frames2py.VoxelGrid(bins=3, bin_us=10)).snapshot()
    assert snapshot is not None
    owned = snapshot.copy()
    moved = torch.from_numpy(owned).to(device)
    moved.zero_()
    assert np.array_equal(owned, snapshot.frame) and owned.any()


# ---------------------------------------------------------------- what the recipe avoids


@pytest.mark.parametrize("convert", ["from_numpy", "as_tensor"])
def test_from_numpy_on_the_published_frame_writes_through_to_every_consumer(convert: str) -> None:
    engine = published(frames2py.EventCount())
    snapshot = engine.snapshot()
    assert snapshot is not None and not snapshot.frame.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        alias = getattr(torch, convert)(snapshot.frame)
    alias.fill_(99)
    assert (engine.snapshot().frame == 99).all()  # another consumer's read
    assert not snapshot.frame.flags.writeable


def test_from_dlpack_on_the_published_frame_writes_through_to_every_consumer() -> None:
    engine = published(frames2py.EventCount())
    snapshot = engine.snapshot()
    assert snapshot is not None
    alias = torch.from_dlpack(snapshot.frame)
    alias.fill_(99)
    assert (engine.snapshot().frame == 99).all()
    assert not snapshot.frame.flags.writeable


def test_from_numpy_warns_once_per_process_and_from_dlpack_and_the_copy_never_warn() -> None:
    code = """
import warnings, numpy as np, torch, frames2py
engine = frames2py.Engine((4, 3), "event_count", snapshot_interval_ms=0)
ev = np.zeros(3, dtype=frames2py.EVENT_DTYPE); ev["t"] = [1, 2, 3]
engine.ingest(ev)
frame = engine.snapshot().frame
counts = []
for convert in (lambda: torch.from_dlpack(frame), lambda: torch.from_numpy(engine.snapshot().copy()),
                lambda: torch.from_numpy(frame), lambda: torch.from_numpy(frame)):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert()
    counts.append(sum(issubclass(w.category, UserWarning) and "not writable" in str(w.message) for w in caught))
print(counts)
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[0, 0, 1, 0]"


def test_importing_frames2py_and_its_public_modules_imports_no_torch() -> None:
    code = """
import importlib, pkgutil, sys
import frames2py
for info in pkgutil.walk_packages(frames2py.__path__, "frames2py."):
    if not any(part.startswith("_") for part in info.name.split(".")):
        importlib.import_module(info.name)
assert "torch" not in sys.modules, "frames2py imported torch"
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
