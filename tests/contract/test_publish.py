"""The public snapshot publisher and snapshot type: ``SnapshotPublisher``,
``ImmutablePublisher`` and ``Snapshot``."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.contract.api import impl

publish = importlib.import_module(f"{impl.__name__}.publish")

SHAPE = (3, 4)


def _meta(sequence: int, watermark: int | None = None) -> Any:
    return impl.SnapshotMeta(watermark=watermark, sequence=sequence)


def _published(publisher: Any, value: int, sequence: int) -> None:
    buffer = publisher.begin_write()
    buffer[...] = value
    publisher.end_write(_meta(sequence, watermark=value))


@pytest.fixture
def publisher() -> Any:
    return publish.ImmutablePublisher(SHAPE, np.dtype(np.uint32))


@pytest.fixture
def snapshot(publisher: Any) -> Any:
    buffer = publisher.begin_write()
    buffer[...] = np.arange(12, dtype=np.uint32).reshape(SHAPE)
    publisher.end_write(_meta(1, watermark=11))
    return publisher.read()


def test_constructor_takes_shape_and_dtype_only() -> None:
    assert list(inspect.signature(publish.ImmutablePublisher).parameters) == ["shape", "dtype"]


def test_immutable_publisher_satisfies_the_protocol(publisher: Any) -> None:
    assert isinstance(publisher, publish.SnapshotPublisher)
    for method, params in (("begin_write", []), ("end_write", ["meta"]), ("read", []), ("reset", [])):
        assert list(inspect.signature(getattr(publisher, method)).parameters) == params


def test_nothing_to_read_before_the_first_publication(publisher: Any) -> None:
    assert publisher.read() is None


def test_read_returns_the_latest_publication_shared(publisher: Any) -> None:
    _published(publisher, 7, sequence=1)
    snap = publisher.read()
    assert isinstance(snap, publish.Snapshot)
    assert snap.frame.shape == SHAPE and snap.frame.dtype == np.uint32
    assert (snap.frame == 7).all()
    assert (snap.meta.sequence, snap.meta.watermark) == (1, 7)
    assert publisher.read() is snap


def test_the_published_frame_is_read_only(snapshot: Any) -> None:
    with pytest.raises(ValueError):
        snapshot.frame[0, 0] = 1
    with pytest.raises(ValueError):
        snapshot.frame.flags.writeable = True


def test_a_write_in_progress_is_never_visible(publisher: Any) -> None:
    _published(publisher, 7, sequence=1)
    buffer = publisher.begin_write()
    buffer[...] = 9
    snap = publisher.read()
    assert (snap.frame == 7).all() and snap.meta.sequence == 1
    publisher.end_write(_meta(2, watermark=9))
    snap = publisher.read()
    assert (snap.frame == 9).all() and snap.meta.sequence == 2


def test_successive_publications_leave_earlier_snapshots_unchanged(publisher: Any) -> None:
    held = []
    for sequence in range(1, 6):
        _published(publisher, sequence * 10, sequence=sequence)
        snap = publisher.read()
        assert (snap.frame == sequence * 10).all() and snap.meta.sequence == sequence
        held.append(snap)
    for sequence, snap in enumerate(held, start=1):
        assert (snap.frame == sequence * 10).all() and snap.meta.sequence == sequence


def test_reset_forgets_the_snapshot(publisher: Any) -> None:
    _published(publisher, 7, sequence=1)
    publisher.reset()
    assert publisher.read() is None
    _published(publisher, 8, sequence=2)
    snap = publisher.read()
    assert (snap.frame == 8).all() and snap.meta.sequence == 2


class TestCopy:
    def test_copy_is_writable_and_independent(self, snapshot: Any) -> None:
        copy = snapshot.copy()
        np.testing.assert_array_equal(copy, snapshot.frame)
        assert copy.flags.writeable and copy.flags.c_contiguous and copy.dtype == snapshot.frame.dtype
        copy[...] = 0
        assert snapshot.frame.sum() == 66
        other = snapshot.copy()
        assert other is not copy and other.sum() == 66

    def test_copy_into_out_fills_and_returns_it(self, snapshot: Any) -> None:
        out = np.full(SHAPE, 0xFFFFFFFF, dtype=np.uint32)
        assert snapshot.copy(out=out) is out
        np.testing.assert_array_equal(out, snapshot.frame)

    @pytest.mark.parametrize(
        ("make_out", "error"),
        [
            (lambda: np.full((4, 3), 0xA5A5A5A5, dtype=np.uint32), ValueError),  # shape
            (lambda: np.full((3, 4, 1), 0xA5A5A5A5, dtype=np.uint32), ValueError),  # rank
            (lambda: np.full(SHAPE, 0xA5A5A5A5, dtype=np.uint64), TypeError),  # dtype
            (lambda: np.full(SHAPE, 0xA5A5A5A5, dtype=">u4"), TypeError),  # byte order
            (lambda: np.full(SHAPE, 0xA5, dtype=np.float32), TypeError),  # castable dtype
            (lambda: np.full((4, 3), 0xA5A5A5A5, dtype=np.uint32).T, ValueError),  # F-contiguous
            (lambda: np.full((3, 8), 0xA5A5A5A5, dtype=np.uint32)[:, ::2], ValueError),  # strided
            (lambda: _read_only(np.full(SHAPE, 0xA5A5A5A5, dtype=np.uint32)), ValueError),  # read-only
        ],
        ids=["shape", "rank", "dtype", "byte-order", "castable-dtype", "fortran-order", "strided", "read-only"],
    )
    def test_invalid_out_is_rejected_before_anything_is_written(
        self, snapshot: Any, make_out: Callable[[], Any], error: type[Exception]
    ) -> None:
        out = make_out()
        before = out.copy()
        with pytest.raises(error):
            snapshot.copy(out=out)
        assert out.tobytes() == before.tobytes() and out.dtype == before.dtype

    @pytest.mark.parametrize("out", [[[0] * 4] * 3, b"\x00" * 48], ids=["list", "bytes"])
    def test_out_must_be_an_ndarray(self, snapshot: Any, out: Any) -> None:
        with pytest.raises(TypeError):
            snapshot.copy(out=out)

    def test_a_writable_buffer_that_is_not_an_ndarray_is_rejected_unchanged(self, snapshot: Any) -> None:
        storage = bytearray(b"\xa5" * 48)
        out = memoryview(storage).cast("I", SHAPE)
        with pytest.raises(TypeError):
            snapshot.copy(out=out)
        assert storage == bytearray(b"\xa5" * 48)


def _read_only(array: Any) -> Any:
    array.flags.writeable = False
    return array
