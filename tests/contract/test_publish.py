"""The public snapshot publisher: ``SnapshotPublisher`` and ``SeqlockPublisher``."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import numpy as np
import pytest

from tests.contract.api import impl

publish = importlib.import_module(f"{impl.__name__}.publish")


def _meta(sequence: int, watermark: int | None = None) -> Any:
    return impl.SnapshotMeta(watermark=watermark, sequence=sequence)


def _published(publisher: Any, value: int, sequence: int) -> None:
    buffer = publisher.begin_write()
    buffer[...] = value
    publisher.end_write(_meta(sequence, watermark=value))


@pytest.fixture
def publisher() -> Any:
    return publish.SeqlockPublisher((3, 4), np.dtype(np.uint32))


def test_constructor_takes_shape_and_dtype_only() -> None:
    assert list(inspect.signature(publish.SeqlockPublisher).parameters) == ["shape", "dtype"]


def test_seqlock_publisher_satisfies_the_protocol(publisher: Any) -> None:
    assert isinstance(publisher, publish.SnapshotPublisher)
    for method, params in (("begin_write", []), ("end_write", ["meta"]), ("read", []), ("reset", [])):
        assert list(inspect.signature(getattr(publisher, method)).parameters) == params


def test_nothing_to_read_before_the_first_publication(publisher: Any) -> None:
    assert publisher.read() is None


def test_read_returns_a_copy_of_the_latest_snapshot(publisher: Any) -> None:
    _published(publisher, 7, sequence=1)
    frame, meta = publisher.read()
    assert frame.shape == (3, 4) and frame.dtype == np.uint32
    assert (frame == 7).all()
    assert (meta.sequence, meta.watermark) == (1, 7)
    frame[...] = 0
    again, _ = publisher.read()
    assert (again == 7).all()


def test_a_write_in_progress_is_never_visible(publisher: Any) -> None:
    _published(publisher, 7, sequence=1)
    buffer = publisher.begin_write()
    buffer[...] = 9
    frame, meta = publisher.read()
    assert (frame == 7).all() and meta.sequence == 1
    publisher.end_write(_meta(2, watermark=9))
    frame, meta = publisher.read()
    assert (frame == 9).all() and meta.sequence == 2


def test_successive_publications(publisher: Any) -> None:
    for sequence in range(1, 6):
        _published(publisher, sequence * 10, sequence=sequence)
        frame, meta = publisher.read()
        assert (frame == sequence * 10).all() and meta.sequence == sequence


def test_reset_forgets_the_snapshot(publisher: Any) -> None:
    _published(publisher, 7, sequence=1)
    publisher.reset()
    assert publisher.read() is None
    _published(publisher, 8, sequence=2)
    frame, meta = publisher.read()
    assert (frame == 8).all() and meta.sequence == 2
