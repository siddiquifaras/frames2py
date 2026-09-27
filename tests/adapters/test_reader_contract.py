"""The reader contract every adapter's ``open()`` shares: batches, single pass, closing, errors."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from frames2py import EVENT_DTYPE
from frames2py._events import validate
from tests.adapters.cases import CASES, Written


@pytest.fixture(params=sorted(CASES))
def case(request: pytest.FixtureRequest, tmp_path: Path) -> Written:
    written: Written = CASES[request.param](tmp_path)
    return written


def read(case: Written, **kwargs: object) -> list[np.ndarray]:
    with case.open(case.path, **case.open_kwargs, **kwargs) as reader:
        return list(reader)


class TestBatches:
    def test_arrays_are_event_dtype_1d_contiguous_and_never_empty(self, case: Written) -> None:
        batches = read(case)
        assert batches
        for batch in batches:
            assert batch.dtype == EVENT_DTYPE
            assert batch.ndim == 1 and batch.flags.c_contiguous and len(batch) > 0
            validate(batch)
        assert np.concatenate(batches).tolist() == case.expected.tolist()

    @pytest.mark.parametrize("size", [1, 3, 7, 1000])
    def test_batch_size_gives_that_many_events_except_the_last(self, case: Written, size: int) -> None:
        batches = read(case, batch_size=size)
        assert all(len(b) == size for b in batches[:-1])
        assert 1 <= len(batches[-1]) <= size
        assert np.concatenate(batches).tolist() == case.expected.tolist()
        for batch in batches:
            assert batch.dtype == EVENT_DTYPE and batch.flags.c_contiguous

    @pytest.mark.parametrize("bad", [0, -1])
    def test_batch_size_below_one_is_rejected(self, case: Written, bad: int) -> None:
        with pytest.raises(ValueError):
            case.open(case.path, **case.open_kwargs, batch_size=bad)

    @pytest.mark.parametrize("bad", [1.5, "3", True])
    def test_batch_size_must_be_an_int(self, case: Written, bad: object) -> None:
        with pytest.raises(TypeError):
            case.open(case.path, **case.open_kwargs, batch_size=bad)

    @pytest.mark.parametrize("bad", [(0, 480), (640, -1), (640,), (640, 480, 2)])
    def test_invalid_sensor_size_is_rejected(self, case: Written, bad: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            case.open(case.path, **case.open_kwargs, sensor_size=bad)

    def test_arrays_share_no_memory_and_writing_one_changes_nothing_else(self, case: Written) -> None:
        batches = read(case, batch_size=4)
        for a, b in zip(batches, batches[1:]):
            assert not np.shares_memory(a, b)
        for batch in batches:
            batch["x"] = 1
        assert np.concatenate(read(case, batch_size=4)).tolist() == case.expected.tolist()

    def test_path_may_be_str_or_pathlike(self, case: Written) -> None:
        with case.open(str(case.path), **case.open_kwargs) as reader:
            assert np.concatenate(list(reader)).tolist() == case.expected.tolist()


class TestLifecycle:
    def test_second_iteration_raises(self, case: Written) -> None:
        with case.open(case.path, **case.open_kwargs) as reader:
            list(reader)
            with pytest.raises(RuntimeError):
                iter(reader)

    def test_second_iteration_raises_while_the_first_is_running(self, case: Written) -> None:
        with case.open(case.path, **case.open_kwargs, batch_size=1) as reader:
            first = iter(reader)
            next(first)
            with pytest.raises(RuntimeError):
                iter(reader)
            assert len(list(first)) == len(case.expected) - 1

    def test_iteration_after_close_raises(self, case: Written) -> None:
        reader = case.open(case.path, **case.open_kwargs)
        reader.close()
        with pytest.raises(ValueError):
            iter(reader)

    def test_next_after_close_raises(self, case: Written) -> None:
        reader = case.open(case.path, **case.open_kwargs, batch_size=1)
        it = iter(reader)
        next(it)
        reader.close()
        with pytest.raises(ValueError):
            next(it)

    def test_with_block_closes_and_close_is_idempotent(self, case: Written) -> None:
        with case.open(case.path, **case.open_kwargs) as reader:
            assert not reader.closed
        assert reader.closed
        reader.close()
        assert reader.closed

    def test_no_threads_are_started(self, case: Written) -> None:
        before = threading.active_count()
        with case.open(case.path, **case.open_kwargs) as reader:
            list(reader)
            assert threading.active_count() == before
        assert threading.active_count() == before


class TestPathErrors:
    def test_missing_path(self, case: Written, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            case.open(tmp_path / "missing" / case.path.name, **case.open_kwargs)

    def test_directory(self, case: Written, tmp_path: Path) -> None:
        with pytest.raises(OSError) as info:
            case.open(tmp_path, **case.open_kwargs)
        assert not isinstance(info.value, FileNotFoundError)

    @pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                        reason="permission bits don't stop this user")
    def test_unreadable_file(self, case: Written) -> None:
        case.path.chmod(0)
        try:
            with pytest.raises(PermissionError):
                case.open(case.path, **case.open_kwargs)
        finally:
            case.path.chmod(0o644)
