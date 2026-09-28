"""``frames2py.recorder``: round trip through the HDF5 adapter, the file layout, byte determinism,
validation, paths and finalisation."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frames2py import EVENT_DTYPE
from tests import recordings
from tests.adapters.backends import require_backend
from tests.adapters.test_evt_open import OPENEB_SPARKLERS_FIRST_100K_SHA256

DATA = Path(__file__).resolve().parent.parent / "data"
LIMIT = 1 << 63
COMPRESSIONS = [None, "gzip", "blosc"]


@pytest.fixture(autouse=True)
def _h5() -> None:
    require_backend("h5py", "hdf5plugin")


def make_events(n: int, seed: int = 0, width: int = 1280, height: int = 720) -> np.ndarray:
    """Events built directly as ``EVENT_DTYPE``, independent of any Frames2Py code."""
    rng = np.random.default_rng(seed)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.cumsum(rng.integers(0, 3, n), dtype=np.uint64) + np.uint64(1_000_000)
    events["x"] = rng.integers(0, width, n)
    events["y"] = rng.integers(0, height, n)
    events["p"] = rng.integers(0, 2, n)
    return events


def split(events: np.ndarray, sizes: Sequence[int]) -> Iterator[np.ndarray]:
    """*events* in consecutive pieces of the given sizes, then the rest (possibly empty)."""
    start = 0
    for size in sizes:
        yield events[start : start + size]
        start += size
    yield events[start:]


def record(path: Path, pieces: Iterator[np.ndarray] | Sequence[np.ndarray], **kwargs: Any) -> Path:
    from frames2py import recorder

    kwargs.setdefault("sensor_size", (1280, 720))
    with recorder.open(path, **kwargs) as rec:
        for piece in pieces:
            rec.write(piece)
    return path


def read_back(path: Path, group: str = "events") -> np.ndarray:
    from frames2py.adapters import hdf5

    with hdf5.open(path, group=group) as reader:
        batches = list(reader)
    return np.concatenate(batches) if batches else np.empty(0, dtype=EVENT_DTYPE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PLANS: dict[str, list[int]] = {
    "one write": [],
    "300k then the rest": [300_000],
    "100k then varied": [100_000, 1, 65_535, 65_536, 65_537, 2, 30_000],
    "7777 at a time": [7_777] * 60,
    "irregular with empty writes": [0, 1, 0, 65_535, 3, 131_072, 0, 12_345, 99_999, 0],
}


class TestRoundTrip:
    @pytest.mark.parametrize("compression", COMPRESSIONS)
    @pytest.mark.parametrize("plan", sorted(PLANS))
    def test_events_come_back_identical(self, tmp_path: Path, compression: str | None, plan: str) -> None:
        events = make_events(470_001, seed=1)
        path = record(tmp_path / "r.h5", split(events, PLANS[plan]), compression=compression)
        assert read_back(path).tobytes() == events.tobytes()

    @pytest.mark.parametrize("writes", [[], [0], [0, 0, 0]])
    def test_an_empty_recording(self, tmp_path: Path, writes: list[int]) -> None:
        empty = np.empty(0, dtype=EVENT_DTYPE)
        path = record(tmp_path / "r.h5", [empty[:n] for n in writes])
        assert len(read_back(path)) == 0

    def test_one_event(self, tmp_path: Path) -> None:
        events = make_events(1)
        assert read_back(record(tmp_path / "r.h5", [events])).tobytes() == events.tobytes()

    def test_order_backward_timestamps_and_out_of_bounds_events_are_kept(self, tmp_path: Path) -> None:
        events = np.array(
            [(50, 1, 1, 1), (10, 2, 2, 0), (10, 2, 2, 0), (LIMIT - 1, 65_535, 65_535, 255), (0, 9, 9, 7), (30, 0, 0, 1)],
            dtype=EVENT_DTYPE,
        )
        path = record(tmp_path / "r.h5", [events[:2], events[2:5], events[5:]], sensor_size=(4, 4))
        assert read_back(path).tolist() == events.tolist()

    def test_extra_fields_are_ignored(self, tmp_path: Path) -> None:
        wide = np.zeros(3, dtype=[("q", "f8"), ("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("z", "i4")])
        wide["t"], wide["x"], wide["y"], wide["p"], wide["q"] = [3, 1, 2], [4, 5, 6], [7, 8, 9], [0, 1, 2], 1.5
        got = read_back(record(tmp_path / "r.h5", [wide]))
        assert got.tolist() == [(3, 4, 7, 0), (1, 5, 8, 1), (2, 6, 9, 2)]

    def test_a_committed_real_fixture(self, tmp_path: Path) -> None:
        from frames2py.adapters import evt

        with evt.open(DATA / "sparklers_100k.evt2.raw", sensor_size=(640, 480)) as reader:
            path = record(tmp_path / "r.h5", reader, sensor_size=(640, 480))
        assert hashlib.sha256(read_back(path).tobytes()).hexdigest() == OPENEB_SPARKLERS_FIRST_100K_SHA256


class TestLayout:
    """The file as plain h5py sees it, independent of Frames2Py's reader."""

    @pytest.mark.parametrize("compression", COMPRESSIONS)
    def test_datasets_attributes_and_nothing_else(self, tmp_path: Path, compression: str | None) -> None:
        import h5py

        events = make_events(70_000, seed=2)
        path = record(tmp_path / "r.h5", [events], sensor_size=(346, 260), compression=compression)
        names: list[str] = []
        with h5py.File(path, "r") as f:
            f.visit(names.append)
            group = f["events"]
            expected = {"t": "<u8", "x": "<u2", "y": "<u2", "p": "|u1"}
            for name, dtype in expected.items():
                dataset = group[name]
                assert (dataset.dtype.str, dataset.shape, dataset.maxshape, dataset.chunks) == (
                    dtype, (70_000,), (None,), (65_536,))
                assert dataset[...].tobytes() == np.ascontiguousarray(events[name]).tobytes()
            attrs = {key: group.attrs[key] for key in group.attrs}
            assert set(f.attrs) == set()
        assert sorted(names) == ["events", "events/p", "events/t", "events/x", "events/y"]
        assert attrs == {"sensor_width": 346, "sensor_height": 260, "frames2py_format_version": 1}
        assert {type(value) for value in attrs.values()} == {np.int64}

    @pytest.mark.parametrize(("compression", "filters"), [(None, set()), ("gzip", {1, 2}), ("blosc", {32001})])
    def test_filters(self, tmp_path: Path, compression: str | None, filters: set[int]) -> None:
        import h5py

        path = record(tmp_path / "r.h5", [make_events(10)], compression=compression)
        with h5py.File(path, "r") as f:
            for name in ("t", "x", "y", "p"):
                plist = f["events"][name].id.get_create_plist()
                assert {plist.get_filter(i)[0] for i in range(plist.get_nfilters())} == filters

    @pytest.mark.parametrize(("compression", "readable"), [(None, True), ("gzip", True), ("blosc", False)])
    def test_plain_h5py_without_hdf5plugin(self, tmp_path: Path, compression: str | None, readable: bool) -> None:
        events = make_events(1_000, seed=3)
        path = record(tmp_path / "r.h5", [events], compression=compression)
        code = (
            "import sys; sys.modules['hdf5plugin'] = None; import h5py, numpy as np\n"
            f"f = h5py.File({str(path)!r}, 'r')\n"
            "try:\n    t = f['events/t'][...]\nexcept OSError:\n    print('unreadable')\nelse:\n    print(t.sum())\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        assert out == (str(int(events["t"].sum())) if readable else "unreadable")

    @pytest.mark.parametrize(("group", "where"), [("a/b", "a/b"), ("/", "/"), ("/events2", "events2")])
    def test_group(self, tmp_path: Path, group: str, where: str) -> None:
        import h5py

        events = make_events(5)
        path = record(tmp_path / "r.h5", [events], group=group)
        with h5py.File(path, "r") as f:
            assert int(f[where].attrs["frames2py_format_version"]) == 1
        assert read_back(path, group=group).tobytes() == events.tobytes()


class TestDeterminism:
    @pytest.mark.parametrize("compression", COMPRESSIONS)
    def test_bytes_do_not_depend_on_how_writes_are_split(self, tmp_path: Path, compression: str | None) -> None:
        events = make_events(470_001, seed=4)
        digests = {
            plan: sha256(record(tmp_path / f"{index}.h5", split(events, sizes), compression=compression))
            for index, (plan, sizes) in enumerate(sorted(PLANS.items()))
        }
        assert len(set(digests.values())) == 1, digests

    def test_identical_across_paths_and_directories(self, tmp_path: Path) -> None:
        (tmp_path / "other").mkdir()
        events = make_events(100_000, seed=5)
        first = record(tmp_path / "a.h5", [events])
        second = record(tmp_path / "other" / "b.h5", split(events, [3, 70_000]))
        assert sha256(first) == sha256(second)

    def test_configuration_changes_the_bytes(self, tmp_path: Path) -> None:
        events = make_events(1_000, seed=6)
        a = record(tmp_path / "a.h5", [events], sensor_size=(1280, 720))
        b = record(tmp_path / "b.h5", [events], sensor_size=(1280, 721))
        assert sha256(a) != sha256(b)


class TestWriteValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            [(1, 2, 3, 0)],
            np.zeros(3, dtype=[("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")]),
            np.zeros(3, dtype=[("t", ">u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")]),
            np.zeros(3, dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2")]),
            np.zeros((2, 2), dtype=EVENT_DTYPE),
            np.zeros(6, dtype=EVENT_DTYPE)[::2],
            np.zeros(3, dtype="u8"),
        ],
        ids=["list", "t-u4", "t-big-endian", "no-p", "2-d", "strided", "not-structured"],
    )
    def test_malformed_input_raises_type_error_and_records_nothing(self, tmp_path: Path, bad: Any) -> None:
        from frames2py import recorder

        good = make_events(10)
        with recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720)) as rec:
            rec.write(good)
            with pytest.raises(TypeError):
                rec.write(bad)
        assert read_back(tmp_path / "r.h5").tobytes() == good.tobytes()

    @pytest.mark.parametrize("at", [0, 1, 70_000, 99_999])
    def test_a_timestamp_at_2_63_rejects_the_whole_call(self, tmp_path: Path, at: int) -> None:
        from frames2py import recorder

        before, bad = make_events(40_000, seed=7), make_events(100_000, seed=8)
        bad["t"][at] = LIMIT
        with recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720)) as rec:
            rec.write(before)
            with pytest.raises(ValueError, match="2\\*\\*63"):
                rec.write(bad)
            after = make_events(30_000, seed=9)
            rec.write(after)
        assert read_back(tmp_path / "r.h5").tobytes() == np.concatenate([before, after]).tobytes()

    def test_write_after_close(self, tmp_path: Path) -> None:
        from frames2py import recorder

        rec = recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720))
        rec.close()
        rec.close()
        with pytest.raises(ValueError, match="closed"):
            rec.write(make_events(1))


class TestOpenArguments:
    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"sensor_size": (0, 720)}, ValueError),
            ({"sensor_size": (1280,)}, ValueError),
            ({"sensor_size": None}, ValueError),
            ({"sensor_size": (True, 720)}, TypeError),
            ({"sensor_size": (1280.0, 720)}, TypeError),
            ({"sensor_size": (1280, 720), "group": 3}, TypeError),
            ({"sensor_size": (1280, 720), "group": ""}, ValueError),
            ({"sensor_size": (1280, 720), "compression": "lzf"}, ValueError),
            ({"sensor_size": (1280, 720), "compression": "GZIP"}, ValueError),
            ({"sensor_size": (1280, 720), "compression": 4}, ValueError),
            ({"sensor_size": (1280, 720), "overwrite": 1}, TypeError),
        ],
    )
    def test_invalid_arguments_leave_no_file(self, tmp_path: Path, kwargs: dict[str, Any], error: type) -> None:
        from frames2py import recorder

        with pytest.raises(error):
            recorder.open(tmp_path / "r.h5", **kwargs)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("path", [b"r.h5", 3, None])
    def test_path_type(self, path: Any) -> None:
        from frames2py import recorder

        with pytest.raises(TypeError):
            recorder.open(path, sensor_size=(1280, 720))


class TestPaths:
    def test_existing_target_without_overwrite(self, tmp_path: Path) -> None:
        from frames2py import recorder

        target = tmp_path / "r.h5"
        target.write_bytes(b"keep me")
        with pytest.raises(FileExistsError):
            recorder.open(target, sensor_size=(1280, 720))
        assert target.read_bytes() == b"keep me"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["r.h5"]

    def test_directory_missing_parent_and_read_only_directory(self, tmp_path: Path) -> None:
        from frames2py import recorder

        with pytest.raises(IsADirectoryError):
            recorder.open(tmp_path, sensor_size=(1280, 720), overwrite=True)
        with pytest.raises(FileNotFoundError):
            recorder.open(tmp_path / "missing" / "r.h5", sensor_size=(1280, 720))
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            if os.access(locked, os.W_OK):
                pytest.skip("running with permissions that ignore the directory mode")
            with pytest.raises(PermissionError):
                recorder.open(locked / "r.h5", sensor_size=(1280, 720))
        finally:
            locked.chmod(stat.S_IRWXU)
        assert list(locked.iterdir()) == []

    def test_the_target_appears_only_when_the_recording_is_closed(self, tmp_path: Path) -> None:
        from frames2py import recorder

        target = tmp_path / "r.h5"
        rec = recorder.open(target, sensor_size=(1280, 720))
        rec.write(make_events(100_000))
        assert not target.exists()
        (partial,) = tmp_path.iterdir()
        assert partial.name.startswith(".r.h5.") and partial.name.endswith(".partial")
        rec.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["r.h5"]
        assert len(read_back(target)) == 100_000

    def test_overwrite_replaces_the_old_recording_only_at_close(self, tmp_path: Path) -> None:
        from frames2py import recorder

        old, new = make_events(1_000, seed=10), make_events(2_000, seed=11)
        target = record(tmp_path / "r.h5", [old])
        rec = recorder.open(target, sensor_size=(1280, 720), overwrite=True)
        rec.write(new)
        assert read_back(target).tobytes() == old.tobytes()
        rec.close()
        assert read_back(target).tobytes() == new.tobytes()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["r.h5"]

    def test_a_file_that_appears_during_recording_is_not_overwritten(self, tmp_path: Path) -> None:
        from frames2py import recorder

        target, events = tmp_path / "r.h5", make_events(500)
        rec = recorder.open(target, sensor_size=(1280, 720))
        rec.write(events)
        target.write_bytes(b"someone else's")
        with pytest.raises(FileExistsError) as info:
            rec.close()
        assert target.read_bytes() == b"someone else's"
        (partial,) = [p for p in tmp_path.iterdir() if p.name != "r.h5"]
        assert str(partial) in str(info.value)
        assert read_back(partial).tobytes() == events.tobytes()


class TestFinalisation:
    @pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
    def test_an_exception_in_the_with_block_finalises_the_recording(self, tmp_path: Path, exception: type) -> None:
        from frames2py import recorder

        events = make_events(150_000, seed=12)
        with pytest.raises(exception):
            with recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720)) as rec:
                for piece in split(events, [70_000, 1]):
                    rec.write(piece)
                raise exception()
        assert read_back(tmp_path / "r.h5").tobytes() == events.tobytes()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["r.h5"]

    def test_an_interrupted_overwrite_replaces_the_old_recording(self, tmp_path: Path) -> None:
        from frames2py import recorder

        old, new = make_events(1_000, seed=13), make_events(3_000, seed=14)
        record(tmp_path / "r.h5", [old])
        with pytest.raises(KeyboardInterrupt):
            with recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720), overwrite=True) as rec:
                rec.write(new)
                raise KeyboardInterrupt
        assert read_back(tmp_path / "r.h5").tobytes() == new.tobytes()

    @staticmethod
    def count_calls(monkeypatch: pytest.MonkeyPatch, method: str, pieces: list[np.ndarray], path: Path) -> tuple[int, int]:
        """Backend calls of *method* made by the writes, and by the close."""
        import h5py

        from frames2py import recorder

        original, calls = getattr(h5py.Dataset, method), {"n": 0}

        def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(h5py.Dataset, method, counting)
        rec = recorder.open(path, sensor_size=(1280, 720))
        for piece in pieces:
            rec.write(piece)
        in_writes = calls["n"]
        rec.close()
        monkeypatch.setattr(h5py.Dataset, method, original)
        return in_writes, calls["n"] - in_writes

    @staticmethod
    def interrupt_at(monkeypatch: pytest.MonkeyPatch, method: str, k: int) -> None:
        import h5py

        original, calls = getattr(h5py.Dataset, method), {"n": 0}

        def interrupting(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == k:
                raise KeyboardInterrupt
            return original(self, *args, **kwargs)

        monkeypatch.setattr(h5py.Dataset, method, interrupting)

    PIECES = [10_000, 100_000, 65_536, 1]

    @pytest.mark.parametrize("method", ["__setitem__", "resize"])
    def test_an_interrupt_inside_write_leaves_a_prefix_of_what_was_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        # Interrupting the k-th backend call, for every k inside the writes, covers every point
        # inside every chunk append.
        from frames2py import recorder

        pieces = list(split(make_events(300_000, seed=15), self.PIECES))
        stream = np.concatenate(pieces)
        in_writes, _ = self.count_calls(monkeypatch, method, pieces, tmp_path / "count.h5")
        assert in_writes >= 16
        for k in range(1, in_writes + 1):
            self.interrupt_at(monkeypatch, method, k)
            path, completed = tmp_path / f"{k}.h5", 0
            with pytest.raises(KeyboardInterrupt):
                with recorder.open(path, sensor_size=(1280, 720)) as rec:
                    for piece in pieces:
                        rec.write(piece)
                        completed += len(piece)
            monkeypatch.undo()
            got = read_back(path)
            assert completed <= len(got) <= len(stream), k
            assert got.tobytes() == stream[: len(got)].tobytes(), k

    @pytest.mark.parametrize("method", ["__setitem__", "resize"])
    def test_an_interrupt_while_finishing_leaves_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        from frames2py import recorder

        pieces = list(split(make_events(300_000, seed=15), self.PIECES))
        in_writes, in_close = self.count_calls(monkeypatch, method, pieces, tmp_path / "count.h5")
        (tmp_path / "count.h5").unlink()
        assert in_close >= 1
        for k in range(in_writes + 1, in_writes + in_close + 1):
            rec = recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720))
            for piece in pieces:
                rec.write(piece)
            self.interrupt_at(monkeypatch, method, k - in_writes)
            with pytest.raises(KeyboardInterrupt):
                rec.close()
            monkeypatch.undo()
            assert list(tmp_path.iterdir()) == [], k

    def test_a_failure_while_finishing_removes_the_temporary_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import h5py

        from frames2py import recorder

        rec = recorder.open(tmp_path / "r.h5", sensor_size=(1280, 720))
        rec.write(make_events(10))

        def failing(self: Any, *args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(h5py.Dataset, "__setitem__", failing)
        with pytest.raises(OSError, match="disk full"):
            rec.close()
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(ValueError, match="closed"):
            rec.write(make_events(1))


@pytest.mark.recordings
class TestRealRecording:
    def test_a_recording_round_trips_to_the_openeb_digest(self, tmp_path: Path) -> None:
        from frames2py.adapters import evt

        try:
            source = recordings.path("sparklers.raw")
        except (FileNotFoundError, ValueError) as exc:
            pytest.fail(str(exc))
        with evt.open(source, sensor_size=(640, 480)) as reader:
            path = record(tmp_path / "r.h5", reader, sensor_size=(640, 480))
        got = read_back(path)
        assert len(got) == 521_252
        assert hashlib.sha256(got.tobytes()).hexdigest() == (
            "58773c06cd3098d13e0f2781e1134449d2f01ed498d4975aabc2c4e5ec81d44b")
