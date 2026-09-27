"""``frames2py.adapters.hdf5.open``: the DSEC-layout fixture, the schema, t_offset, value ranges, filters, errors."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frames2py import EVENT_DTYPE
from tests.adapters.backends import require_backend
from tests.adapters.test_evt_open import OPENEB_SPARKLERS_FIRST_100K_SHA256

DATA = Path(__file__).resolve().parent.parent / "data"
FIXTURE = DATA / "sparklers_100k.h5"
LIMIT = 1 << 63


@pytest.fixture(autouse=True)
def _h5() -> None:
    require_backend("h5py", "hdf5plugin")


def read_all(path: Path, **kwargs: Any) -> tuple[np.ndarray, list[np.ndarray], Any]:
    from frames2py.adapters import hdf5

    with hdf5.open(path, **kwargs) as reader:
        batches = list(reader)
        size = reader.sensor_size
    events = np.concatenate(batches) if batches else np.empty(0, dtype=EVENT_DTYPE)
    return events, batches, size


def write(path: Path, group: str = "events", *, compression: Any = None, extra: dict[str, Any] | None = None,
          **columns: Any) -> Path:
    import h5py

    with h5py.File(path, "a") as f:
        node = f.require_group(group)
        for name, values in columns.items():
            node.create_dataset(name, data=np.asarray(values[0], dtype=values[1]) if isinstance(values, tuple) else values,
                                **({} if compression is None else compression))
        for name, value in (extra or {}).items():
            f.create_dataset(name, data=value)
    return path


def simple(path: Path, t: Any = (np.array([5, 6, 7]), "u4"), **overrides: Any) -> Path:
    columns = {"t": t, "x": (np.array([0, 1, 639]), "u2"), "y": (np.array([0, 2, 479]), "u2"), "p": (np.array([0, 1, 1]), "u1")}
    columns.update(overrides)
    return write(path, **columns)


class TestFixture:
    def test_dsec_layout_gives_the_source_events(self) -> None:
        # The same 100,000 events as the EVT2 excerpt: t stored relative to /t_offset, Blosc via hdf5plugin.
        events, _, size = read_all(FIXTURE, group="events", t_offset="/t_offset")
        assert size is None
        assert hashlib.sha256(events.tobytes()).hexdigest() == OPENEB_SPARKLERS_FIRST_100K_SHA256
        assert (int(events["t"].min()), int(events["t"].max())) == (913_716_224, 913_728_417)

    def test_without_t_offset_the_stored_relative_times_come_out(self) -> None:
        import h5py

        events, _, _ = read_all(FIXTURE, group="events")
        with h5py.File(FIXTURE) as f:
            offset = int(f["t_offset"][()])
        assert (int(events["t"].min()), int(events["t"].max())) == (0, 913_728_417 - offset)

    def test_int_t_offset_equals_the_dataset_path(self) -> None:
        by_path, _, _ = read_all(FIXTURE, group="events", t_offset="t_offset")
        by_int, _, _ = read_all(FIXTURE, group="events", t_offset=913_716_224)
        assert by_path.tobytes() == by_int.tobytes()

    def test_dataset_index_agrees_with_the_timestamps(self) -> None:
        import h5py

        events, _, _ = read_all(FIXTURE, group="events", t_offset="t_offset")
        with h5py.File(FIXTURE) as f:
            ms_to_idx, offset = f["ms_to_idx"][:], int(f["t_offset"][()])
        relative = events["t"].astype(np.int64) - offset
        for k, i in enumerate(ms_to_idx):
            assert relative[i] >= 1000 * k and (i == 0 or relative[i - 1] < 1000 * k)

    @pytest.mark.parametrize("batch_size", [None, 1, 4096, 100_000, 1_000_000])
    def test_batch_size_changes_only_the_boundaries(self, batch_size: int | None) -> None:
        events, batches, _ = read_all(FIXTURE, group="events", t_offset="t_offset", batch_size=batch_size)
        assert hashlib.sha256(events.tobytes()).hexdigest() == OPENEB_SPARKLERS_FIRST_100K_SHA256
        if batch_size is not None:
            assert {len(b) for b in batches[:-1]} <= {batch_size}

    def test_sensor_size_is_reported_as_given(self) -> None:
        assert read_all(FIXTURE, group="events", sensor_size=(640, 480))[2] == (640, 480)

    def test_reads_span_several_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from frames2py.adapters import hdf5

        monkeypatch.setattr(hdf5, "READ_EVENTS", 7777)
        events, batches, _ = read_all(FIXTURE, group="events", t_offset="t_offset")
        assert len(batches) == -(-100_000 // 7777)
        assert hashlib.sha256(events.tobytes()).hexdigest() == OPENEB_SPARKLERS_FIRST_100K_SHA256


class TestSchema:
    @pytest.mark.parametrize("group", ["events", "/events", "a/b"])
    def test_group_path(self, tmp_path: Path, group: str) -> None:
        path = write(tmp_path / "f.h5", group.strip("/") or "events", t=(np.array([1]), "u8"), x=(np.array([2]), "u2"),
                     y=(np.array([3]), "u2"), p=(np.array([1]), "u1"))
        assert read_all(path, group=group)[0].tolist() == [(1, 2, 3, 1)]

    def test_root_group(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.h5", "/", t=(np.array([1]), "u8"), x=(np.array([2]), "u2"),
                     y=(np.array([3]), "u2"), p=(np.array([1]), "u1"))
        assert read_all(path, group="/")[0].tolist() == [(1, 2, 3, 1)]

    @pytest.mark.parametrize("kind", ["missing-group", "not-a-group", "missing-field", "two-dimensional", "float-t",
                                      "float-x", "string-p", "compound", "unequal-lengths"])
    def test_other_layouts_are_rejected(self, tmp_path: Path, kind: str) -> None:
        from frames2py.adapters import hdf5

        path = tmp_path / "f.h5"
        if kind == "missing-group":
            simple(path)
            group = "elsewhere"
        elif kind == "not-a-group":
            simple(path)
            group = "events/t"
        elif kind == "missing-field":
            write(path, t=(np.array([1]), "u4"), x=(np.array([1]), "u2"), y=(np.array([1]), "u2"))
            group = "events"
        elif kind == "two-dimensional":
            simple(path, t=(np.array([[1, 2, 3]]), "u4"))
            group = "events"
        elif kind == "float-t":
            simple(path, t=(np.array([1.0, 2.0, 3.0]), "f8"))
            group = "events"
        elif kind == "float-x":
            simple(path, x=(np.array([1.0, 2.0, 3.0]), "f4"))
            group = "events"
        elif kind == "string-p":
            simple(path, p=(np.array([b"a", b"b", b"c"]), "S1"))
            group = "events"
        elif kind == "compound":
            import h5py

            with h5py.File(path, "w") as f:
                f.create_dataset("CD/events", data=np.zeros(3, dtype=[("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")]))
            group = "CD"
        else:
            simple(path, t=(np.array([5, 6]), "u4"))
            group = "events"
        with pytest.raises(ValueError):
            hdf5.open(path, group=group)

    def test_group_must_be_a_str(self, tmp_path: Path) -> None:
        from frames2py.adapters import hdf5

        with pytest.raises(TypeError):
            hdf5.open(simple(tmp_path / "f.h5"), group=b"events")  # type: ignore[arg-type]

    def test_empty_datasets(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", t=(np.array([], dtype="u4"), "u4"), x=(np.array([]), "u2"),
                      y=(np.array([]), "u2"), p=(np.array([]), "u1"))
        assert read_all(path, group="events")[1] == []

    def test_not_an_hdf5_file(self, tmp_path: Path) -> None:
        from frames2py.adapters import hdf5

        path = tmp_path / "f.h5"
        path.write_bytes(b"not hdf5" * 100)
        with pytest.raises(ValueError) as info:
            hdf5.open(path, group="events")
        assert isinstance(info.value.__cause__, OSError)


class TestValues:
    @pytest.mark.parametrize("dtype", ["u1", "u2", "u4", "u8", "i2", "i4", "i8"])
    def test_integer_columns_of_any_width(self, tmp_path: Path, dtype: str) -> None:
        path = simple(tmp_path / "f.h5", t=(np.array([5, 6, 7]), dtype), x=(np.array([0, 1, 100]), dtype),
                      y=(np.array([0, 2, 100]), dtype), p=(np.array([0, 1, 100]), dtype))
        assert read_all(path, group="events")[0].tolist() == [(5, 0, 0, 0), (6, 1, 2, 1), (7, 100, 100, 100)]

    def test_bool_polarity(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", p=(np.array([False, True, True]), "?"))
        assert read_all(path, group="events")[0]["p"].tolist() == [0, 1, 1]

    def test_polarity_values_are_copied(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", p=(np.array([0, 2, 255]), "u1"))
        assert read_all(path, group="events")[0]["p"].tolist() == [0, 2, 255]

    @pytest.mark.parametrize(("field", "values", "dtype"),
                             [("p", [0, -1, 1], "i1"), ("p", [0, 256, 1], "i2"), ("x", [0, 65536, 1], "i4"),
                              ("y", [-1, 0, 1], "i2")])
    def test_values_that_do_not_fit_are_refused(self, tmp_path: Path, field: str, values: list[int], dtype: str) -> None:
        from frames2py.adapters import hdf5

        path = simple(tmp_path / "f.h5", **{field: (np.array(values), dtype)})
        with hdf5.open(path, group="events") as reader, pytest.raises(ValueError, match="outside"):
            list(reader)

    def test_timestamps_in_order_and_out_of_order_are_kept(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", t=(np.array([9, 3, 12]), "u8"))
        assert read_all(path, group="events")[0]["t"].tolist() == [9, 3, 12]

    def test_negative_t_with_an_offset_that_lifts_it(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", t=(np.array([-5, 0, 5]), "i8"))
        assert read_all(path, group="events", t_offset=10)[0]["t"].tolist() == [5, 10, 15]

    def test_negative_offset(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", t=(np.array([LIMIT - 1, 100, 5]), "u8"))
        assert read_all(path, group="events", t_offset=-5)[0]["t"].tolist() == [LIMIT - 6, 95, 0]

    def test_largest_timestamps_are_exact(self, tmp_path: Path) -> None:
        path = simple(tmp_path / "f.h5", t=(np.array([LIMIT - 3, 0, 1]), "u8"))
        assert read_all(path, group="events", t_offset=2)[0]["t"].tolist() == [LIMIT - 1, 2, 3]

    @pytest.mark.parametrize(("t", "dtype", "offset"),
                             [([-1, 0, 1], "i8", None), ([5, 6, 7], "u4", -6), ([LIMIT, 0, 1], "u8", None),
                              ([LIMIT - 1, 0, 1], "u8", 1), ([(1 << 64) - 1, 0, 1], "u8", -(1 << 63))])
    def test_timestamps_outside_the_range_are_refused(self, tmp_path: Path, t: list[int], dtype: str,
                                                      offset: int | None) -> None:
        from frames2py.adapters import hdf5

        path = simple(tmp_path / "f.h5", t=(np.array(t, dtype=dtype), dtype))
        with hdf5.open(path, group="events", t_offset=offset) as reader, pytest.raises(ValueError, match="never clamped"):
            list(reader)

    def test_a_bad_batch_is_refused_after_the_good_ones(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from frames2py.adapters import hdf5

        monkeypatch.setattr(hdf5, "READ_EVENTS", 2)
        path = simple(tmp_path / "f.h5", t=(np.array([1, 2, -3]), "i8"))
        with hdf5.open(path, group="events") as reader:
            it = iter(reader)
            assert next(it)["t"].tolist() == [1, 2]
            with pytest.raises(ValueError):
                next(it)


class TestOffset:
    @pytest.mark.parametrize("value", [np.int64(10), np.uint32(10), 10])
    def test_integer_offsets(self, tmp_path: Path, value: Any) -> None:
        path = simple(tmp_path / "f.h5")
        assert read_all(path, group="events", t_offset=value)[0]["t"].tolist() == [15, 16, 17]

    @pytest.mark.parametrize("dtype", ["i8", "u8", "i4"])
    def test_scalar_dataset_offset(self, tmp_path: Path, dtype: str) -> None:
        path = simple(tmp_path / "f.h5")
        write(path, "meta", off=(np.array(10), dtype))
        assert read_all(path, group="events", t_offset="meta/off")[0]["t"].tolist() == [15, 16, 17]

    @pytest.mark.parametrize("kind", ["missing", "array", "float", "group"])
    def test_offset_dataset_must_be_a_scalar_integer(self, tmp_path: Path, kind: str) -> None:
        from frames2py.adapters import hdf5

        path = simple(tmp_path / "f.h5")
        data = {"array": np.array([10, 11]), "float": np.float64(10.0)}
        if kind in data:
            write(path, "meta", off=data[kind])
        target = {"missing": "meta/nothing", "group": "events"}.get(kind, "meta/off")
        with pytest.raises(ValueError, match="t_offset"):
            hdf5.open(path, group="events", t_offset=target)

    @pytest.mark.parametrize("value", [1.5, True, b"t_offset"])
    def test_offset_must_be_an_int_or_a_path(self, tmp_path: Path, value: Any) -> None:
        from frames2py.adapters import hdf5

        with pytest.raises(TypeError):
            hdf5.open(simple(tmp_path / "f.h5"), group="events", t_offset=value)


class TestFilters:
    @pytest.mark.parametrize("name", ["none", "gzip", "lzf", "blosc", "zstd"])
    def test_compressed_datasets(self, tmp_path: Path, name: str) -> None:
        import hdf5plugin

        options = {"none": None, "gzip": {"compression": "gzip"}, "lzf": {"compression": "lzf"},
                   "blosc": dict(hdf5plugin.Blosc()), "zstd": dict(hdf5plugin.Zstd())}[name]
        rng = np.random.default_rng(0)
        n = 5000
        t = np.cumsum(rng.integers(0, 5, n)).astype("u8")
        x, y, p = rng.integers(0, 640, n).astype("u2"), rng.integers(0, 480, n).astype("u2"), rng.integers(0, 2, n).astype("u1")
        path = write(tmp_path / "f.h5", compression=options, t=t, x=x, y=y, p=p)
        events = read_all(path, group="events")[0]
        assert events["t"].tolist() == t.tolist() and events["x"].tolist() == x.tolist()
        assert events["y"].tolist() == y.tolist() and events["p"].tolist() == p.tolist()

    def test_missing_mandatory_filter_is_refused_on_open(self, tmp_path: Path) -> None:
        # A gzip filter record is patched to an id no library registers, flagged mandatory.
        # h5py's default object headers (version 1) carry no checksum, and the filter record
        # is id, name length, flags, value count, then the name.
        from frames2py.adapters import hdf5

        path = write(tmp_path / "f.h5", compression={"compression": "gzip"}, t=np.arange(10, dtype="u2"),
                     x=np.arange(10, dtype="u2"), y=np.arange(10, dtype="u2"), p=np.arange(10, dtype="u2"))
        data = bytearray(path.read_bytes())
        at, patched = 0, 0
        while (at := data.find(b"deflate", at + 1)) >= 0:
            assert data[at - 8 : at - 6] == b"\x01\x00"
            data[at - 8 : at - 6] = (32_750).to_bytes(2, "little")
            data[at - 4 : at - 2] = b"\x00\x00"
            patched += 1
        assert patched == 4
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="32750"):
            hdf5.open(path, group="events")

    def test_missing_optional_filter_is_refused_on_read(self, tmp_path: Path) -> None:
        # hdf5plugin stores Blosc as optional, as DSEC's files do. A child process with
        # hdf5plugin stubbed out has no Blosc: open succeeds, reading a chunk fails.
        import hdf5plugin

        data = np.arange(100_000, dtype="u2") % 7
        path = write(tmp_path / "f.h5", compression={**dict(hdf5plugin.Blosc()), "chunks": (50_000,)},
                     t=data, x=data, y=data, p=data)
        code = f"""
import sys, types
sys.modules["hdf5plugin"] = types.ModuleType("hdf5plugin")
from frames2py.adapters import hdf5
with hdf5.open({str(path)!r}, group="events") as reader:
    try:
        list(reader)
    except ValueError as exc:
        assert isinstance(exc.__cause__, OSError), exc.__cause__
        print("refused on read")
"""
        env = {k: v for k, v in os.environ.items() if k != "HDF5_PLUGIN_PATH"}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
        assert result.stdout.strip() == "refused on read"
        assert path.stat().st_size < 4 * data.nbytes // 10  # the chunks really are compressed
