"""``frames2py.adapters.evt.open``: header, geometry, versions, errors and the committed real fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from frames2py import EVENT_DTYPE
from frames2py.adapters import evt
from tests.adapters import evt_words as w

DATA = Path(__file__).resolve().parent.parent / "data"
EVT2_FIXTURE = DATA / "sparklers_100k.evt2.raw"
EVT3_FIXTURE = DATA / "active_marker_head.evt3.raw"

# Independent expectations: OpenEB 5.2.0's default RAW-path decode of the source recordings
# (Metavision::Camera::from_file, time shifting off), as EVENT_DTYPE bytes.
OPENEB_SPARKLERS_FIRST_100K_SHA256 = "88b70b27df2385c85d4c95b76de08b0bd8606d2f26ce1e74171d852604f05590"
OPENEB_ACTIVE_MARKER_FIRST_46893_SHA256 = "1a747e1ee627a19f89dcce4a23f2d662238512c2f9bc6618fba4d8ec08812cb6"

EVT3_BASIC = [w.evt3_time_high(1), w.evt3_y(7), w.evt3_x(3, on=True), w.evt3_x(4, on=False)]


def read_all(path: Path, **kwargs: object) -> tuple[np.ndarray, tuple[int, int] | None, list[np.ndarray]]:
    with evt.open(path, **kwargs) as reader:  # type: ignore[arg-type]
        batches = list(reader)
        size = reader.sensor_size
    events = np.concatenate(batches) if batches else np.empty(0, dtype=EVENT_DTYPE)
    return events, size, batches


def raw(tmp_path: Path, header: bytes, words: list[int], version: str, name: str = "f.raw") -> Path:
    path = tmp_path / name
    path.write_bytes(header + w.body(words, version))
    return path


class TestHeader:
    @pytest.mark.parametrize("version", ["2.0", "3.0"])
    def test_version_selects_the_decoder(self, tmp_path: Path, version: str) -> None:
        words = [w.evt2_time_high(1), w.evt2_cd(3, 4, on=True, low=2)] if version == "2.0" else EVT3_BASIC
        expected = [(66, 3, 4, 1)] if version == "2.0" else [(4096, 3, 7, 1), (4096, 4, 7, 0)]
        for lines in ([f"% evt {version}"], [f"% format EVT{version[0]}"]):
            path = raw(tmp_path, ("\n".join([*lines, "% geometry 640x480"]) + "\n").encode(), words, version)
            events, size, _ = read_all(path)
            assert events.tolist() == expected
            assert size == (640, 480)

    @pytest.mark.parametrize(
        "lines",
        [["% evt 2.1"], ["% format EVT21;height=480;width=640"], ["% evt 4.0"], ["% format EVT4"], ["% date 2020-01-01"],
         ["% evt 2.0", "% format EVT3"], ["% evt 2.0", "% evt 3.0"]],
        ids=["evt-2.1", "format-EVT21", "evt-4.0", "format-EVT4", "no-version", "evt-and-format-disagree", "two-evt-lines"],
    )
    def test_unsupported_or_unclear_versions_are_rejected(self, tmp_path: Path, lines: list[str]) -> None:
        path = raw(tmp_path, ("\n".join([*lines, "% geometry 640x480"]) + "\n").encode(), EVT3_BASIC, "3.0")
        with pytest.raises(ValueError):
            evt.open(path)

    def test_geometry_from_the_format_line(self, tmp_path: Path) -> None:
        path = raw(tmp_path, b"% evt 3.0\n% format EVT3;height=720;width=1280\n", EVT3_BASIC, "3.0")
        assert read_all(path)[1] == (1280, 720)

    @pytest.mark.parametrize(
        "lines",
        [["% geometry 640x480", "% format EVT3;height=720;width=1280"], ["% geometry 640x"], ["% geometry 0x480"],
         ["% format EVT3;height=720"], ["% geometry 640x480", "% geometry 320x240"]],
        ids=["two-geometries", "unparsable", "empty", "format-height-only", "two-geometry-lines"],
    )
    def test_malformed_geometry_is_rejected(self, tmp_path: Path, lines: list[str]) -> None:
        path = raw(tmp_path, ("\n".join(["% evt 3.0", *lines]) + "\n").encode(), EVT3_BASIC, "3.0")
        with pytest.raises(ValueError):
            evt.open(path)

    def test_header_ends_at_end_line_even_if_the_body_starts_with_percent(self, tmp_path: Path) -> None:
        # 0x2025 is an EVT_ADDR_X word whose first byte is b"%".
        words = [w.evt3_time_high(1), w.evt3_y(7), 0x2025]
        assert w.body(words, "3.0")[4:5] == b"%"
        path = raw(tmp_path, b"% evt 3.0\n% geometry 640x480\n% end\n", words, "3.0")
        assert read_all(path)[0].tolist() == [(4096, 0x25, 7, 0)]

    def test_empty_body(self, tmp_path: Path) -> None:
        path = raw(tmp_path, w.header("3.0"), [], "3.0")
        events, size, batches = read_all(path)
        assert batches == [] and size == (640, 480)

    def test_empty_file_has_no_version(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.raw"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="no EVT version"):
            evt.open(path)

    def test_headerless_file_is_rejected(self, tmp_path: Path) -> None:
        path = raw(tmp_path, b"", EVT3_BASIC, "3.0")
        with pytest.raises(ValueError):
            evt.open(path)


class TestGeometry:
    def test_missing_geometry_needs_sensor_size(self, tmp_path: Path) -> None:
        path = raw(tmp_path, w.header("3.0", geometry=None), EVT3_BASIC, "3.0")
        with pytest.raises(ValueError, match="sensor_size"):
            evt.open(path)
        assert read_all(path, sensor_size=(346, 260))[1] == (346, 260)

    def test_explicit_size_must_match_the_header(self, tmp_path: Path) -> None:
        path = raw(tmp_path, w.header("3.0", geometry=(640, 480)), EVT3_BASIC, "3.0")
        assert read_all(path, sensor_size=(640, 480))[1] == (640, 480)
        with pytest.raises(ValueError, match="conflicts"):
            evt.open(path, sensor_size=(480, 640))

    def test_rows_beyond_the_height_are_yielded(self, tmp_path: Path) -> None:
        words = [w.evt3_time_high(1), w.evt3_y(500), w.evt3_x(3, on=True), w.evt3_y(7), w.evt3_x(4, on=True)]
        path = raw(tmp_path, w.header("3.0", geometry=(640, 480)), words, "3.0")
        assert read_all(path)[0].tolist() == [(4096, 3, 500, 1), (4096, 4, 7, 1)]


class TestRealFixtures:
    def test_evt2_excerpt_matches_openeb(self) -> None:
        # The source header has no geometry; the recording is from a 640x480 Gen3.0 sensor.
        with pytest.raises(ValueError, match="sensor_size"):
            evt.open(EVT2_FIXTURE)
        events, size, _ = read_all(EVT2_FIXTURE, sensor_size=(640, 480))
        assert size == (640, 480)
        assert len(events) == 100_000
        assert hashlib.sha256(events.tobytes()).hexdigest() == OPENEB_SPARKLERS_FIRST_100K_SHA256
        assert (int(events["t"].min()), int(events["t"].max())) == (913_716_224, 913_728_417)
        assert int(events["p"].sum()) == 31_927 and set(np.unique(events["p"]).tolist()) == {0, 1}
        assert int(events["x"].max()) == 639 and int(events["y"].max()) == 479

    def test_evt3_excerpt_matches_openeb(self) -> None:
        events, size, _ = read_all(EVT3_FIXTURE)
        assert size == (1280, 720)
        assert len(events) == 46_893
        assert hashlib.sha256(events.tobytes()).hexdigest() == OPENEB_ACTIVE_MARKER_FIRST_46893_SHA256
        assert (int(events["t"].min()), int(events["t"].max())) == (1148, 140_995)
        assert int(events["x"].max()) < 1280 and int(events["y"].max()) < 720

    @pytest.mark.parametrize("fixture", [EVT2_FIXTURE, EVT3_FIXTURE], ids=["evt2", "evt3"])
    def test_real_stream_is_read_size_invariant(self, fixture: Path) -> None:
        from tests.adapters.test_evt_decoding import feed
        from tests.data.derive import split_header

        data = fixture.read_bytes()
        body = data[split_header(data) :]
        version = "2.0" if fixture is EVT2_FIXTURE else "3.0"
        whole = feed(body, version)
        rng = np.random.default_rng(3)
        for sizes in ([7] * (len(body) // 7 + 1), list(rng.integers(1, 5000, size=2000)), [65_537, 3, 131_071]):
            assert feed(body, version, sizes).tobytes() == whole.tobytes()
        assert whole.tobytes() == read_all(fixture, sensor_size=(640, 480) if version == "2.0" else None)[0].tobytes()

    @pytest.mark.parametrize("fixture", [EVT2_FIXTURE, EVT3_FIXTURE], ids=["evt2", "evt3"])
    def test_truncated_file_yields_a_prefix(self, fixture: Path, tmp_path: Path) -> None:
        kwargs = {"sensor_size": (640, 480)} if fixture is EVT2_FIXTURE else {}
        full = read_all(fixture, **kwargs)[0]
        data = fixture.read_bytes()
        for cut in (len(data) - 1, len(data) - 3, len(data) // 2 + 1, len(data) // 3):
            path = tmp_path / f"cut{cut}.raw"
            path.write_bytes(data[:cut])
            part = read_all(path, **kwargs)[0]
            assert 0 < len(part) <= len(full)
            assert part.tobytes() == full[: len(part)].tobytes()
