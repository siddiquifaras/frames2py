"""``frames2py.adapters.aedat4.open``: the committed fixture, stream choice, timestamps, geometry, errors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from frames2py import EVENT_DTYPE
from tests.adapters import aedat4_files
from tests.adapters.backends import require_backend
from tests.adapters.test_evt_open import OPENEB_SPARKLERS_FIRST_100K_SHA256

DATA = Path(__file__).resolve().parent.parent / "data"
FIXTURE = DATA / "sparklers_100k.aedat4"


@pytest.fixture(autouse=True)
def _dv() -> None:
    require_backend("dv_processing")


def read_all(path: Path, **kwargs: object) -> tuple[np.ndarray, tuple[int, int] | None, list[np.ndarray]]:
    from frames2py.adapters import aedat4

    with aedat4.open(path, **kwargs) as reader:  # type: ignore[arg-type]
        batches = list(reader)
        size = reader.sensor_size
    events = np.concatenate(batches) if batches else np.empty(0, dtype=EVENT_DTYPE)
    return events, size, batches


class TestFixture:
    def test_events_are_the_source_slice(self) -> None:
        # The fixture holds the first 100,000 events of CC0 sparklers.raw, written by dv-processing;
        # OpenEB's decode of the same events is the expectation.
        events, size, batches = read_all(FIXTURE)
        assert size == (640, 480)
        assert len(events) == 100_000
        assert hashlib.sha256(events.tobytes()).hexdigest() == OPENEB_SPARKLERS_FIRST_100K_SHA256
        assert (int(events["t"].min()), int(events["t"].max())) == (913_716_224, 913_728_417)
        assert np.bincount(events["p"]).tolist() == [68_073, 31_927]
        assert len(batches) == 10  # the file's packets

    def test_same_events_as_the_evt2_excerpt(self) -> None:
        from frames2py.adapters import evt

        with evt.open(DATA / "sparklers_100k.evt2.raw", sensor_size=(640, 480)) as reader:
            source = np.concatenate(list(reader))
        assert read_all(FIXTURE)[0].tobytes() == source.tobytes()

    @pytest.mark.parametrize("batch_size", [1, 999, 10_000, 100_000, 250_000])
    def test_batch_size_changes_only_the_boundaries(self, batch_size: int) -> None:
        events, _, batches = read_all(FIXTURE, batch_size=batch_size)
        assert events.tobytes() == read_all(FIXTURE)[0].tobytes()
        assert {len(b) for b in batches[:-1]} <= {batch_size}

    def test_explicit_size_must_match(self) -> None:
        from frames2py.adapters import aedat4

        assert read_all(FIXTURE, sensor_size=(640, 480))[1] == (640, 480)
        with pytest.raises(ValueError, match="conflicts"):
            aedat4.open(FIXTURE, sensor_size=(346, 260))


class TestTimestamps:
    def test_backward_steps_are_kept_in_file_order(self, tmp_path: Path) -> None:
        path = aedat4_files.write_with_timestamps(tmp_path / "back.aedat4", [[2_000_000, 2_000_500, 2_000_100], [1_000_000]])
        assert read_all(path)[0]["t"].tolist() == [2_000_000, 2_000_500, 2_000_100, 1_000_000]

    def test_negative_timestamp_raises_and_is_never_clamped(self, tmp_path: Path) -> None:
        from frames2py.adapters import aedat4

        path = aedat4_files.write_with_timestamps(tmp_path / "neg.aedat4", [[10, 20], [-5, 30]])
        with aedat4.open(path) as reader:
            it = iter(reader)
            assert next(it)["t"].tolist() == [10, 20]
            with pytest.raises(ValueError, match="negative timestamp"):
                next(it)

    def test_large_timestamps_are_exact(self, tmp_path: Path) -> None:
        big = (1 << 62) + 12345
        path = aedat4_files.write_with_timestamps(tmp_path / "big.aedat4", [[big, big + 1]])
        assert read_all(path)[0]["t"].tolist() == [big, big + 1]


class TestStreams:
    def test_polarity_and_coordinates(self, tmp_path: Path) -> None:
        path = aedat4_files.write(tmp_path / "e.aedat4", {"events": [[(5, 0, 0, 0), (6, 639, 479, 1), (7, 3, 400, 1)]]})
        events, size, _ = read_all(path)
        assert events.tolist() == [(5, 0, 0, 0), (6, 639, 479, 1), (7, 3, 400, 1)]
        assert size == (640, 480)

    def test_the_stream_named_events_is_read(self, tmp_path: Path) -> None:
        path = aedat4_files.write(tmp_path / "two.aedat4", {"events": [[(1, 1, 1, 1)]], "extra": [[(2, 2, 2, 0)]]})
        assert read_all(path)[0].tolist() == [(1, 1, 1, 1)]

    def test_a_single_event_stream_under_another_name_is_read(self, tmp_path: Path) -> None:
        path = aedat4_files.write(tmp_path / "left.aedat4", {"left": [[(1, 1, 1, 1)]]}, resolution=(346, 260))
        events, size, _ = read_all(path)
        assert events.tolist() == [(1, 1, 1, 1)] and size == (346, 260)

    def test_several_unnamed_event_streams_are_ambiguous(self, tmp_path: Path) -> None:
        from frames2py.adapters import aedat4

        path = aedat4_files.write(tmp_path / "lr.aedat4", {"left": [[(1, 1, 1, 1)]], "right": [[(2, 2, 2, 0)]]})
        with pytest.raises(ValueError, match="several event streams"):
            aedat4.open(path)

    def test_no_event_stream(self, tmp_path: Path) -> None:
        import dv_processing as dv

        from frames2py.adapters import aedat4

        path = tmp_path / "frames.aedat4"
        writer = dv.io.MonoCameraWriter(str(path), dv.io.MonoCameraWriter.FrameOnlyConfig("cam", (640, 480)))
        del writer
        with pytest.raises(ValueError, match="no event stream"):
            aedat4.open(path)

    def test_empty_event_stream(self, tmp_path: Path) -> None:
        path = aedat4_files.write(tmp_path / "empty.aedat4", {"events": []})
        events, size, batches = read_all(path)
        assert batches == [] and size == (640, 480)

    def test_missing_resolution_needs_sensor_size(self, tmp_path: Path) -> None:
        from frames2py.adapters import aedat4

        path = aedat4_files.write(tmp_path / "e.aedat4", {"events": [[(1, 1, 1, 1)]]}, compressed=False)
        aedat4_files.without_resolution(path)
        with pytest.raises(ValueError, match="sensor_size"):
            aedat4.open(path)
        events, size, _ = read_all(path, sensor_size=(64, 48))
        assert size == (64, 48) and events.tolist() == [(1, 1, 1, 1)]


class TestMalformed:
    @pytest.mark.parametrize(
        "content",
        [b"", b"#!AER-DAT4.0\r\n", b"%evt 3.0\n" + bytes(range(256)) * 4, FIXTURE.read_bytes()[:200],
         FIXTURE.read_bytes()[: FIXTURE.stat().st_size // 2], FIXTURE.read_bytes()[:-5]],
        ids=["empty", "version-line-only", "not-aedat4", "truncated-header", "truncated-half", "truncated-end"],
    )
    def test_unreadable_content_is_a_value_error_chained_from_dv(self, tmp_path: Path, content: bytes) -> None:
        from frames2py.adapters import aedat4

        path = tmp_path / "bad.aedat4"
        path.write_bytes(content)
        with pytest.raises(ValueError) as info:
            aedat4.open(path)
        assert isinstance(info.value.__cause__, RuntimeError)

    def test_negative_coordinate_is_rejected(self, tmp_path: Path) -> None:
        from frames2py.adapters import aedat4

        placeholder = aedat4_files.PLACEHOLDER
        path = aedat4_files.write(tmp_path / "x.aedat4", {"events": [[(placeholder, 0, 7, 1)]]}, compressed=False)
        data = path.read_bytes()
        record = np.int64(placeholder).tobytes() + np.int16(0).tobytes()
        assert data.count(record) == 1
        data = data.replace(record, np.int64(10).tobytes() + np.int16(-1).tobytes())
        path.write_bytes(data.replace(np.int64(placeholder).tobytes(), np.int64(10).tobytes()))
        with aedat4.open(path) as reader, pytest.raises(ValueError, match="negative coordinate"):
            list(reader)
