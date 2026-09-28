"""Opt-in: the adapters on complete real recordings (``pytest --recordings``).

The recordings are downloaded and hash-checked by ``tests.recordings``; a missing one fails
the test. Expected values come from decoders other than Frames2Py's: OpenEB 5.2.0's default
RAW path for EVT and faery 0.7 for AEDAT4, as ``EVENT_DTYPE`` bytes; for DSEC's HDF5, the
dataset's own ``ms_to_idx`` index and the event totals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from frames2py import Accumulator
from tests import recordings

pytestmark = pytest.mark.recordings


def local(name: str) -> Path:
    try:
        return recordings.path(name)
    except (FileNotFoundError, ValueError) as exc:
        pytest.fail(str(exc))


def digest_and_stats(batches: Iterator[np.ndarray]) -> dict[str, object]:
    digest, n, t_min, t_max, sizes = hashlib.sha256(), 0, None, None, set()
    for batch in batches:
        digest.update(batch.tobytes())
        n += len(batch)
        sizes.add(len(batch))
        lo, hi = int(batch["t"].min()), int(batch["t"].max())
        t_min = lo if t_min is None else min(t_min, lo)
        t_max = hi if t_max is None else max(t_max, hi)
    return {"sha256": digest.hexdigest(), "count": n, "t": (t_min, t_max), "sizes": sizes}


# name: (sensor_size to pass, header geometry, count, OpenEB SHA-256, t range)
EVT = {
    "sparklers.raw": ((640, 480), None, 521_252,
                      "58773c06cd3098d13e0f2781e1134449d2f01ed498d4975aabc2c4e5ec81d44b", (913_716_224, 913_812_095)),
    "200_jets_at_200hz.raw": ((640, 480), None, 407_365,
                              "cf1a9025f62e8e3ecef25182304c1cf3122cef7178e2155a76813f0163bc8910", (212_753_170, 214_557_096)),
    "faery_evt3.raw": (None, (1280, 720), 1_218_618,
                       "b546bc71c0f58734231038eb0504e3a4fa3b41b8404b4f8943a36fb6a62023fa", (11_200_224, 21_968_221)),
    "active_marker.raw": (None, (1280, 720), 22_316_758,
                          "52420f6154adbffd7165b648ec16eb5b142ad3720aa6b5055a2e367d4ef5aa7a", (1148, 31_611_953)),
}


@pytest.mark.parametrize("name", sorted(EVT))
class TestEvtRecordings:
    def test_decode_matches_openeb(self, name: str) -> None:
        from frames2py.adapters import evt

        size, geometry, count, sha, t_range = EVT[name]
        path = local(name)
        if geometry is None:
            with pytest.raises(ValueError, match="sensor_size"):
                evt.open(path)
        with evt.open(path, sensor_size=size) as reader:
            assert reader.sensor_size == (size or geometry)
            stats = digest_and_stats(iter(reader))
        assert (stats["count"], stats["sha256"], stats["t"]) == (count, sha, t_range)

    @pytest.mark.parametrize("batch_size", [1_000_003, 4096])
    def test_batch_size_changes_only_the_boundaries(self, name: str, batch_size: int) -> None:
        from frames2py.adapters import evt

        size, _, count, sha, _ = EVT[name]
        with evt.open(local(name), sensor_size=size, batch_size=batch_size) as reader:
            stats = digest_and_stats(iter(reader))
        assert (stats["count"], stats["sha256"]) == (count, sha)
        assert max(stats["sizes"]) == batch_size or count < batch_size  # type: ignore[type-var]

    def test_awkward_read_sizes(self, name: str) -> None:
        from frames2py.adapters._evt_decode import Evt2Decoder, Evt3Decoder
        from tests.data.derive import split_header

        _, _, count, sha, _ = EVT[name]
        data = local(name).read_bytes()
        body = memoryview(data)[split_header(data) :]
        decoder = Evt2Decoder() if recordings.RECORDINGS[name].format == "evt2" else Evt3Decoder()
        rng = np.random.default_rng(len(data))
        sizes = rng.integers(1, 3 * 65_536, size=len(body) // 32_768 + 2)
        pieces, at = [], 0
        for s in sizes:
            pieces += decoder.feed(body[at : at + int(s)])
            at += int(s)
        pieces += decoder.feed(body[at:])
        stats = digest_and_stats(iter(pieces))
        assert (stats["count"], stats["sha256"]) == (count, sha)

    def test_counts_and_time_surface_through_the_accumulator(self, name: str) -> None:
        from frames2py.adapters import evt

        size = EVT[name][0]
        with evt.open(local(name), sensor_size=size) as reader:
            width, height = reader.sensor_size  # type: ignore[misc]
            batches = list(reader)
        events = np.concatenate(batches)
        counts, surface = Accumulator((width, height), "event_count"), Accumulator((width, height), "time_surface")
        for batch in batches:
            counts.accumulate(batch)
            surface.accumulate(batch)
        pixel = events["y"].astype(np.intp) * width + events["x"]
        assert counts.events_out_of_bounds == 0
        np.testing.assert_array_equal(counts.read().ravel(), np.bincount(pixel, minlength=width * height))
        order = np.lexsort((events["t"], pixel))
        last = np.flatnonzero(np.diff(pixel[order], append=-1) != 0)
        expected = np.zeros(width * height, dtype=np.uint64)
        expected[pixel[order][last]] = events["t"][order][last]
        np.testing.assert_array_equal(surface.read().ravel(), expected)
        assert counts.watermark == int(events["t"].max())


# name: (geometry, count, faery SHA-256, t range)
AEDAT4 = {
    "dvp_test-minimal.aedat4": ((640, 480), 255_283,
                                "96d4b0d0ac66c379cdf4cb6aa8a97354fa64e8c265c9f308bc6a055f8e5d762d",
                                (1_631_717_221_674_515, 1_631_717_224_374_502)),
    "dvp_sample_data.aedat4": ((346, 260), 9_193,
                               "d238dfa46833b6806083c3459e464d122ebc1142785e94f7d09d76e3e4a44fe6",
                               (1_663_249_605_734_020, 1_663_249_609_547_839)),
    "faery_davis346.aedat4": ((346, 260), 78_830,
                              "2154b09ad3de1af1a897724baee61d587d8868e2aaad49632f6985c8f9a3f6b9",
                              (1_589_163_147_368_868, 1_589_163_149_728_813)),
}


@pytest.mark.parametrize("name", sorted(AEDAT4))
class TestAedat4Recordings:
    @pytest.mark.parametrize("batch_size", [None, 1000, 1_000_000])
    def test_decode_matches_faery(self, name: str, batch_size: int | None) -> None:
        from tests.adapters.backends import require_backend

        require_backend("dv_processing")
        from frames2py.adapters import aedat4

        geometry, count, sha, t_range = AEDAT4[name]
        with aedat4.open(local(name), batch_size=batch_size) as reader:
            assert reader.sensor_size == geometry
            stats = digest_and_stats(iter(reader))
        assert (stats["count"], stats["sha256"], stats["t"]) == (count, sha, t_range)


class TestDsecRecording:
    """DSEC ``thun_01_a`` left events: 131,482,728 events, ``t`` uint32 relative to ``/t_offset``."""

    NAME = "dsec_thun_01_a_events_left.h5"
    OFFSET = 49_739_900_557

    def test_events_follow_the_dataset_index(self) -> None:
        from tests.adapters.backends import require_backend

        require_backend("h5py", "hdf5plugin")
        import h5py

        from frames2py.adapters import hdf5

        path = local(self.NAME)
        with h5py.File(path) as f:
            ms_to_idx = f["ms_to_idx"][:]
            assert int(f["t_offset"][()]) == self.OFFSET
        n, t_min, t_max, polarity = 0, None, None, np.zeros(2, dtype=np.int64)
        x_max = y_max = 0
        with hdf5.open(path, group="events", t_offset="/t_offset", sensor_size=(640, 480)) as reader:
            for batch in reader:
                relative = batch["t"].astype(np.int64) - self.OFFSET
                inside = ms_to_idx[(ms_to_idx >= n) & (ms_to_idx < n + len(batch))]
                for i in inside.tolist():
                    k = int(np.searchsorted(ms_to_idx, i))
                    assert relative[i - n] >= 1000 * k
                    if i > n:
                        assert relative[i - n - 1] < 1000 * k
                t_min = int(batch["t"].min()) if t_min is None else min(t_min, int(batch["t"].min()))
                t_max = int(batch["t"].max()) if t_max is None else max(t_max, int(batch["t"].max()))
                x_max, y_max = max(x_max, int(batch["x"].max())), max(y_max, int(batch["y"].max()))
                polarity += np.bincount(batch["p"], minlength=2)[:2]
                n += len(batch)
        assert n == 131_482_728
        assert (t_min, t_max) == (self.OFFSET, self.OFFSET + 9_800_999)
        assert (x_max, y_max) == (639, 479)
        assert polarity.tolist() == [61_362_404, 70_120_324]
