"""Opt-in: the adapters on complete real recordings (``pytest --recordings``).

The recordings are downloaded and hash-checked by ``tests.recordings``; a missing one fails
the test. Expected values come from decoders other than Frames2Py's: OpenEB 5.2.0's default
RAW path for EVT (as ``EVENT_DTYPE`` bytes).
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
