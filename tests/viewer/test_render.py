"""``frames2py.viewer.render``: exact images for each kernel's frames, scaling, no mutation, errors.

Headless and without pyglet. Expected images come from a per-pixel oracle in exact rational
arithmetic, with its own percentile, not from the renderer's vectorised code.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from frames2py import Engine, EVENT_DTYPE, ExpDecay, SnapshotMeta, TimestampDecay
from frames2py.publish import Snapshot
from frames2py.viewer import render


def snap(frame: np.ndarray, watermark: int | None = 0, sequence: int = 1) -> Snapshot:
    frame = np.ascontiguousarray(frame)
    frame.flags.writeable = False
    return Snapshot(frame, SnapshotMeta(watermark=watermark, sequence=sequence))


def percentile_99(values: list[Fraction]) -> Fraction:
    """Linear interpolation between closest ranks (NumPy's default method), exactly."""
    ordered = sorted(values)
    rank = Fraction(99, 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def oracle_scale(values: list[Any], scale: float | None) -> Fraction | None:
    if scale is not None:
        return Fraction(scale)
    nonzero = [Fraction(float(v)) for v in values if v != 0]
    if not nonzero:
        return None
    return max(nonzero) if len(nonzero) < 100 else percentile_99(nonzero)


def level(v: Any, s: Fraction | None) -> int:
    if s is None:
        return 0
    value = Fraction(float(v))
    return 255 if value >= s else math.floor(255 * value / s)


def oracle_grey(frame: np.ndarray, scale: float | None = None) -> np.ndarray:
    s = oracle_scale(frame.ravel().tolist(), scale)
    out = np.zeros(frame.shape + (3,), dtype=np.uint8)
    for (row, col), v in np.ndenumerate(frame):
        out[row, col] = level(v, s)
    return out


def oracle_polarity(frame: np.ndarray, scale: float | None = None) -> np.ndarray:
    s = oracle_scale(frame.ravel().tolist(), scale)
    out = np.zeros(frame.shape[:2] + (3,), dtype=np.uint8)
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            off, on = level(frame[row, col, 0], s), level(frame[row, col, 1], s)
            out[row, col] = (on, on, off)
    return out


def oracle_time(frame: np.ndarray, watermark: int | None, window_us: float) -> np.ndarray:
    out = np.zeros(frame.shape + (3,), dtype=np.uint8)
    for (row, col), v in np.ndenumerate(frame):
        if v == 0 or watermark is None:
            continue
        age = max(watermark - int(v), 0)
        out[row, col] = math.floor(255 * max(Fraction(0), 1 - Fraction(age) / Fraction(window_us)))
    return out


class TestCounts:
    def test_all_zero_is_black(self) -> None:
        image = render(snap(np.zeros((3, 4), dtype=np.uint32)))
        assert (image.shape, image.dtype, image.max()) == ((3, 4, 3), np.uint8, 0)

    def test_fewer_than_100_nonzero_values_scale_by_the_maximum(self) -> None:
        frame = np.zeros((10, 10), dtype=np.uint32)
        frame.flat[:99] = np.arange(1, 100)
        frame.flat[0] = 700
        image = render(snap(frame))
        assert image[0, 0].tolist() == [255, 255, 255]
        assert image.flat[3 * 5] == math.floor(255 * 6 / 700)
        assert np.array_equal(image, oracle_grey(frame))

    def test_from_100_nonzero_values_scale_by_the_99th_percentile(self) -> None:
        # 150 values: mostly 7, a few 1 and 3, one outlier of 1000. The percentile falls
        # between two 7s, so the scale is exactly 7 and the outlier doesn't dim the rest.
        frame = np.zeros((16, 16), dtype=np.uint32)
        values = [1] * 20 + [3] * 20 + [7] * 109 + [1000]
        frame.flat[: len(values)] = values
        image = render(snap(frame))
        assert image.flat[0] == math.floor(255 * 1 / 7)
        assert image.flat[3 * 20] == math.floor(255 * 3 / 7)
        assert image.flat[3 * 40] == 255 and image.flat[3 * 149] == 255
        assert np.array_equal(image, oracle_grey(frame))

    def test_exactly_100_nonzero_values_use_the_percentile(self) -> None:
        frame = np.zeros((10, 10), dtype=np.uint32)
        frame.flat[:] = [5] * 99 + [900]
        image = render(snap(frame))
        # The 99th percentile of 100 values lies at rank 98.01: 5 + 0.01 * (900 - 5) = 13.95.
        # With the maximum (900) as the scale, a 5 would be level 1.
        assert image.flat[0] == math.floor(255 * 5 / Fraction("13.95"))
        assert np.array_equal(image, oracle_grey(frame))

    def test_explicit_scale(self) -> None:
        frame = np.array([[0, 1, 2], [4, 8, 2**32 - 1]], dtype=np.uint32)
        image = render(snap(frame), scale=4)
        assert image[..., 0].tolist() == [[0, 63, 127], [255, 255, 255]]
        assert np.array_equal(image, oracle_grey(frame, 4))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_frames_match_the_oracle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        frame = rng.integers(0, 40, (24, 32)).astype(np.uint32) * (rng.random((24, 32)) < 0.4)
        frame = frame.astype(np.uint32)
        assert np.array_equal(render(snap(frame)), oracle_grey(frame))


class TestDecays:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_float32_frames_match_the_oracle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        frame = (rng.random((24, 32)) * 3).astype(np.float32) * (rng.random((24, 32)) < 0.5)
        frame = frame.astype(np.float32)
        assert np.array_equal(render(snap(frame)), oracle_grey(frame))
        assert np.array_equal(render(snap(frame), scale=0.5), oracle_grey(frame, 0.5))

    @pytest.mark.parametrize("kernel", [ExpDecay(0.9), TimestampDecay(1_000.0)])
    def test_real_decay_snapshots(self, kernel: Any) -> None:
        engine = Engine((40, 30), kernel, snapshot_interval_ms=0)
        rng = np.random.default_rng(3)
        events = np.zeros(2_000, dtype=EVENT_DTYPE)
        events["t"] = np.sort(rng.integers(0, 5_000, 2_000))
        events["x"], events["y"] = rng.integers(0, 40, 2_000), rng.integers(0, 30, 2_000)
        engine.ingest(events)
        snapshot = engine.snapshot()
        assert snapshot is not None and snapshot.frame.dtype == np.float32
        assert np.array_equal(render(snapshot), oracle_grey(snapshot.frame))


class TestPolarity:
    def test_off_blue_on_yellow_both_white(self) -> None:
        frame = np.zeros((1, 4, 2), dtype=np.uint32)
        frame[0, 1] = (1, 0)  # OFF only
        frame[0, 2] = (0, 1)  # ON only
        frame[0, 3] = (1, 1)  # both
        assert render(snap(frame)).tolist() == [[[0, 0, 0], [0, 0, 255], [255, 255, 0], [255, 255, 255]]]

    def test_one_scale_for_both_channels(self) -> None:
        frame = np.zeros((1, 2, 2), dtype=np.uint32)
        frame[0, 0] = (4, 0)
        frame[0, 1] = (0, 2)
        assert render(snap(frame)).tolist() == [[[0, 0, 255], [127, 127, 0]]]

    def test_an_engine_polarity_snapshot(self) -> None:
        engine = Engine((3, 1), "polarity", snapshot_interval_ms=0)
        events = np.array([(1, 0, 0, 0), (2, 1, 0, 1), (3, 2, 0, 0), (4, 2, 0, 200)], dtype=EVENT_DTYPE)
        engine.ingest(events)
        snapshot = engine.snapshot()
        assert snapshot is not None
        assert render(snapshot).tolist() == [[[0, 0, 255], [255, 255, 0], [255, 255, 255]]]

    @pytest.mark.parametrize("seed", [0, 1])
    def test_random_frames_match_the_oracle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        frame = (rng.integers(0, 30, (12, 16, 2)) * (rng.random((12, 16, 2)) < 0.5)).astype(np.uint32)
        assert np.array_equal(render(snap(frame)), oracle_polarity(frame))
        assert np.array_equal(render(snap(frame), scale=10), oracle_polarity(frame, 10))


class TestTimeSurface:
    def test_relative_to_the_watermark(self) -> None:
        t = 10_000_000
        frame = np.array([[0, t, t - 1], [t - 25_000, t - 50_000, t - 50_001]], dtype=np.uint64)
        image = render(snap(frame, watermark=t))
        assert image[..., 0].tolist() == [[0, 255, math.floor(255 * (1 - 1 / 50_000))], [127, 0, 0]]
        assert np.array_equal(image, oracle_time(frame, t, 50_000))

    def test_window(self) -> None:
        frame = np.array([[1_000, 900, 1]], dtype=np.uint64)
        assert render(snap(frame, watermark=1_000), window_us=200)[..., 0].tolist() == [[255, 127, 0]]

    def test_zero_is_black_even_within_the_window(self) -> None:
        frame = np.array([[0, 1]], dtype=np.uint64)
        assert render(snap(frame, watermark=10), window_us=1_000)[..., 0].tolist() == [[0, 252]]

    def test_no_watermark_is_black(self) -> None:
        assert render(snap(np.zeros((2, 2), dtype=np.uint64), watermark=None)).max() == 0

    @pytest.mark.parametrize("window_us", [1, 777.5, 50_000])
    def test_random_frames_match_the_oracle(self, window_us: float) -> None:
        rng = np.random.default_rng(4)
        t = 2**40
        frame = (t - rng.integers(0, 80_000, (16, 16))).astype(np.uint64) * (rng.random((16, 16)) < 0.7)
        frame = frame.astype(np.uint64)
        assert np.array_equal(render(snap(frame, watermark=t), window_us=window_us), oracle_time(frame, t, window_us))

    def test_an_engine_time_surface_snapshot(self) -> None:
        engine = Engine((3, 1), "time_surface", snapshot_interval_ms=0)
        engine.ingest(np.array([(100_000, 0, 0, 1), (75_000, 1, 0, 1)], dtype=EVENT_DTYPE))
        snapshot = engine.snapshot()
        assert snapshot is not None
        assert render(snapshot)[..., 0].tolist() == [[255, 127, 0]]


class TestContract:
    @pytest.mark.parametrize(
        "frame",
        [
            np.zeros((4, 4), dtype=np.uint32),
            np.ones((4, 4), dtype=np.float32),
            np.full((4, 4), 7, dtype=np.uint64),
            np.ones((4, 4, 2), dtype=np.uint32),
        ],
    )
    def test_the_snapshot_is_not_modified_and_the_image_is_new(self, frame: np.ndarray) -> None:
        snapshot = snap(frame, watermark=7)
        before = snapshot.frame.copy()
        image = render(snapshot)
        assert np.array_equal(snapshot.frame, before) and not snapshot.frame.flags.writeable
        assert snapshot.meta == SnapshotMeta(watermark=7, sequence=1)
        assert image.flags.c_contiguous and image.flags.writeable and not np.shares_memory(image, snapshot.frame)

    @pytest.mark.parametrize(
        "frame",
        [
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.int32),
            np.zeros((4, 4), dtype=np.uint16),
            np.zeros((4, 4, 3), dtype=np.uint32),
            np.zeros((4, 4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.uint32),
            np.zeros((2, 4, 4), dtype=np.uint64),
        ],
    )
    def test_other_frames_are_refused(self, frame: np.ndarray) -> None:
        with pytest.raises(TypeError, match="no rendering"):
            render(snap(frame))

    @pytest.mark.parametrize("value", [None, np.zeros((2, 2), dtype=np.uint32), (np.zeros((2, 2)), None)])
    def test_not_a_snapshot(self, value: Any) -> None:
        with pytest.raises(TypeError):
            render(value)

    @pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, True, "4"])
    def test_scale_and_window_must_be_finite_and_positive(self, bad: Any) -> None:
        snapshot = snap(np.zeros((2, 2), dtype=np.uint64))
        with pytest.raises(ValueError, match="scale"):
            render(snapshot, scale=bad)
        with pytest.raises(ValueError, match="window_us"):
            render(snapshot, window_us=bad)
