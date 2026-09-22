"""Tests for core data types, metadata containers, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from frames2py.core.types import (
    EVENT_DTYPE,
    BatchMeta,
    EngineStats,
    OverflowPolicy,
    SnapshotMeta,
    validate_event_batch,
)


class TestEventDtype:
    """Verify the canonical event dtype layout."""

    def test_field_names(self):
        assert EVENT_DTYPE.names == ("t", "x", "y", "p")

    def test_field_types(self):
        assert EVENT_DTYPE["t"] == np.dtype("<u8")
        assert EVENT_DTYPE["x"] == np.dtype("<u2")
        assert EVENT_DTYPE["y"] == np.dtype("<u2")
        assert EVENT_DTYPE["p"] == np.dtype("u1")

    def test_itemsize_is_13(self):
        assert EVENT_DTYPE.itemsize == 13

    def test_create_empty_array(self):
        arr = np.empty(0, dtype=EVENT_DTYPE)
        assert arr.shape == (0,)
        assert arr.dtype == EVENT_DTYPE

    def test_create_and_read_back(self):
        arr = np.empty(3, dtype=EVENT_DTYPE)
        arr[0] = (1000, 100, 50, 1)
        arr[1] = (2000, 200, 100, 0)
        arr[2] = (3000, 300, 150, 1)
        assert arr["t"][0] == 1000
        assert arr["x"][1] == 200
        assert arr["y"][2] == 150
        assert arr["p"][0] == 1


class TestValidateEventBatch:
    """Verify validate_event_batch accepts valid and rejects invalid input."""

    def test_valid_batch(self, small_events):
        result = validate_event_batch(small_events)
        assert result is small_events

    def test_empty_batch_is_valid(self):
        empty = np.empty(0, dtype=EVENT_DTYPE)
        result = validate_event_batch(empty)
        assert len(result) == 0

    def test_rejects_non_numpy(self):
        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validate_event_batch([1, 2, 3])  # type: ignore

    def test_rejects_unstructured_array(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(TypeError, match="structured array"):
            validate_event_batch(arr)

    def test_rejects_missing_fields(self):
        dt = np.dtype([("t", "<u8"), ("x", "<u2")])
        arr = np.empty(5, dtype=dt)
        with pytest.raises(ValueError, match="missing required fields"):
            validate_event_batch(arr)

    def test_rejects_2d_array(self):
        arr = np.empty((5, 2), dtype=EVENT_DTYPE)
        with pytest.raises(ValueError, match="must be 1-D"):
            validate_event_batch(arr)

    def test_rejects_non_contiguous(self):
        arr = np.empty(10, dtype=EVENT_DTYPE)[::2]
        assert not arr.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="C-contiguous"):
            validate_event_batch(arr)

    def test_accepts_superset_fields(self):
        dt = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("extra", "<f4")])
        arr = np.empty(5, dtype=dt)
        arr["t"] = 0
        arr["x"] = 0
        arr["y"] = 0
        arr["p"] = 0
        arr["extra"] = 0.0
        result = validate_event_batch(arr)
        assert len(result) == 5


class TestBatchMeta:
    """Verify BatchMeta frozen dataclass."""

    def test_defaults(self):
        m = BatchMeta()
        assert m.monotonic is False
        assert m.reordered is False
        assert m.source == ""
        assert m.sensor_size is None

    def test_is_frozen(self):
        m = BatchMeta(source="test")
        with pytest.raises(AttributeError):
            m.source = "changed"  # type: ignore

    def test_with_values(self):
        m = BatchMeta(monotonic=True, source="prophesee", sensor_size=(1280, 720))
        assert m.monotonic is True
        assert m.sensor_size == (1280, 720)


class TestSnapshotMeta:
    """Verify SnapshotMeta fields and defaults."""

    def test_defaults(self):
        m = SnapshotMeta()
        assert m.timestamp == 0
        assert m.seq == 0
        assert m.events_accumulated == 0
        assert m.events_dropped == 0
        assert m.wall_time_ns > 0

    def test_custom_values(self):
        m = SnapshotMeta(timestamp=5000, seq=10, events_accumulated=1000)
        assert m.timestamp == 5000
        assert m.seq == 10


class TestEngineStats:
    """Verify EngineStats mutable dataclass."""

    def test_defaults_are_zero(self):
        s = EngineStats()
        assert s.events_ingested == 0
        assert s.events_dropped == 0
        assert s.chunks_dropped == 0
        assert s.buffer_fill_ratio == 0.0
        assert s.snapshots_published == 0
        assert s.uptime_ns == 0

    def test_is_mutable(self):
        s = EngineStats()
        s.events_ingested = 100
        assert s.events_ingested == 100


class TestOverflowPolicy:
    """Verify overflow policy enum."""

    def test_values(self):
        assert OverflowPolicy.DROP_OLDEST.value == "drop_oldest"
        assert OverflowPolicy.DROP_NEWEST.value == "drop_newest"

    def test_no_block(self):
        names = [p.name for p in OverflowPolicy]
        assert "BLOCK" not in names
