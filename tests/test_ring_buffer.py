"""Tests for ChunkedRingBuffer: roundtrip, overflow, accounting, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from frames2py.core.transport.ring_buffer import ChunkedRingBuffer
from frames2py.core.types import EVENT_DTYPE, OverflowPolicy


def _make_events(n: int, t_start: int = 0) -> np.ndarray:
    """Create n deterministic events for testing."""
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(t_start, t_start + n, dtype=np.uint64)
    events["x"] = np.arange(n, dtype=np.uint16) % 64
    events["y"] = np.arange(n, dtype=np.uint16) % 48
    events["p"] = np.arange(n, dtype=np.uint8) % 2
    return events


class TestWriteReadRoundtrip:
    """Write events, read them back, verify identity."""

    def test_single_chunk_roundtrip(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        events = _make_events(50)
        dropped = rb.write(events)
        assert dropped == 0
        result = rb.read_new()
        assert result is not None
        assert len(result) == 50
        np.testing.assert_array_equal(result["t"], events["t"])
        np.testing.assert_array_equal(result["x"], events["x"])

    def test_multiple_writes_multiple_reads(self):
        rb = ChunkedRingBuffer(capacity=8, chunk_size=100)
        for i in range(4):
            events = _make_events(80, t_start=i * 80)
            rb.write(events)

        all_t = []
        for _ in range(4):
            chunk = rb.read_new()
            assert chunk is not None
            all_t.extend(chunk["t"].tolist())

        assert len(all_t) == 320
        assert all_t == list(range(320))

    def test_drain_concatenates_all_chunks(self):
        rb = ChunkedRingBuffer(capacity=8, chunk_size=100)
        for i in range(3):
            rb.write(_make_events(50, t_start=i * 50))
        result = rb.drain()
        assert result is not None
        assert len(result) == 150
        np.testing.assert_array_equal(result["t"], np.arange(150, dtype=np.uint64))

    def test_empty_read_returns_none(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        assert rb.read_new() is None

    def test_empty_drain_returns_none(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        assert rb.drain() is None

    def test_empty_write_drops_nothing(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        dropped = rb.write(np.empty(0, dtype=EVENT_DTYPE))
        assert dropped == 0
        assert rb.size == 0


class TestLargeBatchSplit:
    """Batches larger than chunk_size are split across multiple slots."""

    def test_split_across_two_chunks(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        events = _make_events(180)
        dropped = rb.write(events)
        assert dropped == 0
        assert rb.size == 2

        chunk1 = rb.read_new()
        chunk2 = rb.read_new()
        assert chunk1 is not None
        assert chunk2 is not None
        assert len(chunk1) == 100
        assert len(chunk2) == 80
        combined = np.concatenate([chunk1, chunk2])
        np.testing.assert_array_equal(combined["t"], events["t"])

    def test_exact_multiple_chunks(self):
        rb = ChunkedRingBuffer(capacity=8, chunk_size=100)
        events = _make_events(300)
        dropped = rb.write(events)
        assert dropped == 0
        assert rb.size == 3


class TestOverflowDropOldest:
    """DROP_OLDEST policy evicts the oldest chunk when full."""

    def test_overflow_drops_oldest(self):
        rb = ChunkedRingBuffer(
            capacity=3, chunk_size=10,
            overflow_policy=OverflowPolicy.DROP_OLDEST,
        )
        for i in range(5):
            rb.write(_make_events(10, t_start=i * 10))

        # Buffer holds 3 chunks; first 2 were evicted.
        assert rb.chunks_dropped == 2
        assert rb.total_dropped == 20

        # Remaining chunks: batches 2, 3, 4.
        result = rb.drain()
        assert result is not None
        assert len(result) == 30
        assert result["t"][0] == 20
        assert result["t"][-1] == 49

    def test_exact_drop_accounting_with_partial_chunks(self):
        rb = ChunkedRingBuffer(capacity=2, chunk_size=100)
        # Write 60 events (partial chunk, slot 0)
        rb.write(_make_events(60, t_start=0))
        # Write 80 events (partial chunk, slot 1)
        rb.write(_make_events(80, t_start=60))
        # Write 50 events → must evict slot 0 (60 events)
        rb.write(_make_events(50, t_start=140))

        assert rb.chunks_dropped == 1
        assert rb.total_dropped == 60  # exact count, not chunk_size

    def test_fill_ratio(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        assert rb.fill_ratio == 0.0
        rb.write(_make_events(50))
        assert rb.fill_ratio == 0.25
        rb.write(_make_events(50))
        assert rb.fill_ratio == 0.5
        rb.write(_make_events(50))
        rb.write(_make_events(50))
        assert rb.fill_ratio == 1.0

    def test_fill_ratio_never_exceeds_one(self):
        rb = ChunkedRingBuffer(capacity=2, chunk_size=10)
        for i in range(100):
            rb.write(_make_events(10, t_start=i * 10))
        assert rb.fill_ratio <= 1.0


class TestOverflowDropNewest:
    """DROP_NEWEST policy discards incoming writes when buffer is full."""

    def test_drop_newest_discards_incoming(self):
        rb = ChunkedRingBuffer(
            capacity=2, chunk_size=10,
            overflow_policy=OverflowPolicy.DROP_NEWEST,
        )
        rb.write(_make_events(10, t_start=0))
        rb.write(_make_events(10, t_start=10))
        # Buffer full. This write should be discarded.
        dropped = rb.write(_make_events(10, t_start=20))
        assert dropped == 10
        assert rb.total_dropped == 10

        # Only original two chunks survive.
        result = rb.drain()
        assert result is not None
        assert len(result) == 20
        assert result["t"][0] == 0
        assert result["t"][-1] == 19


class TestClear:
    """Verify clear resets all state."""

    def test_clear_empties_buffer(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=100)
        rb.write(_make_events(200))
        assert rb.size > 0
        rb.clear()
        assert rb.size == 0
        assert rb.fill_ratio == 0.0
        assert rb.total_dropped == 0
        assert rb.chunks_dropped == 0
        assert rb.read_new() is None


class TestEdgeCases:
    """Edge cases and invalid inputs."""

    def test_capacity_one(self):
        rb = ChunkedRingBuffer(capacity=1, chunk_size=10)
        rb.write(_make_events(10, t_start=0))
        assert rb.size == 1
        rb.write(_make_events(10, t_start=10))
        assert rb.chunks_dropped == 1
        result = rb.read_new()
        assert result is not None
        assert result["t"][0] == 10

    def test_chunk_size_one(self):
        rb = ChunkedRingBuffer(capacity=4, chunk_size=1)
        events = _make_events(3)
        rb.write(events)
        assert rb.size == 3

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            ChunkedRingBuffer(capacity=0, chunk_size=10)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            ChunkedRingBuffer(capacity=4, chunk_size=0)

    def test_stress_many_cycles(self):
        """Write-read cycles without leaks or corruption."""
        rb = ChunkedRingBuffer(capacity=8, chunk_size=256)
        total_written = 0
        total_read = 0
        for i in range(1000):
            n = 100 + (i % 200)
            rb.write(_make_events(n, t_start=total_written))
            total_written += n
            chunk = rb.read_new()
            if chunk is not None:
                total_read += len(chunk)
        # Drain remaining
        while True:
            chunk = rb.read_new()
            if chunk is None:
                break
            total_read += len(chunk)
        assert total_read + rb.total_dropped == total_written
