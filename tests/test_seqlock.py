"""Tests for the Seqlock snapshot bridge: consistency, tearing, monotonicity."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from frames2py.core.transport.seqlock import Seqlock
from frames2py.core.types import SnapshotMeta

FREE_THREADED_WITHOUT_GIL = not getattr(sys, "_is_gil_enabled", lambda: True)()


class TestBasicReadWrite:
    """Single-threaded read/write correctness."""

    def test_initial_read_returns_none(self):
        sl = Seqlock((48, 64), np.dtype(np.float32))
        assert sl.try_read() is None

    def test_write_then_read(self):
        sl = Seqlock((48, 64), np.dtype(np.float32))
        buf = sl.begin_write()
        buf[:] = 42.0
        meta = SnapshotMeta(timestamp=1000, seq=1, events_accumulated=500)
        sl.end_write(meta)

        result = sl.try_read()
        assert result is not None
        frame, rmeta = result
        np.testing.assert_array_equal(frame, 42.0)
        assert rmeta.timestamp == 1000
        assert rmeta.seq == 1

    def test_read_returns_copy(self):
        sl = Seqlock((4, 4), np.dtype(np.float32))
        buf = sl.begin_write()
        buf[:] = 1.0
        sl.end_write(SnapshotMeta(seq=1))

        result = sl.try_read()
        assert result is not None
        frame, _ = result
        # Modifying the returned frame should not affect the seqlock.
        frame[:] = 999.0
        result2 = sl.try_read()
        assert result2 is not None
        np.testing.assert_array_equal(result2[0], 1.0)

    def test_multiple_writes_latest_visible(self):
        sl = Seqlock((4, 4), np.dtype(np.float32))
        for i in range(5):
            buf = sl.begin_write()
            buf[:] = float(i)
            sl.end_write(SnapshotMeta(seq=i + 1, timestamp=i * 100))

        result = sl.try_read()
        assert result is not None
        np.testing.assert_array_equal(result[0], 4.0)
        assert result[1].seq == 5

    def test_seq_increments_correctly(self):
        sl = Seqlock((4, 4), np.dtype(np.float32))
        assert sl.seq == 0

        sl.begin_write()
        assert sl.seq == 1  # odd = writing

        sl.end_write(SnapshotMeta(seq=1))
        assert sl.seq == 2  # even = published

    def test_3d_frame_shape(self):
        sl = Seqlock((48, 64, 2), np.dtype(np.float32))
        buf = sl.begin_write()
        assert buf.shape == (48, 64, 2)
        buf[:, :, 0] = 1.0
        buf[:, :, 1] = 2.0
        sl.end_write(SnapshotMeta(seq=1))

        result = sl.try_read()
        assert result is not None
        assert result[0].shape == (48, 64, 2)
        np.testing.assert_array_equal(result[0][:, :, 0], 1.0)


class TestReset:
    """Verify reset clears state."""

    def test_reset_clears_buffers_and_seq(self):
        sl = Seqlock((4, 4), np.dtype(np.float32))
        buf = sl.begin_write()
        buf[:] = 99.0
        sl.end_write(SnapshotMeta(seq=1))
        assert sl.seq == 2

        sl.reset()
        assert sl.seq == 0
        assert sl.try_read() is None


class TestConcurrentTearing:
    """Multi-threaded tearing detection.

    One writer thread fills frames with a known pattern (all pixels = seq).
    Multiple reader threads verify that every pixel in a successfully read
    frame matches the metadata seq.  Any mismatch = tearing = test failure.
    """

    @pytest.mark.timeout(30)
    def test_no_tearing_4_readers(self):
        width, height = 16, 12
        sl = Seqlock((height, width), np.dtype(np.float32))
        n_writes = 50_000
        n_readers = 4
        tearing_detected = threading.Event()
        writer_done = threading.Event()
        readers_ready = threading.Barrier(n_readers + 1, timeout=5)
        reader_counts = [0] * n_readers

        def writer():
            readers_ready.wait()
            for seq in range(1, n_writes + 1):
                buf = sl.begin_write()
                buf[:] = float(seq)
                sl.end_write(SnapshotMeta(seq=seq, timestamp=seq * 10))
                if seq % 2000 == 0:
                    time.sleep(0)  # yield GIL so readers can execute
            writer_done.set()

        def reader(idx: int):
            readers_ready.wait()
            while True:
                result = sl.try_read()
                if result is None:
                    if writer_done.is_set():
                        break
                    continue
                frame, meta = result
                if not np.all(frame == float(meta.seq)):
                    tearing_detected.set()
                    return
                reader_counts[idx] += 1
                if writer_done.is_set():
                    break
            # Final read after writer done
            final = sl.try_read()
            if final is not None:
                frame, meta = final
                if not np.all(frame == float(meta.seq)):
                    tearing_detected.set()
                    return
                reader_counts[idx] += 1

        w_thread = threading.Thread(target=writer)
        r_threads = [
            threading.Thread(target=reader, args=(i,))
            for i in range(n_readers)
        ]

        for rt in r_threads:
            rt.start()
        w_thread.start()

        w_thread.join(timeout=15)
        for rt in r_threads:
            rt.join(timeout=5)

        if FREE_THREADED_WITHOUT_GIL and tearing_detected.is_set():
            pytest.xfail(
                "intentional expected failure of the historical prototype Seqlock: its "
                "reads race with writes, and with the GIL disabled that tears frames. "
                "The v1 publisher (frames2py.publish.ImmutablePublisher) replaced it "
                "with immutable publication."
            )
        assert not tearing_detected.is_set(), "Tearing detected!"
        assert sum(reader_counts) >= n_readers, (
            f"Total reads too low: {reader_counts}"
        )

    @pytest.mark.timeout(30)
    def test_monotonic_seq_per_reader(self):
        """Readers never observe seq regression."""
        width, height = 16, 16
        sl = Seqlock((height, width), np.dtype(np.float32))
        n_writes = 50_000
        n_readers = 4
        regression_detected = threading.Event()

        def writer():
            for seq in range(1, n_writes + 1):
                buf = sl.begin_write()
                buf[:] = float(seq)
                sl.end_write(SnapshotMeta(seq=seq))

        def reader(idx: int):
            last_seq = -1
            while not regression_detected.is_set():
                result = sl.try_read()
                if result is None:
                    continue
                _, meta = result
                if meta.seq < last_seq:
                    regression_detected.set()
                    return
                last_seq = meta.seq
                if meta.seq >= n_writes:
                    return

        w = threading.Thread(target=writer)
        rs = [threading.Thread(target=reader, args=(i,)) for i in range(n_readers)]

        w.start()
        for r in rs:
            r.start()
        w.join(timeout=15)
        for r in rs:
            r.join(timeout=15)

        assert not regression_detected.is_set(), "Seq regression detected!"
