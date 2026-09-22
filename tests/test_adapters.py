"""Tests for adapters: AEDAT4 roundtrip, UDP pack/unpack, converter matrix."""

from __future__ import annotations

import os
import socket
import struct
import tempfile
import threading
import time

import numpy as np
import pytest

from frames2py.core.types import EVENT_DTYPE


def _make_events(n: int, w: int = 640, h: int = 480, seed: int = 42):
    rng = np.random.default_rng(seed)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = np.arange(1000, 1000 + n, dtype=np.uint64)
    events["x"] = rng.integers(0, w, size=n, dtype=np.uint16)
    events["y"] = rng.integers(0, h, size=n, dtype=np.uint16)
    events["p"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    return events


# ===================================================================
# AEDAT4 adapter
# ===================================================================
class TestAedat4Roundtrip:
    """Write AEDAT4 → read back → verify identity."""

    def test_roundtrip_small(self, tmp_path):
        from frames2py.adapters.aedat4 import from_aedat4, to_aedat4

        events = _make_events(500, w=346, h=260)
        path = str(tmp_path / "test.aedat4")
        to_aedat4(path, events, sensor_size=(346, 260))

        recovered = []
        for batch, meta in from_aedat4(path, chunk_size=1000):
            recovered.append(batch)
            assert meta.source == "aedat4"

        result = np.concatenate(recovered) if recovered else np.empty(0, dtype=EVENT_DTYPE)
        assert len(result) == 500

        # Verify timestamps survive roundtrip (int32 truncation for large ts)
        np.testing.assert_array_equal(
            result["t"][:10],
            events["t"][:10].astype(np.int32).astype(np.uint64),
        )

    def test_roundtrip_polarity(self, tmp_path):
        from frames2py.adapters.aedat4 import from_aedat4, to_aedat4

        events = _make_events(200, w=100, h=80)
        path = str(tmp_path / "pol.aedat4")
        to_aedat4(path, events, sensor_size=(100, 80))

        recovered = list(from_aedat4(path, chunk_size=500))
        result = np.concatenate([b for b, _ in recovered])

        # Polarity should survive
        np.testing.assert_array_equal(result["p"], events["p"])

    def test_nonexistent_file_raises(self):
        from frames2py.adapters.aedat4 import from_aedat4

        with pytest.raises(FileNotFoundError):
            list(from_aedat4("/nonexistent/path.aedat4"))


# ===================================================================
# UDP adapter
# ===================================================================
class TestUdpPackUnpack:
    """Test UDP wire format serialization/deserialization."""

    def test_pack_unpack_roundtrip(self):
        from frames2py.adapters.udp import pack_events_udp, unpack_events_udp

        events = _make_events(100)
        wire = pack_events_udp(events)
        result = unpack_events_udp(wire)

        assert result is not None
        assert len(result) == 100
        np.testing.assert_array_equal(result["t"], events["t"])
        np.testing.assert_array_equal(result["x"], events["x"])
        np.testing.assert_array_equal(result["y"], events["y"])
        np.testing.assert_array_equal(result["p"], events["p"])

    def test_unpack_bad_magic(self):
        from frames2py.adapters.udp import unpack_events_udp

        bad = struct.pack("<III", 0xDEADBEEF, 10, 0) + b"\x00" * 130
        assert unpack_events_udp(bad) is None

    def test_unpack_truncated(self):
        from frames2py.adapters.udp import unpack_events_udp

        assert unpack_events_udp(b"\x00\x01") is None

    def test_unpack_short_payload(self):
        from frames2py.adapters.udp import unpack_events_udp, _MAGIC

        header = struct.pack("<III", _MAGIC, 100, 0)
        short_payload = b"\x00" * 10
        assert unpack_events_udp(header + short_payload) is None

    @pytest.mark.timeout(10)
    def test_send_receive_loopback(self):
        from frames2py.adapters.udp import from_udp, to_udp

        events = _make_events(50)
        port = 15432

        received: list[np.ndarray] = []

        def receiver():
            for batch, meta in from_udp(host="127.0.0.1", port=port, timeout=3.0):
                received.append(batch)
                if len(received) >= 1:
                    return

        t = threading.Thread(target=receiver)
        t.start()
        time.sleep(0.3)  # let receiver bind

        to_udp(events, host="127.0.0.1", port=port)

        t.join(timeout=5.0)
        assert len(received) > 0
        np.testing.assert_array_equal(received[0]["t"], events["t"])


# ===================================================================
# Format converter
# ===================================================================
class TestConverter:
    """Test the universal format converter: read/write/convert for all formats."""

    def test_npy_roundtrip(self, tmp_path):
        from frames2py.adapters.convert import read_events, write_events

        events = _make_events(200)
        path = str(tmp_path / "test.npy")
        write_events(path, events)
        result = read_events(path)
        assert len(result) == 200
        np.testing.assert_array_equal(result["t"], events["t"])

    def test_csv_roundtrip(self, tmp_path):
        from frames2py.adapters.convert import read_events, write_events

        events = _make_events(50)
        path = str(tmp_path / "test.csv")
        write_events(path, events)
        result = read_events(path)
        assert len(result) == 50
        np.testing.assert_array_equal(result["t"], events["t"])
        np.testing.assert_array_equal(result["p"], events["p"])

    def test_convert_npy_to_csv(self, tmp_path):
        from frames2py.adapters.convert import convert, read_events, write_events

        events = _make_events(100)
        npy_path = str(tmp_path / "src.npy")
        csv_path = str(tmp_path / "dst.csv")
        write_events(npy_path, events)

        n = convert(npy_path, csv_path)
        assert n == 100

        result = read_events(csv_path)
        np.testing.assert_array_equal(result["t"], events["t"])

    def test_convert_csv_to_npy(self, tmp_path):
        from frames2py.adapters.convert import convert, read_events, write_events

        events = _make_events(80)
        csv_path = str(tmp_path / "src.csv")
        npy_path = str(tmp_path / "dst.npy")
        write_events(csv_path, events)

        n = convert(csv_path, npy_path)
        assert n == 80

        result = read_events(npy_path)
        np.testing.assert_array_equal(result["x"], events["x"])

    def test_aedat4_to_npy(self, tmp_path):
        from frames2py.adapters.aedat4 import to_aedat4
        from frames2py.adapters.convert import convert, read_events

        events = _make_events(150, w=100, h=80)
        aedat_path = str(tmp_path / "src.aedat4")
        npy_path = str(tmp_path / "dst.npy")

        to_aedat4(aedat_path, events, sensor_size=(100, 80))
        n = convert(aedat_path, npy_path)
        assert n == 150

    def test_unknown_format_raises(self):
        from frames2py.adapters.convert import read_events

        with pytest.raises(ValueError, match="Cannot infer format"):
            read_events("file.xyz")

    def test_explicit_format_override(self, tmp_path):
        from frames2py.adapters.convert import read_events, write_events

        events = _make_events(30)
        # np.save appends .npy if missing, so use .npy extension
        path = str(tmp_path / "data.npy")
        write_events(path, events, fmt="npy")
        result = read_events(path, fmt="npy")
        assert len(result) == 30


# ===================================================================
# H5 adapter (conditional on h5py availability)
# ===================================================================
class TestH5Adapter:
    """H5 roundtrip tests -- skip if h5py not installed."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_h5py(self):
        pytest.importorskip("h5py")

    def test_compound_roundtrip(self, tmp_path):
        from frames2py.adapters.h5 import from_h5, to_h5

        events = _make_events(300)
        path = str(tmp_path / "test.h5")
        to_h5(path, events)

        recovered = []
        for batch, meta in from_h5(path, chunk_size=100):
            recovered.append(batch)
            assert meta.source == "h5"

        result = np.concatenate(recovered)
        assert len(result) == 300
        np.testing.assert_array_equal(result["t"], events["t"])

    def test_separate_dataset_roundtrip(self, tmp_path):
        import h5py
        from frames2py.adapters.h5 import from_h5

        events = _make_events(100)
        path = str(tmp_path / "separate.h5")

        with h5py.File(path, "w") as f:
            g = f.create_group("events")
            g.create_dataset("t", data=events["t"])
            g.create_dataset("x", data=events["x"])
            g.create_dataset("y", data=events["y"])
            g.create_dataset("p", data=events["p"])

        recovered = list(from_h5(path, chunk_size=50))
        result = np.concatenate([b for b, _ in recovered])
        assert len(result) == 100
        np.testing.assert_array_equal(result["t"], events["t"])

    def test_monotonic_flag_set_correctly(self, tmp_path):
        from frames2py.adapters.h5 import from_h5, to_h5

        events = _make_events(50)
        path = str(tmp_path / "mono.h5")
        to_h5(path, events)

        for batch, meta in from_h5(path, chunk_size=1000):
            assert meta.monotonic is True  # timestamps are monotonic

    def test_h5_convert_roundtrip(self, tmp_path):
        from frames2py.adapters.convert import convert, read_events, write_events

        events = _make_events(100)
        h5_path = str(tmp_path / "src.h5")
        npy_path = str(tmp_path / "dst.npy")

        write_events(h5_path, events)
        n = convert(h5_path, npy_path)
        assert n == 100

        result = read_events(npy_path)
        np.testing.assert_array_equal(result["t"], events["t"])

    def test_1m_events_h5_roundtrip(self, tmp_path):
        """1M events through H5: write → read → verify field identity."""
        from frames2py.adapters.h5 import from_h5, to_h5

        n = 1_000_000
        events = _make_events(n, w=1280, h=720, seed=77)
        path = str(tmp_path / "big.h5")
        to_h5(path, events, compression="gzip")

        recovered: list[np.ndarray] = []
        for batch, meta in from_h5(path, chunk_size=100_000):
            recovered.append(batch)
            assert meta.source == "h5"
            assert meta.monotonic is True

        result = np.concatenate(recovered)
        assert len(result) == n
        np.testing.assert_array_equal(result["t"], events["t"])
        np.testing.assert_array_equal(result["x"], events["x"])
        np.testing.assert_array_equal(result["y"], events["y"])
        np.testing.assert_array_equal(result["p"], events["p"])

    def test_h5_separate_datasets_large(self, tmp_path):
        """500K events via separate t/x/y/p datasets."""
        import h5py
        from frames2py.adapters.h5 import from_h5

        n = 500_000
        events = _make_events(n, w=640, h=480, seed=55)
        path = str(tmp_path / "separate_big.h5")

        with h5py.File(path, "w") as f:
            g = f.create_group("events")
            g.create_dataset("t", data=events["t"])
            g.create_dataset("x", data=events["x"])
            g.create_dataset("y", data=events["y"])
            g.create_dataset("p", data=events["p"])

        recovered = list(from_h5(path, chunk_size=50_000))
        result = np.concatenate([b for b, _ in recovered])
        assert len(result) == n
        np.testing.assert_array_equal(result["t"], events["t"])
