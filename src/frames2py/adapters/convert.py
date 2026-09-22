"""Universal event format converter.

Provides :func:`convert` for zero-friction conversion between all
supported event data formats:

- **h5** -- HDF5 compound dataset (requires ``h5py``)
- **aedat4** -- iniVation AEDAT4 binary (pure Python, no vendor SDK)
- **npy** -- NumPy ``.npy`` structured array
- **csv** -- Human-readable CSV
- **udp_bin** -- Raw UDP wire-format binary

No format lock-in: any format in → any format out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frames2py.core.types import EVENT_DTYPE, EventBatch


# ------------------------------------------------------------------
# Readers -- format → EVENT_DTYPE array
# ------------------------------------------------------------------

def read_events(path: str, fmt: str | None = None) -> EventBatch:
    """Read events from *path* in the specified (or inferred) format.

    Parameters:
        path: Filesystem path to the event file.
        fmt: One of ``"h5"``, ``"aedat4"``, ``"npy"``, ``"csv"``.
            If ``None``, inferred from the file extension.

    Returns:
        1-D structured array with dtype :data:`EVENT_DTYPE`.
    """
    if fmt is None:
        fmt = _infer_format(path)

    readers = {
        "h5": _read_h5,
        "hdf5": _read_h5,
        "aedat4": _read_aedat4,
        "aedat": _read_aedat4,
        "npy": _read_npy,
        "csv": _read_csv,
    }
    reader = readers.get(fmt)
    if reader is None:
        raise ValueError(
            f"Unknown format {fmt!r}. Supported: {sorted(readers)}"
        )
    return reader(path)


def write_events(
    path: str,
    events: EventBatch,
    fmt: str | None = None,
    **kwargs: object,
) -> None:
    """Write events to *path* in the specified (or inferred) format.

    Parameters:
        path: Output filesystem path.
        events: 1-D structured array with dtype :data:`EVENT_DTYPE`.
        fmt: One of ``"h5"``, ``"aedat4"``, ``"npy"``, ``"csv"``.
            If ``None``, inferred from the file extension.
        **kwargs: Passed to the format-specific writer (e.g.
            ``compression="gzip"`` for H5).
    """
    if fmt is None:
        fmt = _infer_format(path)

    writers = {
        "h5": _write_h5,
        "hdf5": _write_h5,
        "aedat4": _write_aedat4,
        "aedat": _write_aedat4,
        "npy": _write_npy,
        "csv": _write_csv,
    }
    writer = writers.get(fmt)
    if writer is None:
        raise ValueError(
            f"Unknown format {fmt!r}. Supported: {sorted(writers)}"
        )
    writer(path, events, **kwargs)


def convert(
    src: str,
    dst: str,
    src_fmt: str | None = None,
    dst_fmt: str | None = None,
    **kwargs: object,
) -> int:
    """Convert an event file from one format to another.

    Parameters:
        src: Source file path.
        dst: Destination file path.
        src_fmt: Source format (inferred from extension if ``None``).
        dst_fmt: Destination format (inferred from extension if ``None``).
        **kwargs: Passed to the destination writer.

    Returns:
        Number of events converted.
    """
    events = read_events(src, fmt=src_fmt)
    write_events(dst, events, fmt=dst_fmt, **kwargs)
    return len(events)


# ------------------------------------------------------------------
# Format inference
# ------------------------------------------------------------------

_EXT_MAP = {
    ".h5": "h5",
    ".hdf5": "h5",
    ".aedat4": "aedat4",
    ".aedat": "aedat4",
    ".npy": "npy",
    ".csv": "csv",
    ".txt": "csv",
}


def _infer_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    fmt = _EXT_MAP.get(ext)
    if fmt is None:
        raise ValueError(
            f"Cannot infer format from extension {ext!r}. "
            f"Supported: {sorted(set(_EXT_MAP.values()))}"
        )
    return fmt


# ------------------------------------------------------------------
# H5
# ------------------------------------------------------------------

def _read_h5(path: str) -> EventBatch:
    import h5py

    with h5py.File(path, "r") as f:
        if "events" in f:
            node = f["events"]
            import h5py as h5m

            if isinstance(node, h5m.Dataset):
                raw = node[:]
                out = np.empty(len(raw), dtype=EVENT_DTYPE)
                out["t"] = raw["t"]
                out["x"] = raw["x"]
                out["y"] = raw["y"]
                out["p"] = raw["p"]
                return out
            elif isinstance(node, h5m.Group):
                n = node["t"].shape[0]
                out = np.empty(n, dtype=EVENT_DTYPE)
                out["t"] = node["t"][:]
                out["x"] = node["x"][:]
                out["y"] = node["y"][:]
                out["p"] = node["p"][:]
                return out
        raise ValueError("No 'events' dataset found in H5 file")


def _write_h5(path: str, events: EventBatch, **kwargs: object) -> None:
    from frames2py.adapters.h5 import to_h5

    compression = kwargs.get("compression", "gzip")
    to_h5(path, events, compression=compression)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# AEDAT4
# ------------------------------------------------------------------

def _read_aedat4(path: str) -> EventBatch:
    from frames2py.adapters.aedat4 import from_aedat4

    parts = [batch for batch, _meta in from_aedat4(path, chunk_size=500_000)]
    if not parts:
        return np.array([], dtype=EVENT_DTYPE)
    return np.concatenate(parts)


def _write_aedat4(path: str, events: EventBatch, **kwargs: object) -> None:
    from frames2py.adapters.aedat4 import to_aedat4

    sensor_size = kwargs.get("sensor_size", (640, 480))
    to_aedat4(path, events, sensor_size=sensor_size)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# NumPy .npy
# ------------------------------------------------------------------

def _read_npy(path: str) -> EventBatch:
    raw = np.load(path)
    if raw.dtype == EVENT_DTYPE:
        return raw
    out = np.empty(len(raw), dtype=EVENT_DTYPE)
    out["t"] = raw["t"]
    out["x"] = raw["x"]
    out["y"] = raw["y"]
    out["p"] = raw["p"]
    return out


def _write_npy(path: str, events: EventBatch, **kwargs: object) -> None:
    np.save(path, events)


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------

def _read_csv(path: str) -> EventBatch:
    """Read events from a CSV file with columns t, x, y, p."""
    raw = np.loadtxt(
        path,
        delimiter=",",
        dtype=np.int64,
        skiprows=1,  # skip header
        ndmin=2,
    )
    n = raw.shape[0]
    out = np.empty(n, dtype=EVENT_DTYPE)
    out["t"] = raw[:, 0].astype(np.uint64)
    out["x"] = raw[:, 1].astype(np.uint16)
    out["y"] = raw[:, 2].astype(np.uint16)
    out["p"] = raw[:, 3].astype(np.uint8)
    return out


def _write_csv(path: str, events: EventBatch, **kwargs: object) -> None:
    """Write events to a CSV file with header t,x,y,p."""
    header = "t,x,y,p"
    data = np.column_stack([
        events["t"].astype(np.int64),
        events["x"].astype(np.int64),
        events["y"].astype(np.int64),
        events["p"].astype(np.int64),
    ])
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%d")
