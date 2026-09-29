# AEDAT 4.0

`frames2py.adapters.aedat4.open(path, *, sensor_size=None, batch_size=None)` reads iniVation
AEDAT 4.0 files through [dv-processing](https://pypi.org/project/dv-processing/), which the
`aedat4` extra installs (dv-processing >= 2.0.4).

```python title="read_aedat4.py"
--8<-- "read_aedat4.py"
```

```text title="Output"
--8<-- "read_aedat4.out"
```

`sparklers_100k.aedat4` is the same 100,000 Prophesee events as the EVT example, re-encoded
as AEDAT 4.0 by dv-processing (`tests/data/` in the repository). It is derived data, not an
iniVation recording, and its timestamps are the Prophesee sensor clock.

## Pass a `batch_size` if throughput matters

AEDAT 4.0 files store events in packets, and without `batch_size` each packet becomes one
array. In the files measured, packets held a median of 36 to 944 events, and ingesting arrays
that small is dominated by the Engine's fixed per-call cost. With `batch_size=10_000` the same
events ingested 2 to 6 times faster ([Performance](../reference/performance.md#adapters)).
Larger batches hold more events in memory and deliver them later; pick the size for your
latency needs.

## What is read

- dv-processing reads the file's first camera, as named in its description. Of that
  camera's event streams, the one named `events` is read, or the only one there is.
  dv-processing's Python API doesn't tell which of several streams comes first in the
  description, so the adapter refuses to guess: several event streams with none named
  `events`, or a first camera with no event stream, raise `ValueError`.
- **Timestamps** are the file's int64 microseconds, unchanged. The format page describes Unix
  time; files from other sources can use another clock. A negative timestamp raises
  `ValueError` when iteration reaches its packet, and is never clamped.
- **Coordinates:** a negative `x` or `y` raises `ValueError` when iteration reaches its
  packet; nothing is clamped or wrapped.
- **Polarity:** ON is `p = 1`, OFF is `p = 0`.
- **Geometry** is the stream's resolution; a stream without one needs `sensor_size`.

## Platform notes

- dv-processing 2.0.4 publishes macOS ARM64 wheels for macOS 15 and later only. On older
  macOS, `frames2py[aedat4]` has no wheel to install.
- On Linux, its wheels load the system's `libatomic.so.1`: on Debian or Ubuntu,
  `apt install libatomic1`.
- dv-processing has been seen to give no result for minutes on a file with corrupted bytes
  inside a packet. That happens inside dv-processing, and the adapter can't cut it short.

On three real recordings the adapter's output equals faery 0.7's decode, event for event.
