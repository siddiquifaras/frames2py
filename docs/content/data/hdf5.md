# HDF5

`frames2py.adapters.hdf5.open(path, *, group, t_offset=None, sensor_size=None,
batch_size=None)` reads events stored as four 1-D datasets in an HDF5 group, through h5py and
hdf5plugin (`frames2py[hdf5]`: h5py >= 3.16, hdf5plugin >= 7.1). It reads the files the
[recorder](recorder.md) writes, DSEC's event files, and anything else in the same layout.

```python title="read_hdf5.py"
--8<-- "read_hdf5.py"
```

```text title="Output"
--8<-- "read_hdf5.out"
```

The example runs from a checkout's `tests/data/`, where `sparklers_100k.h5` holds the same
100,000 events as the EVT example, laid out as DSEC stores its event files. `t_offset` is part
of that layout; files the [recorder](recorder.md) writes store absolute timestamps and have no
`t_offset`, so open them without it: `hdf5.open(path, group="events", sensor_size=...)`.

`group` has no default when reading, because other files put their events elsewhere; the
[recorder](recorder.md) writes to `group="events"` unless told otherwise, so its recordings
open with `hdf5.open(path, group="events", ...)`.

## The schema

It is fixed; nothing is autodetected.

- `group` holds four datasets, `t`, `x`, `y` and `p`, each 1-D and of the same length, one
  element per event, in event order. `t` is in microseconds.
- `t_offset` is added to every `t`: an int, or the path of a scalar integer dataset in the
  file. It is added only when you pass it.
- `t`, `x` and `y` are integer datasets; `p` is integer or bool. Values are copied exactly or
  refused with `ValueError`: `t + t_offset` must be in `[0, 2^63)`, `x` and `y` in
  `[0, 65535]`, `p` in `[0, 255]` (a bool as 0 / 1). The core treats `p == 0` as OFF and
  anything else as ON, so a file that stores OFF as `-1` is refused rather than silently read
  as all ON.
- **Geometry:** HDF5 has no standard geometry field, so `reader.sensor_size` is the
  `sensor_size` you pass, or `None`. The recorder stores `sensor_width` and `sensor_height`
  attributes, but the reader never uses them.
- **Format version:** if `group` has a `frames2py_format_version` attribute, `open()` raises
  `ValueError` unless it is exactly the integer 1, so a file written by a later Frames2Py
  with a different layout is refused rather than misread. A group without the attribute is
  read as described above.

## Compression

Compressed datasets need their filter from HDF5, h5py or hdf5plugin (Blosc, Zstd, LZ4 and
more). A dataset whose mandatory filter is missing raises `ValueError` from `open()`; one
whose optional filter is missing (Blosc is usually stored as optional) raises `ValueError`
when iteration reaches a chunk that needs it.

Blosc-compressed files may decode faster with the `BLOSC_NTHREADS` environment variable set.
Frames2Py never sets it: it applies to the whole process, so the choice is yours. On one DSEC
file, `BLOSC_NTHREADS=4` decoded about 1.5 times faster at more total CPU time
([Performance](../reference/performance.md#adapters)).

## Other HDF5 layouts

- **DSEC** event files follow this layout: `events/{t,x,y,p}`, `t` as uint32 relative to the
  scalar `/t_offset`, Blosc compression. Open them with
  `hdf5.open(path, group="events", t_offset="/t_offset", sensor_size=(640, 480))`.
- **Prophesee's own HDF5 export** does not: it stores one compound dataset compressed with
  Prophesee's ECF filter, which neither h5py nor hdf5plugin provides. Read the RAW file with
  the [EVT adapter](evt.md) instead.
