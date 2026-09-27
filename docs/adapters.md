# Reading recorded files

Three adapter modules read recordings and hand you `EVENT_DTYPE` arrays, ready for
`Engine.ingest()` or `Accumulator.accumulate()`. They sit outside the core: no adapter
owns an Engine, starts a thread, or calls `reset()`.

| module | format | decoding done by | extra |
|---|---|---|---|
| `frames2py.adapters.evt` | Prophesee RAW, EVT 2.0 and EVT 3.0 | Frames2Py's own NumPy decoder | `evt` (nothing beyond NumPy) |
| `frames2py.adapters.aedat4` | iniVation AEDAT 4.0 | [dv-processing](https://pypi.org/project/dv-processing/) | `aedat4` |
| `frames2py.adapters.hdf5` | HDF5 with 1-D `t`, `x`, `y`, `p` datasets | [h5py](https://pypi.org/project/h5py/) and [hdf5plugin](https://pypi.org/project/hdf5plugin/) | `hdf5` |

AEDAT 4.0 and HDF5 are decoded by maintained libraries; the adapters only map their output
to `EVENT_DTYPE` and check it. EVT 2.0 and 3.0 are the exception: no maintained,
pip-installable decoder keeps timestamps as recorded on every supported platform (faery
clamps them, evlib has no wheels for 3.14t or Linux ARM64, OpenEB is not on PyPI), so
Frames2Py decodes their CD events itself.

## Install

```sh
pip install "frames2py[evt]"      # EVT 2.0 / 3.0
pip install "frames2py[aedat4]"   # AEDAT 4.0, pulls in dv-processing
pip install "frames2py[hdf5]"     # HDF5, pulls in h5py and hdf5plugin
```

`import frames2py` never imports an adapter or its backend. Opening a file without the
backend installed raises `ImportError` naming the extra to install.

## Use

```python
import frames2py
from frames2py.adapters import evt

with evt.open("recording.raw") as reader:
    engine = frames2py.Engine(reader.sensor_size, "event_count")
    for events in reader:
        engine.ingest(events)
    engine.stop()
```

```python
from frames2py.adapters import aedat4, hdf5

aedat4.open("recording.aedat4", batch_size=100_000)
hdf5.open("events.h5", group="events", t_offset="/t_offset", sensor_size=(640, 480))  # a DSEC file
```

The signatures:

```python
frames2py.adapters.evt.open(path, *, sensor_size=None, batch_size=None)
frames2py.adapters.aedat4.open(path, *, sensor_size=None, batch_size=None)
frames2py.adapters.hdf5.open(path, *, group, t_offset=None, sensor_size=None, batch_size=None)
```

## The reader

`open()` returns a reader. It is single-pass: iterate it once.

- Each item is a 1-D, C-contiguous `EVENT_DTYPE` array, never empty, in the file's order.
- `batch_size=None` gives the decoder's own boundaries: one array per 64 KiB of EVT 3.0
  words, per 1 MiB read of EVT 2.0, per AEDAT 4.0 packet, per 1,048,576 HDF5 events.
  `batch_size=N` gives N events per array, the last one excepted. Only the boundaries
  change; the events are the same.
- Every array is new memory (or a view of new memory no other array shares). Writing to one
  changes nothing else, and never the file.
- `reader.sensor_size` is `(width, height)`: from the file, or the `sensor_size` you passed.
  Missing geometry is never guessed. An EVT header without geometry, or an AEDAT 4.0 stream
  without a resolution, needs `sensor_size`; one you pass must match the file's. HDF5 files
  have no geometry field, so there it is whatever you passed, or `None`.
- Leaving the `with` block, or calling `close()`, closes the file. Iterating again raises
  `RuntimeError`; iterating after `close()` raises `ValueError`.

## Timestamps

Timestamps are microseconds, and the adapters reconstruct them without changing them:
no sorting, no clamping, no shifting to start at zero, no epoch guessing, no reset. Events
that arrive out of order stay out of order; the core handles that (the watermark is the
largest timestamp seen).

When a source's clock jumps backward or restarts, the jump reaches you. Frames2Py doesn't
detect it and the adapter doesn't reset anything: if the new timestamps belong to a new
time domain, call `Engine.reset()` (or `Accumulator.reset()`) yourself.

## EVT 2.0 and EVT 3.0

The header's `% evt 2.0` / `% evt 3.0` line, or its `% format EVT2` / `% format EVT3` line,
picks the decoder; any other version (EVT 2.1 included) raises `ValueError`, and so do a
missing version or two lines that disagree. Geometry comes from `% geometry WxH` or the
format line's `width=` and `height=`. Only CD events are decoded; triggers and monitoring
words are skipped. Timestamps are the sensor clock as recorded; they are not shifted to
start at the first TIME_HIGH.

**EVT 2.0.** An event's time is the last TIME_HIGH (time bits 33..6) joined to its own 6 low
bits. The TIME_HIGH counter wraps after 2^34 µs (about 4.8 hours). A TIME_HIGH lower than the
previous one by at least `(2^28 - 1) × 64 − 10000` µs is read as a wrap, and 2^34 µs is
added from then on; any other backward step is kept as a backward step. This is OpenEB 5.2.0's
rule. CD events before the first TIME_HIGH are dropped, since their time is unknown.

**EVT 3.0.** An event's time is TIME_HIGH (bits 23..12) joined to TIME_LOW (bits 11..0), plus
2^24 µs (about 16.8 s) for each counter wrap.

- A TIME_HIGH step from 4095 to 0, or any backward step of more than 3840 units, is a wrap.
  Every other backward step is kept, and you see it as a backward jump in time.
- OpenEB 5.2.0's decoder reads any backward step of 2048 units or more as a wrap, while its
  own validator reports steps of 2048 to 3840 as violations. Frames2Py follows the validator:
  reading such a step as a wrap would turn a visible backward jump into a silent 16.8 s
  forward one. The format page says each TIME_HIGH value is sent 256 times, every 16 µs, so
  a genuine wrap is 4095 to 0; this rule still accepts one across which up to 254 TIME_HIGH
  values were lost.
- A TIME_HIGH that changes the value sets TIME_LOW to 0 until the next TIME_LOW. A TIME_LOW
  lower than the one before it is kept (the format page allows it across event sources).
- Decoding starts at the first TIME_HIGH. After it, events before the first EVT_ADDR_Y, and
  vector words before the first VECT_BASE_X, are dropped, as OpenEB does: their row or base
  is unknown.
- Vector words follow the format page: each VECT_12 or VECT_8 emits its set bits at the
  current base, then moves the base on by 12 or 8. No 12 + 12 + 8 grouping is assumed and
  nothing is dropped near the right edge (OpenEB's default decoder does both).
- Rows of type 0x1, which the format page reserves, emit nothing, and vector words inside
  them don't move the base (OpenEB's behaviour).
- A row at or beyond the sensor height is decoded and yielded; the Accumulator or Engine
  counts its events as out of bounds.

Both: reads can end anywhere; a partial word carries into the next read, so the events never
depend on read size. A partial word at the end of the file is ignored, so a truncated file
yields a prefix of the full file's events. A vector event whose x would exceed 65535, which
only a corrupt stream produces, raises `ValueError`.

On the four real recordings in the test registry (below), the output equals OpenEB 5.2.0's
default decode event for event. None of them contains a case where the rules above differ
from OpenEB's.

## AEDAT 4.0

- dv-processing reads the file's first camera, as named in its description. Of that
  camera's event streams, the one named `events` is read, or the only one there is.
  dv-processing's Python API doesn't tell which of several streams comes first in the
  description, so several event streams with none named `events` raise `ValueError`.
- Timestamps are the file's int64 microseconds. The format page describes Unix time; files
  from other sources can use another clock. A negative timestamp raises `ValueError` when
  iteration reaches its packet, and is never clamped.
- ON is `p = 1`, OFF is `p = 0`. Geometry is the stream's resolution.
- Known limitation: dv-processing has been seen to give no result for minutes on a file with
  corrupted bytes inside a packet. It happens inside dv-processing and the adapter can't cut
  it short.

On three real recordings the output equals faery 0.7's decode event for event.

## HDF5

The schema is fixed; nothing is autodetected.

- `group` holds four datasets, `t`, `x`, `y` and `p`, each 1-D and of the same length, one
  element per event, in event order. `t` is in microseconds.
- `t_offset` is added to every `t`: an int, or the path of a scalar integer dataset in the
  file. It is added only when you pass it.
- `t`, `x` and `y` are integers; `p` is an integer or a bool. Values are copied exactly or
  refused with `ValueError`: `t + t_offset` must be in `[0, 2^63)`, `x` and `y` in
  `[0, 65535]`, `p` in `[0, 255]`. The core treats `p == 0` as OFF and anything else as ON,
  so a file that stores OFF as `-1` is refused rather than silently read as all ON.
- Compressed datasets need a filter from HDF5, h5py or hdf5plugin (Blosc, Zstd, LZ4 and
  more). A dataset whose mandatory filter is missing raises `ValueError` from `open()`; one
  whose optional filter is missing (Blosc is usually stored as optional) raises
  `ValueError` when iteration reaches a chunk that needs it.

DSEC's event files follow this layout (`events/{t,x,y,p}`, `t` as uint32 relative to the
scalar `/t_offset`, Blosc compression). Prophesee's own HDF5 export does not: it stores one
compound dataset compressed with Prophesee's ECF filter, which neither h5py nor hdf5plugin
provides; read the RAW file with `evt` instead.

## Errors

| condition | exception |
|---|---|
| the path doesn't exist | `FileNotFoundError` |
| the path is a directory or can't be read | `OSError` (`IsADirectoryError`, `PermissionError`) |
| malformed input, an unsupported version or schema, a decoder or backend failure | `ValueError`, chained from the backend's exception where there is one |
| no geometry and no `sensor_size`, or a `sensor_size` that conflicts with the file | `ValueError` |
| a negative AEDAT 4.0 timestamp; an HDF5 `t + t_offset` below 0 or at or above 2^63 | `ValueError` |
| a second iteration | `RuntimeError` |
| iteration after `close()` | `ValueError` |
| the adapter's backend isn't installed | `ImportError` naming the extra, from `open()` |
| `batch_size` below 1; a `sensor_size` that isn't a pair of positive values | `ValueError` |
| a `batch_size`, `sensor_size` value, `t_offset` or `group` of the wrong type | `TypeError` |

Errors in the file's body surface when iteration reaches them, so earlier batches have
already been yielded.

## Cameras

There is no live-camera adapter. dv-processing can also read iniVation cameras, but that
path hasn't been tested against hardware, so it isn't offered. Prophesee's Metavision SDK
(and OpenEB) is not on PyPI and doesn't support macOS, so it can't be an extra; its RAW files
are read by `evt`.

## Test data

The default test suite uses small committed fixtures cut from, or derived from, Prophesee
sample recordings published under CC0 (`tests/data/PROVENANCE.md`), plus byte streams built
in the tests. Full-size recordings are fetched on request and checked against a SHA-256:

```sh
uv run python -m tests.recordings list
uv run python -m tests.recordings download          # about 400 MB into ~/.cache/frames2py/recordings
uv run pytest tests/adapters --recordings
```

| recording | format | events | licence |
|---|---|---|---|
| `sparklers.raw` | EVT 2.0, 640x480 | 521,252 | CC0 1.0 |
| `200_jets_at_200hz.raw` | EVT 2.0, 640x480 | 407,365 | CC0 1.0 |
| `active_marker.raw` | EVT 3.0, 1280x720 | 22,316,758 | CC0 1.0 |
| `faery_evt3.raw` | EVT 3.0, 1280x720 | 1,218,618 | none stated; fetched for local use only |
| `dvp_test-minimal.aedat4` | AEDAT 4.0, 640x480 | 255,283 | none stated for the file |
| `dvp_sample_data.aedat4` | AEDAT 4.0, 346x260 | 9,193 | none stated for the file |
| `faery_davis346.aedat4` | AEDAT 4.0, 346x260 | 78,830 | none stated; fetched for local use only |
| `dsec_thun_01_a_events_left.h5` | HDF5 (DSEC), 640x480 | 131,482,728 | CC BY-SA 4.0 |

These are a handful of recordings. In the two EVT 3.0 recordings TIME_HIGH only repeats or
steps by one and wraps exactly from 4095 to 0, and neither has a long idle period. Clock
resets, lost TIME_HIGH words and reserved 0x1 rows are covered only by constructed byte
streams.
