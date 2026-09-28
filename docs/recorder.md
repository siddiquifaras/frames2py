# Recording events

`frames2py.recorder` writes the events you give it to an HDF5 file that
`frames2py.adapters.hdf5` reads back unchanged. It records events, not frames: what you get
back is the event stream, which you can feed to any kernel afterwards. It is a sink you call
yourself, next to `Engine.ingest()`; the Engine knows nothing about it.

```sh
pip install "frames2py[recorder]"   # h5py and hdf5plugin, the same backends as frames2py[hdf5]
```

`import frames2py` and `import frames2py.recorder` import neither backend; `recorder.open()`
without them raises `ImportError` naming the extra.

## Record, then read back

```python
import frames2py
from frames2py import recorder
from frames2py.adapters import hdf5

engine = frames2py.Engine((1280, 720), "event_count")
with recorder.open("session.h5", sensor_size=(1280, 720)) as rec:
    for events in source:          # your camera loop, an adapter, ...
        rec.write(events)          # compression and file I/O happen here, on this thread
        engine.ingest(events)      # the Engine never calls the recorder
engine.stop()

with hdf5.open("session.h5", group="events", sensor_size=(1280, 720)) as reader:
    for events in reader:          # the same events, in the same order
        ...
```

The whole API:

```python
frames2py.recorder.open(path, *, sensor_size, group="events", compression="blosc", overwrite=False)
    .write(events)   # an EVENT_DTYPE array, as Engine.ingest() takes
    .close()         # also on leaving the with block
```

`examples/record_and_read_back.py` records the committed `sparklers` excerpt while an Engine
ingests it, reads the file back and compares it event by event.

## What is stored

A recording is one HDF5 file:

```
/events                      a group (the `group` argument; "/" puts it at the root)
    t   uint64   µs, absolute, as given
    x   uint16
    y   uint16
    p   uint8    as given: 0 is OFF, anything else ON, as everywhere in Frames2Py
    attributes (int64):
        sensor_width             from sensor_size
        sensor_height
        frames2py_format_version 1
```

The four datasets are 1-D, equally long, one element per event, in the order `write()` got
them, chunked 65,536 events at a time. There is no `t_offset`: `t` is stored as the absolute
microsecond value you passed in. Nothing else is in the file.

Events are recorded exactly as given. Out-of-order and backward timestamps, events outside
`sensor_size` and repeated events are all kept; nothing is sorted, clamped, deduplicated or
repaired. Timestamp discontinuities stay the caller's responsibility, as they are for the
Engine (see "Timestamps" in [adapters.md](adapters.md)).

`write()` checks events the way `Engine.ingest()` does, with the same code. A malformed array
(wrong dtype for `t`, `x`, `y` or `p`, byte order included, not 1-D, not C-contiguous) raises
`TypeError`; extra fields are ignored. Any event with `t >= 2**63` rejects the whole call with
`ValueError`, and nothing of that call is recorded; earlier and later calls are unaffected.

### The attributes

`sensor_width` and `sensor_height` record the geometry you declared. They are informative:
`frames2py.adapters.hdf5` does not read them, so pass `sensor_size` again when reading, as
above. They are not checked against the events either.

`frames2py_format_version` is the version of this layout, 1. The HDF5 adapter reads a group
with the attribute only if it is exactly the integer 1, and raises `ValueError` otherwise, so
a file from a later Frames2Py with a different layout is refused rather than misread. A group
without the attribute (a DSEC file, or anything else not written by the recorder) is read as
it always was.

## Compression

| `compression` | filters | who can read it | cost (measured below) |
|---|---|---|---|
| `"blosc"` (default) | Blosc, LZ4, level 5, byte shuffle, through hdf5plugin | needs the Blosc plugin | fast, about 3 bytes per event |
| `"gzip"` | gzip level 4 and the shuffle filter, built into HDF5 | any HDF5 reader | slowest, about 2 bytes per event |
| `None` | none | any HDF5 reader | fastest, 13 bytes per event |

Blosc is the default because it was the one option both compact (about a quarter of the raw
13 bytes per event) and fast enough to keep up with the recordings measured below. The cost
is portability: a plain HDF5 install doesn't have the Blosc filter, so tools outside Frames2Py
need it made available. From Python, importing hdf5plugin registers it:

```python
import h5py
import hdf5plugin  # noqa: F401  registers Blosc with HDF5

with h5py.File("session.h5") as f:
    t = f["events/t"][:]
```

Outside Python, HDF5 loads filter plugins from the directories in `HDF5_PLUGIN_PATH`, and
hdf5plugin ships Blosc as such a plugin. Pointing the variable at hdf5plugin's plugin directory
makes HDF5 tools find it:

```sh
export HDF5_PLUGIN_PATH="$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')"
```

(Checked with h5py reading a Blosc recording without importing hdf5plugin; not checked with
h5dump, HDFView or other tools.)

Use `compression="gzip"` when the file has to be read by tools you don't control: every
HDF5 reader has gzip. It is several times slower to write (below). `None` is fastest and
largest.

## Identical bytes

The recorder buffers events until it has a whole chunk of 65,536 and writes each chunk once.
The bytes of a recording therefore depend only on the events and the `open()` arguments, not
on how the events were split across `write()` calls: one call with a million events and a
million calls with one event each give the same file. The test suite checks this for each
compression across several deliberately irregular splits. The buffer holds at most one chunk
(65,535 events, about 0.85 MB); the last partial chunk is written at `close()`.

## Finishing a recording

The recorder writes to a hidden temporary file next to the target
(`.session.h5.<random>.partial`) and moves it onto the target with `os.replace()` only once the
recording has been finished and closed. So a file at the target path is always a complete,
closed recording.

- **`close()`, or leaving the `with` block normally:** the buffered events are written, the
  file is closed, and it replaces the target.
- **An exception leaving the `with` block, `KeyboardInterrupt` included:** the recording is
  finished the same way and becomes the output. Ctrl-C is a normal way to end a live
  recording; the file then holds every event of every `write()` that completed. If the
  interrupt lands inside a `write()`, that call contributes the first part of its events,
  possibly none: the recording is always a prefix of everything passed to `write()`.
- **`overwrite=False`** (the default): `open()` raises `FileExistsError` if the target exists.
  The check is repeated just before the final move; if a file appeared at the target in the
  meantime, `close()` raises `FileExistsError` and leaves the finished recording at the
  temporary path its message names.
- **`overwrite=True`:** the old target stays untouched while recording and is replaced when
  the new recording is finished, an interrupted one included. Ctrl-C during an
  `overwrite=True` recording therefore replaces the old file with the partial new one.
- **If finishing fails** (the disk fills up while the last chunk is written, or a second
  Ctrl-C lands inside `close()`), the temporary file is removed and the error raised; the
  target is unchanged.

**A crash or power loss is not covered.** If the process is killed, or the machine loses
power, the target is untouched (or absent) and a `.partial` file may be left behind. It is
not a valid recording: HDF5 files that were never closed generally can't be opened, and
Frames2Py makes no attempt to recover them. There is no journal, no SWMR mode and no fsync;
the final move puts the file in place, but whether it has reached the disk is up to the
operating system.

Other errors from `open()`: `IsADirectoryError` for a directory, `FileNotFoundError` if the
target's directory doesn't exist, `PermissionError` if it can't be written; `TypeError` and
`ValueError` for bad arguments, before any file is created. `write()` after `close()` raises
`ValueError`.

## Threads and blocking

`write()` does its work on the thread that calls it: it validates, copies into the buffer,
and whenever a chunk is full, compresses and writes it. So yes, `write()` blocks its caller on
compression and file I/O. The recorder starts no thread and has no queue, and the Engine
never calls it or waits for it: `Engine.ingest()` does exactly the same work with or without a
recorder.

The consequence: if one thread both records and ingests, as in the example above, that loop
can sustain at most what the two together sustain. The measurements below give both figures.
If the recorder's share matters, call `write()` from a thread of your own; that thread and
its hand-off are yours, not Frames2Py's.

## What recording costs

Measured with `python -m benchmarks recorder` (`benchmarks/consumers.py`) on an Apple M4, macOS
15.7.7, CPython 3.11.14 with NumPy 2.4.6, h5py 3.16.0 and hdf5plugin 7.1.0: the first 20 million
events of each recording (all of `sparklers.raw`), `open()` to `close()`, the file left in the
page cache (not fsynced), 5 runs, medians. CPython 3.14.2t with the GIL disabled (NumPy 2.5.3)
measured within 2% of these figures in every row.

| recording, its own event rate | `None` | `"gzip"` | `"blosc"` | bytes per event (none / gzip / blosc) |
|---|---|---|---|---|
| Prophesee `sparklers`, 5.4M events/s | 112M events/s | 10.1M | 62.3M | 13.1 / 1.9 / 3.1 |
| Prophesee `active_marker`, 0.7M events/s | 134M | 10.2M | 74.6M | 13.0 / 2.1 / 3.0 |
| DSEC `thun_01_a` left, 13.8M events/s | 135M | 9.7M | 64.2M | 13.0 / 2.0 / 3.3 |

These are for 100,000 events per `write()`; at 10,000 per call Blosc wrote 54-64M events/s,
at about a million 72-100M. With `write()` and `Engine.ingest()` (`event_count`) on the same
thread, 100,000 events per call, the loop sustained 51-59M events/s with Blosc, 9.3-9.8M with
gzip and 82-90M uncompressed.

So on this machine gzip did not keep up with the DSEC recording's own rate (about 0.7x); Blosc
and no compression kept up with all three with room to spare. Other machines, disks and
recordings will differ. A `write()` call that completes chunks compresses them before it
returns: on the two 20-million-event recordings its 99th-percentile time was about 2 ms with
Blosc and 13 ms with gzip at 100,000 events per call, and 12-13 and 103-107 ms at a million. The recorder's own memory is its one-chunk
buffer; peak traced memory while recording was 1.4 MiB.
