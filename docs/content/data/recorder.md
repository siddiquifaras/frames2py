# Recorder

`frames2py.recorder` writes the events you give it to an HDF5 file that
[`frames2py.adapters.hdf5`](hdf5.md) reads back unchanged. It records events, not frames:
what you get back is the event stream, which you can feed to any kernel afterwards. It is a
sink your own loop calls, next to `Engine.ingest()`; the Engine knows nothing about it.

Install `frames2py[recorder]` (h5py and hdf5plugin, the same backends as `frames2py[hdf5]`).
`import frames2py.recorder` imports neither; `recorder.open()` without them raises
`ImportError` naming the extra.

## Record, then read back

```python title="record_read_back.py"
--8<-- "record_read_back.py"
```

```text title="Output"
--8<-- "record_read_back.out"
```

The API is `recorder.open(path, *, sensor_size, group="events", compression="blosc",
overwrite=False)`, which returns a recorder with `write(events)` and `close()`, used as a
context manager. Full signatures: [API reference](../reference/api/tools.md).

## What is stored

A recording is one HDF5 file:

```text
/events                      a group (the `group` argument; "/" puts it at the root)
    t   uint64   µs, absolute, as given
    x   uint16
    y   uint16
    p   uint8    as given: 0 is OFF, anything else ON
    attributes (int64):
        sensor_width             from sensor_size
        sensor_height
        frames2py_format_version 1
```

The four datasets are 1-D, equally long, one element per event, in the order `write()` got
them, chunked 65,536 events at a time. There is no `t_offset`: `t` is stored as the absolute
microsecond value you passed in. Nothing else is in the file.

**Events are recorded exactly as given.** Out-of-order and backward timestamps, events
outside `sensor_size` and repeated events are all kept; nothing is sorted, clamped,
deduplicated or repaired. Timestamp discontinuities stay the caller's responsibility, as for
the Engine.

**Validation.** `write()` checks events the way `Engine.ingest()` does, with the same code.
A malformed array (wrong dtype for `t`, `x`, `y` or `p`, byte order included, not 1-D, not
C-contiguous) raises `TypeError`; extra fields are ignored. Any event with `t >= 2**63`
rejects the whole call with `ValueError`, and nothing of that call is recorded; earlier and
later calls are unaffected.

**Reading it back** takes `hdf5.open(path, group="events")`: the reader's `group` has no
default. The reader doesn't take the geometry from the file either, so pass
`sensor_size=...` as well if you want `reader.sensor_size` (for example to build an Engine
from it); without it, `reader.sensor_size` is `None`.

**The attributes.** `sensor_width` and `sensor_height` record the geometry you declared.
They are informative: the HDF5 adapter doesn't read them. They are not checked against the
events either. `frames2py_format_version` is the
version of this layout, 1; the HDF5 adapter refuses a group whose version isn't exactly 1.

## Compression

| `compression` | filters | who can read it | measured cost |
|---|---|---|---|
| `"blosc"` (default) | Blosc, LZ4, level 5, byte shuffle, through hdf5plugin | needs the Blosc plugin | fast, about 3 bytes per event |
| `"gzip"` | gzip level 4 and the shuffle filter, built into HDF5 | any HDF5 reader | slowest, about 2 bytes per event |
| `None` | none | any HDF5 reader | fastest, 13 bytes per event |

Blosc is the default because it was the one option both compact (about a quarter of the raw
13 bytes per event) and fast enough to keep up with the recordings measured
([Performance](../reference/performance.md#recorder)). The cost is portability: a plain HDF5
install doesn't have the Blosc filter. From Python, importing hdf5plugin registers it:

```python
# Sketch (not runnable): needs a recording called session.h5.
import h5py
import hdf5plugin  # noqa: F401  registers Blosc with HDF5

with h5py.File("session.h5") as f:
    t = f["events/t"][:]
```

Outside Python, HDF5 loads filter plugins from the directories in `HDF5_PLUGIN_PATH`, and
hdf5plugin ships Blosc as such a plugin:

```sh
export HDF5_PLUGIN_PATH="$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')"
```

That was checked with h5py reading a Blosc recording without importing hdf5plugin; not with
h5dump, HDFView or other tools. Use `compression="gzip"` when the file has to be read by
tools you don't control.

## Identical bytes

The recorder buffers events until it has a whole chunk of 65,536 and writes each chunk once.
The bytes of a recording therefore depend only on the events and the `open()` arguments, not
on how the events were split across `write()` calls: one call with a million events and a
million calls with one event each give the same file. The test suite checks this for each
compression across several irregular splits. The buffer holds at most one chunk (65,535
events, about 0.85 MB); the last partial chunk is written at `close()`.

## Finishing a recording

The recorder writes to a hidden temporary file next to the target
(`.session.h5.<random>.partial`) and moves it onto the target with `os.replace()` only once
the recording has been finished and closed. A file at the target path is therefore always a
complete, closed recording.

- **`close()`, or leaving the `with` block normally:** the buffered events are written, the
  file is closed, and it replaces the target.
- **An exception leaving the `with` block, `KeyboardInterrupt` included:** the recording is
  finished the same way and becomes the output. Ctrl-C is a normal way to end a live
  recording; the file then holds every event of every `write()` that completed. If the
  interrupt lands inside a `write()`, the part of that call's events already written stays
  in the recording, possibly none of it: the recording is always a prefix of everything
  passed to `write()`. A rejected call (malformed, or `t >= 2**63`) is different: nothing of
  it is buffered or written.
- **`overwrite=False`** (the default): `open()` raises `FileExistsError` if the target
  exists. The check is repeated just before the final move; if a file appeared at the target
  in the meantime, `close()` raises `FileExistsError` and leaves the finished recording at
  the temporary path its message names.
- **`overwrite=True`:** the old target stays untouched while recording and is replaced when
  the new recording is finished, an interrupted one included.
- **If finishing fails** (the disk fills while the last chunk is written, or a second
  Ctrl-C lands inside `close()`), the temporary file is removed and the error raised; the
  target is unchanged.

**A crash or power loss is not covered.** If the process is killed or the machine loses
power, the target is untouched (or absent) and a `.partial` file may be left behind. It is
not a valid recording: HDF5 files that were never closed generally can't be opened, and
Frames2Py makes no attempt to recover them. There is no journal, no SWMR mode and no fsync;
whether the moved file has reached the disk is up to the operating system.

Other errors from `open()`: `IsADirectoryError` for a directory, `FileNotFoundError` if the
target's directory doesn't exist, `PermissionError` if it can't be written; `TypeError` and
`ValueError` for bad arguments, before any file is created. `write()` after `close()`
raises `ValueError`.

## Threads and blocking

`write()` does its work on the thread that calls it: it validates, copies into the buffer,
and whenever a chunk is full, compresses and writes it. So `write()` blocks its caller on
compression and file I/O. The recorder starts no thread and has no queue, and the Engine
never calls it or waits for it.

The consequence: if one thread both records and ingests, as in the example, that loop
sustains at most what the two together sustain. The measured rates are on the
[Performance](../reference/performance.md#recorder) page. If the recorder's share matters,
call `write()` from a thread of your own; that thread and its hand-off are yours, not
Frames2Py's.
