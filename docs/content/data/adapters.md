# Adapters

Three adapter modules read recordings and yield `EVENT_DTYPE` arrays, ready for
`Engine.ingest()` or `Accumulator.accumulate()`. They sit outside the core: no adapter owns
an Engine, starts a thread or calls `reset()`.

| module | format | decoding done by | extra |
|---|---|---|---|
| [`frames2py.adapters.evt`](evt.md) | Prophesee RAW, EVT 2.0 and EVT 3.0 | Frames2Py's own NumPy decoder | `evt` (nothing beyond NumPy) |
| [`frames2py.adapters.aedat4`](aedat4.md) | iniVation AEDAT 4.0 | [dv-processing](https://pypi.org/project/dv-processing/) | `aedat4` |
| [`frames2py.adapters.hdf5`](hdf5.md) | HDF5 with 1-D `t`, `x`, `y`, `p` datasets | [h5py](https://pypi.org/project/h5py/) and [hdf5plugin](https://pypi.org/project/hdf5plugin/) | `hdf5` |

AEDAT 4.0 and HDF5 are decoded by maintained libraries; the adapters map their output to
`EVENT_DTYPE` and check it. EVT 2.0 and 3.0 are the exception: no maintained,
pip-installable decoder keeps timestamps as recorded on every supported platform (faery
clamps them, evlib has no wheels for 3.14t or Linux ARM64, OpenEB is not on PyPI), so
Frames2Py decodes their CD events itself. Other formats stay with the tools that read them
(Tonic, dv-processing, evutils, AEStream, faery, vendor SDKs): convert their output to
`EVENT_DTYPE` and pass it in.

`import frames2py` never imports an adapter or its backend. `open()` without the backend
installed raises `ImportError` naming the extra ([Installation](../getting-started/installation.md#optional-extras)).

## Reading a file

```python title="read_evt.py"
--8<-- "read_evt.py"
```

```text title="Output"
--8<-- "read_evt.out"
```

The example uses an `Accumulator` because it wants totals over the whole file. Fed to an
`Engine` instead, a windowed kernel such as `polarity` starts a new window at every
publication, so the last snapshot holds only the events since the previous one: to count a
whole file through an Engine, add up every published window (a runnable example is under
[Seeing every publication](../core/engine.md#seeing-every-publication) on the Engine page), or use a running
kernel.

Every adapter module has one function, `open(path, *, ...)`, which returns a **reader**:

- **Single pass.** Iterate it once. A second iteration raises `RuntimeError`; iterating
  after `close()` raises `ValueError`.
- **A context manager.** Leaving the `with` block, or calling `close()`, closes the file or
  decoder. Closing twice is harmless.
- **What it yields:** 1-D, C-contiguous `EVENT_DTYPE` arrays, never empty, in the file's
  order. Every array is new memory (or a view of new memory no other array shares): writing
  to one changes nothing else, and never the file.
- **`reader.sensor_size`** is `(width, height)`: from the file, or the `sensor_size` you
  passed. Missing geometry is never guessed. When the file has geometry, a `sensor_size`
  you pass must match it (`ValueError` otherwise); when it has none, the value you pass is
  used unchecked, so make sure it is the sensor's. HDF5 files have no geometry field, so
  there it is whatever you passed, or `None`.

## Batching

`batch_size=None` (the default) yields the decoder's own boundaries: one array per 64 KiB of
EVT 3.0 words, per 1 MiB read of EVT 2.0, per AEDAT 4.0 packet, per 1,048,576 HDF5 events.
`batch_size=N` yields N events per array, the last one excepted. Only the boundaries change;
the events are the same.

Batch size matters for throughput, because every `ingest()` call has a fixed cost on top of
its per-event work. AEDAT 4.0 files in particular come in small packets (tens to hundreds of
events); pass a `batch_size` around 10,000 when throughput matters ([AEDAT 4.0](aedat4.md)).
Larger batches hold more events in memory and deliver them later. For `exp_decay`, whose
decay is per call, the batching also changes the result
([Kernels](../core/kernels.md#expdecay)).

## Timestamps

Timestamps are microseconds, and the adapters reconstruct them without changing them: no
sorting, no clamping, no shift to start at zero, no epoch guessing, no reset. Events that
arrive out of order stay out of order; the core handles that (the watermark is the largest
timestamp seen).

When a source's clock jumps backward or restarts, the jump reaches you. Frames2Py doesn't
detect it, and the adapter doesn't reset anything: if the new timestamps belong to a new time
domain, call `Engine.reset()` (or `Accumulator.reset()`) yourself. The one repair an adapter
does make is unwrapping a narrow hardware counter into 64-bit microseconds, which the
[EVT decoder](evt.md) does for the EVT counters.

## Errors

| condition | exception |
|---|---|
| the path doesn't exist | `FileNotFoundError` |
| the path is a directory or can't be read | `OSError` (`IsADirectoryError`, `PermissionError`) |
| malformed input, an unsupported version or schema, a decoder or backend failure | `ValueError`, chained from the backend's exception where there is one |
| no geometry and no `sensor_size`, or a `sensor_size` that conflicts with the file | `ValueError` |
| a negative AEDAT 4.0 timestamp or coordinate; an HDF5 `t + t_offset` below 0 or at or above 2^63 | `ValueError` |
| a second iteration | `RuntimeError` |
| iteration after `close()` | `ValueError` |
| the adapter's backend isn't installed | `ImportError` naming the extra, from `open()` |
| `batch_size` below 1; a `sensor_size` that isn't a pair of positive values | `ValueError` |
| a `batch_size`, `sensor_size` value, `t_offset` or `group` of the wrong type | `TypeError` |

Errors in the body of a file surface when iteration reaches them, so earlier batches have
already been yielded.

## Live cameras and vendor SDKs

Frames2Py ships no vendor SDK adapter. A live-camera adapter is only offered once its path
has been validated against real hardware, and none can be yet: dv-processing can read
iniVation cameras, but that path hasn't been tested with a camera; Prophesee's Metavision SDK
(and OpenEB) is not on PyPI and doesn't support macOS. Replaying recordings doesn't validate
a live path.

Until then, feed a camera SDK's output to the Engine yourself: build a 1-D `EVENT_DTYPE`
array from each buffer the SDK delivers and call `ingest()`, as in the
[event contract](../core/event-contract.md#building-event-arrays). Recordings are covered by
the file adapters: Prophesee RAW by `evt`, AEDAT 4.0 by `aedat4`.

## Measured cost

Decode rates for eight real recordings, on one machine, are on the
[Performance](../reference/performance.md#adapters) page. In those files decoding cost more than
ingesting; read the figures as measurements of those files, not as a rate for yours.
