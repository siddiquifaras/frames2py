# Handing snapshots to PyTorch

Frames2Py doesn't depend on PyTorch, doesn't import it, and has no PyTorch API: no extra, no
module, no tensor type. A consumer that wants tensors converts snapshots itself. This page is
the recipe for doing that without touching the frame other consumers share:

```text
engine.snapshot()  →  snapshot.copy()  →  torch.from_numpy(...)  →  .to(dtype)  →  .to(device)
   shared, read-only     yours              shares the copy          as needed       copies
```

```python title="torch_handoff.py"
--8<-- "torch_handoff.py"
```

```text title="Output"
--8<-- "torch_handoff.out"
```

## Who owns what

- **Frames2Py owns the published frame.** `snapshot.frame` is shared by every consumer that
  reads that publication, and Frames2Py never writes it again
  ([Snapshots](../core/snapshots.md)).
- **`snapshot.copy()` gives you a new NumPy array**, writable and C-contiguous, that nothing
  else references.
- **`torch.from_numpy()` doesn't copy.** The tensor uses the array's memory and keeps the
  array alive for as long as the tensor lives. A write through either one shows in the other.
- So the tensor is your copy. Change it, in place or otherwise, and no other consumer sees
  it. Frames2Py never sees it either: it keeps no reference to the copy.
- **With `copy(out=buffer)`** the tensor is your buffer. Each refill changes the tensor's
  values, as the last part of the example shows; `clone()` a tensor you need to keep, and
  don't refill the buffer while other code may still be reading the tensor.

## Don't convert `snapshot.frame` itself

`torch.from_numpy(snapshot.frame)` succeeds, but it is not a read-only handoff. PyTorch
doesn't support read-only tensors: the tensor shares the published frame's memory and is
writable. A write
through it, an in-place operation included, changes the frame every other consumer reads,
even though NumPy still reports the frame as read-only. PyTorch emits a `UserWarning` ("The
given NumPy array is not writable...") for the first such conversion in a process only.
`torch.as_tensor(snapshot.frame)` without a dtype change shares the memory the same way.

DLPack doesn't help. NumPy marks a read-only array's DLPack export as read-only, but the
tensor `torch.from_dlpack(snapshot.frame)` returns is writable, shares the frame's memory,
and comes without a warning.

```python
# Sketch (not runnable): what the recipe avoids
alias = torch.from_numpy(snapshot.frame)   # shares the published frame: one warning per process
alias = torch.from_dlpack(snapshot.frame)  # shares it too, writable, no warning
alias.zero_()                              # every consumer of this publication now reads zeros
```

A consumer that only reads could use either call, but nothing would stop a later in-place
operation from writing through. Frames2Py makes no claim that a tensor sharing the published
frame is safe. Copy first.

## What copies, and what doesn't

| step | copies? |
|---|---|
| `snapshot.copy()`, `snapshot.copy(out=buffer)` | yes: the whole frame, into a new array or your buffer |
| `torch.from_numpy(array)` | no: the tensor shares `array`'s memory |
| `.unsqueeze()`, `.reshape()` of a contiguous tensor, `.permute()` | no: views |
| `.contiguous()` after `.permute()` | yes |
| `.to(dtype)` | yes, when the dtype changes; otherwise it returns the same tensor |
| `.to(device)` | yes, when the device changes; otherwise it returns the same tensor |

The one copy the recipe always makes is `snapshot.copy()`. None of this is free, and none of
it happens on the producer's thread: conversion is consumer work.

## Shapes, dtypes and layouts

`torch.from_numpy` keeps the frame's shape, its memory layout (C-contiguous) and its dtype.

| kernel | frame | tensor | before arithmetic |
|---|---|---|---|
| `event_count` | `(H, W)` uint32 | `torch.uint32` | `.to(torch.int64)` or a float type |
| `polarity` | `(H, W, 2)` uint32 | `torch.uint32` | as above; `.permute(2, 0, 1)` for channels first |
| `time_surface` | `(H, W)` uint64 | `torch.uint64` | `.to(torch.int64)` |
| `exp_decay` | `(H, W)` float32 | `torch.float32` | none |
| `timestamp_decay` | `(H, W)` float32 | `torch.float32` | none |
| `StackedHistogram` | `(2, bins, H, W)` uint32 | `torch.uint32` | `.to(torch.int64)` or a float type |
| `VoxelGrid` | `(bins, H, W)` float32 | `torch.float32` | none |

**Unsigned integers: convert first.** Most `uint32` arithmetic is unsupported in PyTorch;
convert first. PyTorch's documentation says unsigned types other than `uint8` "are currently
planned to only have limited support in eager mode"
([Tensor Attributes](https://docs.pytorch.org/docs/2.14/tensor_attributes.html), PyTorch 2.14).

- **`.to(torch.int64)`** is exact for every `uint32` count. It is exact for every
  `time_surface` value too, because the event contract rejects timestamps of 2^63 and above.
- **`.to(torch.float32)`** is exact only up to 2^24 (16,777,216). Above that, not every
  integer has a float32: 16,777,217 becomes 16,777,216, and 4,294,967,295 becomes
  4,294,967,296. Counts that stay below 2^24 convert exactly. Microsecond timestamps pass
  2^24 after about 16.8 seconds, so a time surface converted straight to float32 loses
  precision; subtract a reference time in int64 first if your model wants small floats.
- **`.to(torch.float64)`** is exact for every `uint32`.

Counts wrap modulo 2^32 in Frames2Py ([Kernels](../core/kernels.md)), so a converted count is
the wrapped value.

**Layouts stay as Frames2Py publishes them.** The temporal kernels are time-first, and the
recipe doesn't transpose them:

- **`VoxelGrid`** `(bins, H, W)` is already channels-first. `unsqueeze(0)` gives the NCHW
  batch `(1, bins, H, W)` that `torch.nn.Conv2d(in_channels=bins, ...)` takes, without a copy.
- **`StackedHistogram`** `(2, bins, H, W)`: `reshape(1, 2 * bins, H, W)` merges polarity and
  time into channels, polarity-major, so channel `p * bins + b` is `frame[p, b]`. A model that
  expects another channel order needs its own permute.
- **`polarity`** `(H, W, 2)` is channels-last. `permute(2, 0, 1)` gives `(2, H, W)`, channel
  0 OFF and 1 ON, as a non-contiguous view; `.contiguous()` copies it if a model needs a
  contiguous tensor.
- **2-D frames** `(H, W)` become `(1, 1, H, W)` with `[None, None]`.

## Devices

`.to(device)` is PyTorch's: it copies the tensor to the device when the device differs and
returns the same tensor when it doesn't. Frames2Py's frames live in CPU memory, and
Frames2Py makes no GPU claim. In CI, the test that a device transfer copies runs where PyTorch
reports a device: Apple's MPS on the macOS runners. The Linux runners have none and skip it.
No CUDA device is tested.

## Tested versions

The recipe's tests and the example above run in a separate CI workflow
(`.github/workflows/torch.yml`), apart from the main CI, which never installs PyTorch:

- torch **2.14.1**, from PyTorch's CPU wheel index (`https://download.pytorch.org/whl/cpu`;
  on macOS that index serves the standard macOS wheel);
- CPython **3.11**, and **3.14t** with the GIL disabled, on Linux x86_64, Linux ARM64 and
  macOS ARM64.

In those 3.14t environments, importing torch 2.14.1 did not re-enable the GIL: the job checks
the GIL after the import and at the start and end of the test run. That is the whole finding.
It is not a statement about PyTorch on free-threaded Python in general.

Other PyTorch versions, CPython 3.12 to 3.14, and CUDA builds of PyTorch are not tested.
