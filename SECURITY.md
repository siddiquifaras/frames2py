# Security policy

Frames2Py is a Python library; there is no hosted service. A security problem is one in the
`frames2py` package or in how it is released: for example, a crafted recording that makes an
adapter read or write outside its buffers, hang, or exhaust memory, or a flaw in the release
workflow that could put something other than the reviewed code on PyPI.

## Reporting a vulnerability

Report it privately, through GitHub's private vulnerability reporting: on the repository's
[Security tab](https://github.com/siddiquifaras/frames2py/security), choose
"Report a vulnerability". The report is visible only to you and the maintainer.

If that option is not shown, open an ordinary issue that says only that you have a security
report and asks for a private channel. Put no details, code or files in the issue.

Please don't report security problems in public issues, pull requests or discussions.

## What to include

The more of this a report has, the faster it can be reproduced:

- **Versions:** `frames2py.__version__`; Python, with whether it is a free-threaded build
  and whether the GIL is enabled (`sys._is_gil_enabled()`); NumPy; the operating system and
  architecture; the installed extras and their versions (`pip freeze` or `uv pip freeze`).
- **The component:** the core (`Engine`, `Accumulator`, a kernel, snapshots), a file adapter
  (EVT, AEDAT 4.0, HDF5), the recorder, replay, the viewer, or the release process.
- **A minimal reproducer:** the code, and for an adapter the smallest input that triggers the
  problem. A crafted file or a byte string is best; don't send recordings you have no right
  to share.
- **What happens and what you expected:** the traceback, the output of
  `python -X faulthandler` for a crash, sanitizer output if you have it, and how memory or
  time grows for a resource problem.
- **The impact as you see it:** who can trigger it, for instance anyone who can supply a file
  to a program that opens it with an adapter.

Remove secrets, tokens, credentials and personal data from logs and files before sending
them.

## Before a fix is public

Until a fixed release is available and the advisory is published:

- don't disclose the details, reproducers or exploit files publicly: no issue, pull request,
  discussion, post or talk;
- agree a disclosure date with the maintainer;
- test only on your own installations and data.

## What happens next

Frames2Py has one maintainer and is maintained on a best-effort basis, so there is no
guaranteed response time. A confirmed problem is fixed in a new release, published through
the normal release workflow; a published version is never changed or reused. The advisory is
published through GitHub Security Advisories once the fix is released, crediting the
reporter if they wish.

## Not in scope here

- **Vulnerabilities in CPython, NumPy or an optional backend** (dv-processing, h5py,
  hdf5plugin, pyglet): report them to those projects. If the problem is in how Frames2Py uses
  one of them, report it here.
- **Writing to a shared snapshot frame.** NumPy's read-only flag stops accidental writes; it
  is not a memory-protection boundary, and libraries such as `torch.from_numpy` ignore it.
  Code in the same process can modify a published frame by design, as the
  [snapshot documentation](https://siddiquifaras.github.io/frames2py/core/snapshots/#read-only-is-numpys-flag-not-memory-protection)
  states. Consumers that modify data use `snapshot.copy()`.
- **Cross-process isolation.** Consumers are threads in the producer's process; Frames2Py
  offers no isolation between them.
