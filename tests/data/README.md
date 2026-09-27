# Test fixtures

Small committed files for the default test suite. Each is derived from a real recording
whose licence allows redistribution; `derive.py` rebuilds them from the source recordings
(`uv run python -m tests.data.derive --check` compares without writing). The source
recordings themselves are not committed: `tests/recordings.py` fetches and hash-checks them.

| file | bytes | SHA-256 | derived from |
|---|---|---|---|
| `sparklers_100k.evt2.raw` | 403,218 | `5570baca8e7cf16ff62e2f067008b54dce0a5daeb17bf231251c125af3219e16` | `sparklers.raw`: its header and every word up to its 100,000th CD event |
| `active_marker_head.evt3.raw` | 262,457 | `39f9a88866453449573222b7502134bc3f7b4ccb4a9d63757db31e75235f5a24` | `active_marker.raw`: its header and the first 262,144 bytes of its body |
| `sparklers_100k.aedat4` | 601,525 | `0b79d227ad39d41cac3cba68acc52dbdf7f4ef95747ff9b7775a19a6b4d22b97` | the events of `sparklers_100k.evt2.raw`, written by dv-processing 2.0.4 |

## Sources

Both are Prophesee sample recordings from https://docs.prophesee.ai/stable/datasets.html,
where the page states: "Those files are shared under the Creative Commons Zero v1.0
Universal (CC0 1.0) license. This means that you are free to copy, modify, distribute, and
use the data for any purpose, including commercial purposes, without asking permission."
(checked 2026-09-28).

| recording | format | sensor | bytes | SHA-256 |
|---|---|---|---|---|
| `sparklers.raw` | EVT 2.0, header without geometry | Gen3.0, 640x480 | 2,109,142 | `e84afbecdc07d2910ae846a4ae0ee246f5b9c97a53816c637d4f85c023d7c234` |
| `active_marker.raw` | EVT 3.0 | IMX636, 1280x720 | 108,741,633 | `700e9c6dd5df7ff4b36c4167b5fb6d64124e8050cbf5b52b9736e9a1368028c5` |

## What the fixtures contain

**`sparklers_100k.aedat4` is derived data**: not an iniVation recording, but Prophesee CD
events re-encoded as AEDAT 4.0 by dv-processing's `MonoCameraWriter` (event-only
configuration, camera name `sparklers_cc0_derived`, 640x480, the writer's default settings;
two writes give identical bytes). Its timestamps are the Prophesee sensor clock's
microseconds, kept unchanged so the file can be checked against the EVT 2.0 decode of the
same events; the AEDAT 4.0 format page describes Unix time. It was first written from
OpenEB 5.2.0's decode of `sparklers.raw`; `derive.py` writes it from the EVT adapter's decode
of the excerpt, and gets the same bytes.

The excerpts are byte prefixes of the recordings, cut at a word boundary, so their events
are a prefix of the recordings' events.

- `sparklers_100k.evt2.raw`: 100,000 CD events, t 913,716,224 to 913,728,417 µs (the sensor
  clock, not shifted), x 0 to 639, y 1 to 479, 68,073 OFF and 31,927 ON.
- `sparklers_100k.aedat4`: the same 100,000 events, in 10 packets, one event stream.
- `active_marker_head.evt3.raw`: 46,893 CD events, t 1,148 to 140,995 µs, all ON (the
  recording has ON events only).

The tests check these events against OpenEB 5.2.0's decode of the same recordings (its
default RAW-file path), recorded as SHA-256 digests in `tests/adapters/test_evt_open.py`.
