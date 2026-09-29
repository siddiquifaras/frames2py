# EVT 2.0 and 3.0

`frames2py.adapters.evt.open(path, *, sensor_size=None, batch_size=None)` reads Prophesee
RAW files in EVT 2.0 or EVT 3.0 and yields their CD (contrast detection) events. The decoder
is Frames2Py's own, in NumPy; the `evt` extra installs nothing and exists so the install
command stays the same if that ever changes.

```python title="read_evt.py"
--8<-- "read_evt.py"
```

```text title="Output"
--8<-- "read_evt.out"
```

The file in this example is the first 100,000 CD events of Prophesee's CC0 `sparklers.raw`
sample, recorded with a 640x480 Gen3.0 sensor and committed to the repository as
`tests/data/sparklers_100k.evt2.raw`. Its header has no geometry line, so the example passes
`sensor_size`. When a header lacks geometry, take the sensor's resolution from the camera's
specification (for example 640x480 for Prophesee Gen3.0 sensors, 1280x720 for the Sony
IMX636); Frames2Py never guesses it. The [Adapters](adapters.md#reading-a-file) page explains
why the example counts with an Accumulator.

## The header

Header lines start with `%`. A `% end` line ends the header; otherwise it ends at the first
line that doesn't start with `%`.

- **Version.** `% evt 2.0` / `% evt 3.0`, or `% format EVT2` / `% format EVT3`, picks the
  decoder. A missing version, any other version (EVT 2.1 included), an `evt` and a `format`
  line that disagree, or any header keyword given two different values raise `ValueError`
  from `open()`.
- **Geometry** comes from `% geometry WxH` or the format line's `width=` and `height=`. A
  header without geometry (like the example's) needs `sensor_size`; one you pass must match.

Only CD events are decoded. Triggers and monitoring words are skipped.

## Timestamps

Timestamps are the sensor clock as recorded, in microseconds. They are not shifted to start
at the first TIME_HIGH, and nothing is sorted or clamped. The decoder unwraps the formats'
narrow time counters into continuous 64-bit microseconds, by these rules:

**EVT 2.0.** An event's time is the last TIME_HIGH (time bits 33..6) joined to its own 6 low
bits. The TIME_HIGH counter wraps after 2^34 µs (about 4.8 hours). A TIME_HIGH lower than the
previous one by at least `(2^28 - 1) × 64 − 10000` µs is read as a wrap, and 2^34 µs is added
from then on; any other backward step is kept as a backward step. This is OpenEB 5.2.0's rule.
CD events before the first TIME_HIGH are dropped, since their time is unknown.

**EVT 3.0.** An event's time is TIME_HIGH (bits 23..12) joined to TIME_LOW (bits 11..0), plus
2^24 µs (about 16.8 s) for each counter wrap.

- A TIME_HIGH step from 4095 to 0, or any backward step of more than 3840 units, is a wrap.
  Every other backward step is kept, and you see it as a backward jump in time.
- OpenEB 5.2.0's decoder reads any backward step of 2048 units or more as a wrap, while its
  own validator reports steps of 2048 to 3840 as violations. Frames2Py follows the validator:
  reading such a step as a wrap would turn a visible backward jump into a silent 16.8 s
  forward one. The format page says each TIME_HIGH value is sent 256 times, every 16 µs, so a
  genuine wrap is 4095 to 0; this rule still accepts one across which up to 254 TIME_HIGH
  values were lost.
- A TIME_HIGH that changes the value sets TIME_LOW to 0 until the next TIME_LOW. A TIME_LOW
  lower than the one before it is kept (the format page allows it across event sources).
- Decoding starts at the first TIME_HIGH. After it, events before the first EVT_ADDR_Y, and
  vector words before the first VECT_BASE_X, are dropped, as OpenEB does: their row or base
  is unknown.

## EVT 3.0 vectors and rows

- Vector words follow the format page: each VECT_12 or VECT_8 emits its set bits at the
  current base, then moves the base on by 12 or 8. No 12 + 12 + 8 grouping is assumed and
  nothing is dropped near the right edge (OpenEB's default decoder does both).
- Rows of type 0x1, which the format page reserves, emit nothing, and vector words inside
  them don't move the base (OpenEB's behaviour).
- A row at or beyond the sensor height is decoded and yielded; the Accumulator or Engine
  counts its events as out of bounds.
- A vector event whose x would exceed 65535, which only a corrupt stream produces, raises
  `ValueError`.

## Streaming

The file is read 1 MiB at a time. A read can end anywhere: a partial word carries into the
next read, so the events never depend on the read size. A partial word at the end of the
file is ignored, so a truncated file yields a prefix of the full file's events. With
`batch_size=None`, EVT 2.0 yields one array per 1 MiB read and EVT 3.0 one per 32,768 words
(64 KiB).

## Agreement with OpenEB

On the four real EVT recordings in the test registry, the output equals OpenEB 5.2.0's
default decode event for event. None of them contains a case where the rules above differ
from OpenEB's. The cases where they do differ, and the cases the format page leaves open
where Frames2Py follows OpenEB, are pinned by 40 crafted inputs with OpenEB 5.2.0's recorded
output for each (`tests/data/evt_golden/` in the repository).
