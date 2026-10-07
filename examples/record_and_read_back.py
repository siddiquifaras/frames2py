"""Record a real recording's events while an Engine ingests them, then read the recording back.

The source is the committed EVT 2.0 excerpt of Prophesee's CC0 ``sparklers`` recording, or
another EVT file you name. Each batch goes to ``recorder.write()`` and to
``engine.ingest()`` on the same thread; the Engine never calls the recorder. The recording is
then read back with ``frames2py.adapters.hdf5`` and compared event by event.

    uv run --extra recorder python examples/record_and_read_back.py
    uv run --extra recorder python examples/record_and_read_back.py --out "$TMPDIR/sparklers.h5"
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

import frames2py
from frames2py import recorder
from frames2py.adapters import evt, hdf5

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "data" / "sparklers_100k.evt2.raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", type=Path, default=FIXTURE, help="an EVT file (default: the fixture)")
    parser.add_argument("--sensor-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    parser.add_argument("--out", type=Path, help="the recording to write (default: a temporary directory)")
    parser.add_argument("--compression", choices=["blosc", "gzip", "none"], default="blosc")
    args = parser.parse_args()
    size = tuple(args.sensor_size)
    compression = None if args.compression == "none" else args.compression

    with tempfile.TemporaryDirectory() as scratch:
        out = args.out or Path(scratch) / "recording.h5"
        engine = frames2py.Engine(size, "event_count")
        original = []
        start = time.perf_counter()
        with evt.open(args.source, sensor_size=size) as reader, recorder.open(
            out, sensor_size=size, compression=compression, overwrite=args.out is not None
        ) as rec:
            for events in reader:
                rec.write(events)
                engine.ingest(events)
                original.append(events)
        engine.stop()
        elapsed = time.perf_counter() - start
        written = np.concatenate(original)

        with hdf5.open(out, group="events", sensor_size=size) as reader:
            read_back = np.concatenate(list(reader))
        identical = read_back.tobytes() == written.tobytes()

        print(f"source      {args.source.name}: {len(written):,} events in {len(original)} batches")
        print(f"recording   {out.name}, {compression or 'uncompressed'}, {out.stat().st_size:,} bytes")
        print(f"record+ingest {elapsed * 1e3:.1f} ms on this machine")
        print(f"read back   {len(read_back):,} events, identical to the source: {identical}")
        print(f"engine      {engine.stats}")
        if not identical:
            raise SystemExit("the recording does not match its source")


if __name__ == "__main__":
    main()
