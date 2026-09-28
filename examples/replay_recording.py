"""Replay a recording at its recorded pace into an Engine, and watch it.

A producer thread reads the file with a Frames2Py adapter, paces the batches with
``frames2py.replay.paced`` and ingests them; the viewer runs on the main thread. The default
source is the committed EVT 3.0 excerpt of Prophesee's CC0 ``active_marker`` recording
(0.14 s), played at a tenth of its speed. For the whole recording (31.6 s):

    uv run python -m tests.recordings download active_marker.raw
    uv run --extra viewer python examples/replay_recording.py ~/.cache/frames2py/recordings/active_marker.raw --speed 1

    uv run --extra viewer python examples/replay_recording.py --kernel time_surface --seconds 4

The window stays open on the last snapshot when the replay ends; close it to stop.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import frames2py
from frames2py import viewer
from frames2py.kernels import Kernel
from frames2py.publish import Snapshot
from frames2py.adapters import evt
from frames2py.replay import paced

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "data" / "active_marker_head.evt3.raw"
KERNELS = ("event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay")


def make_kernel(name: str) -> str | Kernel:
    """A kernel name, or a configured instance for the kernels that take a parameter."""
    if name == "exp_decay":
        return frames2py.ExpDecay(0.9)
    if name == "timestamp_decay":
        return frames2py.TimestampDecay(10_000.0)
    return name


class TimeUp(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", type=Path, default=FIXTURE, help="an EVT file (default: the fixture)")
    parser.add_argument("--speed", type=float, default=0.1, help="replay speed relative to the recording")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="event_count")
    parser.add_argument("--seconds", type=float, help="close the viewer after this long")
    args = parser.parse_args()

    reader = evt.open(args.source, batch_size=2_000)
    if reader.sensor_size is None:
        raise SystemExit("the file's header has no geometry")
    engine = frames2py.Engine(reader.sensor_size, make_kernel(args.kernel), snapshot_interval_ms=16.0)
    stop = threading.Event()

    def produce() -> None:
        with reader:
            for events in paced(reader, speed=args.speed):
                if stop.is_set():
                    break
                engine.ingest(events)
        engine.stop()  # publishes the last window
        print(f"replay done: {engine.stats}")

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    deadline = None if args.seconds is None else time.monotonic() + args.seconds

    def source() -> Snapshot | None:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeUp
        return engine.snapshot()

    try:
        viewer.run(source, title=f"Frames2Py: {args.source.name} at {args.speed}x, {args.kernel}")
    except TimeUp:
        pass
    finally:
        stop.set()
        producer.join()


if __name__ == "__main__":
    main()
