"""Watch an Engine while a producer thread feeds it synthetic events.

The producer runs on its own thread and calls ``engine.ingest()``; the viewer runs on the
main thread and reads ``engine.snapshot()`` at its own cadence. Neither waits for the other.

    uv run --extra viewer python examples/view_synthetic.py --kernel polarity
    uv run --extra viewer python examples/view_synthetic.py --kernel time_surface --seconds 5

Close the window (or press Esc) to stop, or pass ``--seconds``.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

import frames2py
from frames2py import viewer
from frames2py.kernels import Kernel
from frames2py.publish import Snapshot

KERNELS = ("event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay")


def make_kernel(name: str) -> str | Kernel:
    """A kernel name, or a configured instance for the kernels that take a parameter."""
    if name == "exp_decay":
        return frames2py.ExpDecay(0.9)
    if name == "timestamp_decay":
        return frames2py.TimestampDecay(20_000.0)
    return name


WIDTH, HEIGHT = 640, 480


def produce(engine: frames2py.Engine, rate: float, stop: threading.Event) -> None:
    """Events at *rate* per second: a bright disc circling the frame over a sparse background."""
    rng = np.random.default_rng(0)
    batch, t, start = 10_000, 0, time.monotonic()
    while not stop.is_set():
        events = np.empty(batch, dtype=frames2py.EVENT_DTYPE)
        events["t"] = t + np.arange(batch) * (1e6 / rate)
        angle = t / 1e6 * 2.0
        cx, cy = WIDTH / 2 + 180 * np.cos(angle), HEIGHT / 2 + 140 * np.sin(angle)
        on_disc = rng.random(batch) < 0.8
        x = np.where(on_disc, cx + rng.normal(0, 25, batch), rng.uniform(0, WIDTH, batch))
        y = np.where(on_disc, cy + rng.normal(0, 25, batch), rng.uniform(0, HEIGHT, batch))
        events["x"] = np.clip(x, 0, WIDTH - 1)
        events["y"] = np.clip(y, 0, HEIGHT - 1)
        events["p"] = x > cx  # the leading edge ON, the trailing edge OFF
        engine.ingest(events)
        t = int(events["t"][-1]) + 1
        ahead = t / 1e6 - (time.monotonic() - start)  # keep to the stated rate
        if ahead > 0:
            time.sleep(ahead)
    engine.stop()


class TimeUp(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="polarity")
    parser.add_argument("--rate", type=float, default=2e6, help="events per second (default 2e6)")
    parser.add_argument("--seconds", type=float, help="close the viewer after this long")
    args = parser.parse_args()

    engine = frames2py.Engine((WIDTH, HEIGHT), make_kernel(args.kernel), snapshot_interval_ms=16.0)
    stop = threading.Event()
    producer = threading.Thread(target=produce, args=(engine, args.rate, stop), daemon=True)
    producer.start()

    deadline = None if args.seconds is None else time.monotonic() + args.seconds

    def source() -> Snapshot | None:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeUp  # run() closes its window and lets the exception through
        return engine.snapshot()

    try:
        viewer.run(source, title=f"Frames2Py: synthetic, {args.kernel}")
    except TimeUp:
        pass
    finally:
        stop.set()
        producer.join()
    print(engine.stats)


if __name__ == "__main__":
    main()
