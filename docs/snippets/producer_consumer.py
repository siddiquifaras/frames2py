import threading

import numpy as np

import frames2py

WIDTH, HEIGHT, BATCH, BATCHES = 640, 480, 10_000, 200
engine = frames2py.Engine((WIDTH, HEIGHT), "event_count", snapshot_interval_ms=1.0)
done = threading.Event()
seen = []


def produce():
    rng = np.random.default_rng(seed=1)
    for b in range(BATCHES):
        events = np.zeros(BATCH, dtype=frames2py.EVENT_DTYPE)
        events["t"] = np.arange(b * BATCH, (b + 1) * BATCH)
        events["x"] = rng.integers(0, WIDTH, BATCH)
        events["y"] = rng.integers(0, HEIGHT, BATCH)
        engine.ingest(events)  # never waits for the consumer
    engine.stop()              # publishes the events accumulated since the last publication
    done.set()


def consume():
    last = None
    while True:
        finished = done.is_set()
        snapshot = engine.wait_for_newer(last, timeout=0.1)  # blocks until a newer publication
        if snapshot is not None:                              # None: nothing newer within 0.1 s
            last = snapshot.meta.sequence
            seen.append(last)
        if finished:
            break


producer = threading.Thread(target=produce)
consumer = threading.Thread(target=consume)
consumer.start()
producer.start()
producer.join()
consumer.join()

final = engine.snapshot()
print("consumer saw new publications:", len(seen) > 0)
print("sequences only increase:", all(a < b for a, b in zip(seen, seen[1:])))
print("final watermark:", final.meta.watermark)
print("events ingested:", engine.stats.events_ingested)
