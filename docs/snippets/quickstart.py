import numpy as np

import frames2py

# 10,000 synthetic events on a 640x480 sensor, one every microsecond.
rng = np.random.default_rng(seed=0)
events = np.zeros(10_000, dtype=frames2py.EVENT_DTYPE)
events["t"] = np.arange(10_000)              # timestamps, µs
events["x"] = rng.integers(0, 640, 10_000)   # column
events["y"] = rng.integers(0, 480, 10_000)   # row
events["p"] = rng.integers(0, 2, 10_000)     # polarity: 0 is OFF, anything else ON

engine = frames2py.Engine((640, 480), "event_count")  # sensor_size is (width, height)
engine.ingest(events)                                 # the first ingest() always publishes

snapshot = engine.snapshot()  # the latest publication: shared, read-only
print(snapshot.frame.shape, snapshot.frame.dtype)
print("events counted:", int(snapshot.frame.sum()))
print("watermark:", snapshot.meta.watermark, "sequence:", snapshot.meta.sequence)
print("ingested:", engine.stats.events_ingested, "out of bounds:", engine.stats.events_out_of_bounds)
