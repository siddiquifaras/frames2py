import numpy as np

import frames2py


def events(t, x):
    out = np.zeros(len(t), dtype=frames2py.EVENT_DTYPE)
    out["t"], out["x"] = t, x
    return out


engine = frames2py.Engine((3, 1), "time_surface", snapshot_interval_ms=0)
engine.ingest(events(t=[5_000_000, 5_000_001], x=[0, 1]))   # a source running for 5 s

engine.ingest(events(t=[10, 11], x=[1, 2]))                 # the source clock restarted near 0
snapshot = engine.snapshot()
print("frame:", snapshot.frame.tolist(), "watermark:", snapshot.meta.watermark)

engine.reset()                                              # the caller's job: a new time domain
engine.ingest(events(t=[10, 11], x=[1, 2]))
snapshot = engine.snapshot()
print("frame:", snapshot.frame.tolist(), "watermark:", snapshot.meta.watermark)
