import numpy as np

import frames2py

engine = frames2py.Engine((4, 3), "event_count", snapshot_interval_ms=0)  # publish on every ingest()
print("before the first publication:", engine.snapshot())

events = np.zeros(3, dtype=frames2py.EVENT_DTYPE)
events["t"] = [1, 2, 3]
engine.ingest(events)

snapshot = engine.snapshot()
print(snapshot.meta)
print("same object for every reader:", engine.snapshot() is snapshot)
print("writeable:", snapshot.frame.flags.writeable)

try:
    snapshot.frame[0, 0] = 7
except ValueError as error:
    print("ValueError:", error)

mine = snapshot.copy()                 # an independent, writable copy
mine[0, 0] = 7
print("copy changed, snapshot not:", int(mine[0, 0]), int(snapshot.frame[0, 0]))

buffer = np.empty_like(snapshot.frame)  # or fill an array you own
print("copy(out=) returns out:", snapshot.copy(out=buffer) is buffer)
