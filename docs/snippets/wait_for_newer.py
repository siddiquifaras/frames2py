import threading

import numpy as np

import frames2py

engine = frames2py.Engine((4, 3), "event_count", snapshot_interval_ms=0)  # publish on every ingest()
events = np.zeros(2, dtype=frames2py.EVENT_DTYPE)
events["t"] = [10, 20]

print("nothing published, no wait:", engine.wait_for_newer(None, timeout=0))

producer = threading.Thread(target=engine.ingest, args=(events,))  # the producer's thread
producer.start()
snapshot = engine.wait_for_newer(None)  # blocks until there is a publication
producer.join()
print(snapshot.meta, int(snapshot.frame.sum()))

print("nothing newer within 50 ms:", engine.wait_for_newer(snapshot.meta.sequence, timeout=0.05))
