import numpy as np

import frames2py
from frames2py.adapters import evt

# Whole-file totals through an Engine with a windowed kernel: read the snapshot after every
# ingest() and after stop(), and add each new publication's window once (by its sequence).
# Any snapshot_interval_ms works; 0 here only makes every call publish, so the output below
# doesn't depend on timing. The totals are uint64, so they can't wrap as uint32 counts can.
totals, last = None, None


def collect(engine):
    global totals, last
    snapshot = engine.snapshot()
    if snapshot is not None and snapshot.meta.sequence != last:
        totals = snapshot.frame.astype(np.uint64) if totals is None else totals + snapshot.frame
        last = snapshot.meta.sequence


with evt.open("sparklers_100k.evt2.raw", sensor_size=(640, 480), batch_size=10_000) as reader:
    engine = frames2py.Engine(reader.sensor_size, "polarity", snapshot_interval_ms=0)
    for events in reader:
        engine.ingest(events)
        collect(engine)
    engine.stop()
    collect(engine)

print("publications:", last, "events:", int(totals.sum()))
print("OFF:", int(totals[..., 0].sum()), "ON:", int(totals[..., 1].sum()))
print("last window only:", int(engine.snapshot().frame.sum()), "events")
