# Needs frames2py[recorder] (and frames2py[hdf5] to read back; both install h5py and hdf5plugin).
import tempfile
from pathlib import Path

import numpy as np

import frames2py
from frames2py import recorder
from frames2py.adapters import evt, hdf5

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "session.h5"
    engine = frames2py.Engine((640, 480), "event_count")
    written = []
    with evt.open("sparklers_100k.evt2.raw", sensor_size=(640, 480)) as reader, \
            recorder.open(path, sensor_size=(640, 480)) as rec:
        for events in reader:
            rec.write(events)        # compression and file I/O on this thread
            engine.ingest(events)    # the Engine never calls the recorder
            written.append(events)
    engine.stop()

    with hdf5.open(path, group="events", sensor_size=(640, 480)) as reader:
        read_back = np.concatenate(list(reader))

original = np.concatenate(written)
print("events recorded:", len(read_back))
print("identical, in order:", all(np.array_equal(original[f], read_back[f]) for f in "txyp"))
