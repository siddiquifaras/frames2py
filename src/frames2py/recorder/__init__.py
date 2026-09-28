"""Record events to an HDF5 file that ``frames2py.adapters.hdf5`` reads back.

::

    from frames2py import recorder

    with recorder.open("session.h5", sensor_size=(1280, 720)) as rec:
        for events in source:
            rec.write(events)       # on this thread; it may wait on compression and disk
            engine.ingest(events)   # the Engine never calls the recorder

    # read it back
    from frames2py.adapters import hdf5
    with hdf5.open("session.h5", group="events", sensor_size=(1280, 720)) as reader:
        ...

The file: ``group`` (``events`` by default) holds 1-D datasets ``t`` (uint64, µs), ``x``,
``y`` (uint16) and ``p`` (uint8), one element per event, in the order written; integer
attributes ``sensor_width``, ``sensor_height`` and ``frames2py_format_version`` (1) are on
the group. Events are recorded as given: nothing is sorted, clamped or repaired. Needs the
``frames2py[recorder]`` extra.
"""

from frames2py.recorder._writer import open

__all__ = ["open"]
