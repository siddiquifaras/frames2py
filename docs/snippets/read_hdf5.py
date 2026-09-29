# Needs frames2py[hdf5].
from frames2py.adapters import hdf5

# DSEC's layout: events/{t,x,y,p}, with t relative to the scalar dataset /t_offset.
with hdf5.open("sparklers_100k.h5", group="events", t_offset="/t_offset", sensor_size=(640, 480),
               batch_size=25_000) as reader:
    sizes = [len(events) for events in reader]
    print("arrays:", sizes)
