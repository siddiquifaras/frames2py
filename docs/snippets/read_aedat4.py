# Needs frames2py[aedat4].
from frames2py.adapters import aedat4

with aedat4.open("sparklers_100k.aedat4", batch_size=10_000) as reader:
    total = 0
    first = None
    for events in reader:
        first = events[0] if first is None else first
        total += len(events)

print("sensor:", reader.sensor_size, "events:", total)
print("first event: t", int(first["t"]), "x", int(first["x"]), "y", int(first["y"]), "p", int(first["p"]))
