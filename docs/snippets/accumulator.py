import numpy as np

import frames2py


def events(t, x, y, p=1):
    out = np.zeros(len(t), dtype=frames2py.EVENT_DTYPE)
    out["t"], out["x"], out["y"], out["p"] = t, x, y, p
    return out


acc = frames2py.Accumulator((4, 3), "time_surface")  # 4 columns, 3 rows
print("watermark before any event:", acc.watermark)

# Out of order, and one event outside the 4x3 sensor (x = 7).
acc.accumulate(events(t=[30, 10, 20, 99], x=[0, 1, 0, 7], y=[0, 0, 0, 2]))

print(acc.read())                        # a copy; reading changes nothing
print("watermark:", acc.watermark)       # the largest in-bounds timestamp, not the last one
print("out of bounds:", acc.events_out_of_bounds)

acc.reset()
print("after reset:", acc.watermark, acc.events_out_of_bounds, int(acc.read().sum()))
