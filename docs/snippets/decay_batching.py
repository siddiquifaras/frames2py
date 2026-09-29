import numpy as np

import frames2py

# 1,000 events on one pixel, 10 µs apart, fed once as a single call and once as 10 calls.
events = np.zeros(1_000, dtype=frames2py.EVENT_DTYPE)
events["t"] = np.arange(1_000) * 10


def final_value(kernel, calls):
    acc = frames2py.Accumulator((1, 1), kernel)
    for part in np.array_split(events, calls):
        acc.accumulate(part)
    return float(acc.read()[0, 0])


for name, make in [("ExpDecay(0.9)", lambda: frames2py.ExpDecay(0.9)),
                   ("TimestampDecay(2000.0)", lambda: frames2py.TimestampDecay(2000.0))]:
    one, ten = final_value(make(), 1), final_value(make(), 10)
    print(f"{name:24} 1 call: {one:9.3f}   10 calls: {ten:9.3f}")
