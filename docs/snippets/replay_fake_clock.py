import numpy as np

import frames2py
from frames2py.replay import paced


def batch(t):
    out = np.zeros(len(t), dtype=frames2py.EVENT_DTYPE)
    out["t"] = t
    return out


# A simulated clock and sleep, so the example runs instantly and prints exact times.
now = [0]
clock = lambda: now[0]                                   # ns
sleep = lambda seconds: now.__setitem__(0, now[0] + round(seconds * 1e9))

batches = [batch([1_000, 2_000]), batch([6_000]), batch([4_000]), batch([9_000])]  # µs
for events in paced(batches, speed=2.0, clock=clock, sleep=sleep):
    print(f"yielded at {now[0] / 1e6:5.1f} ms: t = {events['t'].tolist()}")
