import numpy as np

import frames2py
from frames2py.replay import windows


def batch(*t):
    out = np.zeros(len(t), dtype=frames2py.EVENT_DTYPE)
    out["t"] = t
    out["p"] = 1
    return out


# Events at pixel (0, 0), split into batches anywhere; nothing happens between t = 25 and 61.
batches = [batch(3, 12), batch(15, 25), batch(61, 64)]
for t_us, frame in windows(batches, (2, 1), "event_count", every_us=10):
    print(f"frame at {t_us} µs: count {frame[0, 0]}")
