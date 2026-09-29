import numpy as np

import frames2py
from frames2py import viewer

engine = frames2py.Engine((4, 1), "polarity")
events = np.zeros(4, dtype=frames2py.EVENT_DTYPE)
events["t"] = [1, 2, 3, 4]
events["x"] = [0, 1, 2, 2]
events["p"] = [0, 1, 0, 1]   # pixel 0 OFF, pixel 1 ON, pixel 2 both, pixel 3 nothing
engine.ingest(events)

image = viewer.render(engine.snapshot())   # NumPy only: no window, no pyglet
print(image.shape, image.dtype)
print(image[0].tolist())
