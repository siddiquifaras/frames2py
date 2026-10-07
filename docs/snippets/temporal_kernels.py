import numpy as np

import frames2py

# Five ON events at pixel (0, 0) of a 2x1 sensor, at t = 0, 5, 12, 25 and 31 µs.
events = np.zeros(5, dtype=frames2py.EVENT_DTYPE)
events["t"] = [0, 5, 12, 25, 31]
events["p"] = 1

# Bins of 10 µs. The watermark is 31, in bin 3 ([30, 40)), which is still in progress.
for kernel in (frames2py.StackedHistogram(bins=3, bin_us=10), frames2py.VoxelGrid(bins=3, bin_us=10)):
    accumulator = frames2py.Accumulator((2, 1), kernel)
    accumulator.accumulate(events)
    frame = accumulator.read()
    print(kernel.name, frame.shape, frame.dtype)
    print(frame[..., 0, 0])  # pixel (0, 0): every channel and bin, or every knot
