# Needs PyTorch, installed by you: Frames2Py neither depends on it nor imports it.
import numpy as np
import torch

import frames2py

events = np.zeros(6, dtype=frames2py.EVENT_DTYPE)
events["t"] = [3, 8, 14, 21, 27, 33]
events["x"] = [0, 1, 2, 3, 1, 2]
events["y"] = [0, 0, 1, 2, 1, 2]
events["p"] = [1, 0, 1, 1, 0, 1]

# A 2-D frame: counts, (H, W) uint32.
engine = frames2py.Engine((4, 3), "event_count", snapshot_interval_ms=0)
engine.ingest(events)
snapshot = engine.snapshot()

counts = torch.from_numpy(snapshot.copy())  # copy first: the tensor shares the copy's memory, which is yours
print("counts:", tuple(counts.shape), counts.dtype)
counts = counts.to(torch.int64)  # torch has few uint32 ops; int64 holds every uint32 exactly
counts += 1  # yours to change: the published frame doesn't see it
print("published frame unchanged:", int(snapshot.frame.sum()) == len(events))

# The voxel grid: (bins, H, W) float32, time-first. As NCHW, the bins are the channels.
engine = frames2py.Engine((4, 3), frames2py.VoxelGrid(bins=3, bin_us=10), snapshot_interval_ms=0)
engine.ingest(events)
voxels = torch.from_numpy(engine.snapshot().copy())
batch = voxels.unsqueeze(0)  # (1, bins, H, W), a view: nothing transposed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1).to(device)
with torch.no_grad():
    features = model(batch.to(device))  # .to() copies when the device changes
print("voxels:", tuple(voxels.shape), voxels.dtype, "-> features:", tuple(features.shape))

# The histogram: (2, bins, H, W) uint32. Merging polarity and time into 2 * bins channels is a
# reshape, polarity-major: channel p * bins + b is frame[p, b].
engine = frames2py.Engine((4, 3), frames2py.StackedHistogram(bins=3, bin_us=10), snapshot_interval_ms=0)
engine.ingest(events)
histogram = torch.from_numpy(engine.snapshot().copy())
channels = histogram.reshape(1, 2 * 3, 3, 4).to(torch.float32)  # exact while every count is below 2**24
print("histogram:", tuple(histogram.shape), histogram.dtype, "-> channels:", tuple(channels.shape), channels.dtype)

# Without an allocation per read: one buffer of your own, refilled in place. The tensor is that
# buffer, so each refill changes it; clone() what you need to keep.
engine = frames2py.Engine((4, 3), "event_count", snapshot_interval_ms=0)
buffer = np.zeros((3, 4), dtype=np.uint32)
frame = torch.from_numpy(buffer)
for batch_of_events in (events[:2], events[2:]):
    engine.ingest(batch_of_events)
    engine.snapshot().copy(out=buffer)
    print("window total:", int(frame.sum()))
