import frames2py
from frames2py.adapters import evt

# The header of this EVT 2.0 file has no geometry, so sensor_size is required.
# An Accumulator's windowed kernels span everything since construction, so the totals
# below cover the whole file however the reader batches it.
with evt.open("sparklers_100k.evt2.raw", sensor_size=(640, 480), batch_size=10_000) as reader:
    acc = frames2py.Accumulator(reader.sensor_size, "polarity")
    for events in reader:
        acc.accumulate(events)

frame = acc.read()
print("sensor:", reader.sensor_size, "frame:", frame.shape)
print("events:", int(frame.sum()), "OFF:", int(frame[..., 0].sum()), "ON:", int(frame[..., 1].sum()))
print("watermark:", acc.watermark)
