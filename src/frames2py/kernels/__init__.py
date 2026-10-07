"""Representation kernels and the Kernel protocol.

The kernels Frames2Py ships are ``EventCount``, ``Polarity``, ``TimeSurface``,
``ExpDecay`` and ``TimestampDecay``, and the temporal kernels ``StackedHistogram`` and
``VoxelGrid``. Any object implementing ``Kernel`` can be passed to ``Accumulator`` or
``Engine`` in their place.
"""

from frames2py.kernels._builtin import EventCount, ExpDecay, Polarity, TimeSurface, TimestampDecay
from frames2py.kernels._protocol import Kernel
from frames2py.kernels._temporal import StackedHistogram, VoxelGrid

__all__ = [
    "Kernel",
    "EventCount",
    "Polarity",
    "TimeSurface",
    "ExpDecay",
    "TimestampDecay",
    "StackedHistogram",
    "VoxelGrid",
]
