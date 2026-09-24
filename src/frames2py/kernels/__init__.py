"""Representation kernels and the Kernel protocol.

The five kernels Frames2Py ships are ``EventCount``, ``Polarity``, ``TimeSurface``,
``ExpDecay`` and ``TimestampDecay``. Any object implementing ``Kernel`` can be passed
to ``Accumulator`` or ``Engine`` in their place.
"""

from frames2py.kernels._builtin import EventCount, ExpDecay, Polarity, TimeSurface, TimestampDecay
from frames2py.kernels._protocol import Kernel, KernelState

__all__ = ["Kernel", "KernelState", "EventCount", "Polarity", "TimeSurface", "ExpDecay", "TimestampDecay"]
