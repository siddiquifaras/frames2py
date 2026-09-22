"""Accumulation kernels for event-to-frame conversion.

Available kernels:

- ``"event_count"`` -- counts events per pixel (NumPy)
- ``"polarity"`` -- 2-channel ON/OFF accumulation (NumPy)
- ``"time_surface"`` -- latest timestamp per pixel (NumPy)
- ``"exp_decay"`` -- exponential decay surface (NumPy)
- ``"native_event_count"`` -- C++ event count (pybind11, fallback to NumPy)
- ``"native_polarity"`` -- C++ polarity (pybind11, fallback to NumPy)
- ``"native_time_surface"`` -- C++ time surface (pybind11, fallback to NumPy)
"""

from frames2py.kernels.base import Kernel, get_kernel, register_kernel

# Trigger registration of built-in kernels.
import frames2py.kernels.numpy_kernels as _np  # noqa: F401
import frames2py.kernels.native_kernels as _nat  # noqa: F401

__all__ = ["Kernel", "get_kernel", "register_kernel"]
