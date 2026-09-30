"""Per-process resource usage on macOS: ``proc_pid_rusage(RUSAGE_INFO_V6)`` through ctypes.

``sample()`` returns the structure's fields as integers, with the Mach-time fields converted
to nanoseconds. Elsewhere, or if the call fails, it returns ``None``: nothing is guessed.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Final

FIELDS: Final = tuple(
    """user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size
    phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time
    child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime
    diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance
    cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy
    cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time
    serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles
    billed_energy serviced_energy interval_max_phys_footprint runnable_time flags user_ptime
    system_ptime pinstructions pcycles energy_nj penergy_nj secure_time_in_system
    secure_ptime_in_system neural_footprint lifetime_max_neural_footprint
    interval_max_neural_footprint""".split()
)
"""``struct rusage_info_v6`` after its UUID, in declaration order (``<sys/resource.h>``)."""

TIME_FIELDS: Final = frozenset(
    {"user_time", "system_time", "user_ptime", "system_ptime", "runnable_time",
     "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
     "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
     "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time"}
)
"""Fields in Mach absolute time units, converted to nanoseconds by ``sample()``."""

RUSAGE_INFO_V6: Final = 6


class _V6(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in FIELDS] + [
        ("reserved", ctypes.c_uint64 * 9)]


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class _Reader:
    def __init__(self) -> None:
        self._libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        self._libc.proc_pid_rusage.restype = ctypes.c_int
        self._libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        timebase = _Timebase()
        if self._libc.mach_timebase_info(ctypes.byref(timebase)) != 0 or not timebase.denom:
            raise OSError("mach_timebase_info failed")
        self.numer, self.denom = int(timebase.numer), int(timebase.denom)

    def sample(self, pid: int) -> dict[str, int] | None:
        info = _V6()
        if self._libc.proc_pid_rusage(pid, RUSAGE_INFO_V6, ctypes.byref(info)) != 0:
            return None
        out = {f: int(getattr(info, f)) for f in FIELDS}
        for f in TIME_FIELDS:
            out[f] = out[f] * self.numer // self.denom
        return out


_READER: _Reader | None = None


def available() -> bool:
    """Whether ``sample()`` can return values on this platform."""
    return sys.platform == "darwin"


def sample(pid: int | None = None) -> dict[str, int] | None:
    """The V6 usage record of *pid* (this process by default), or ``None`` if unavailable."""
    global _READER
    if not available():
        return None
    if _READER is None:
        _READER = _Reader()
    return _READER.sample(os.getpid() if pid is None else pid)
