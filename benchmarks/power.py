"""Machine power state for benchmark runs, and idle-sleep prevention on macOS.

A benchmark measured while the machine sleeps, or runs in macOS DarkWake (awake for
maintenance, graphics off, heavily throttled), produces numbers that look valid and aren't.
``hold_awake()`` guards a run against that:

- on macOS it refuses to start unless the machine is in full wake (the power-management
  root domain's "System Capabilities" has both the CPU and the graphics bit set; DarkWake
  clears graphics), then takes its own ``PreventUserIdleSystemSleep`` assertion through
  IOKit, checks that the assertion exists at level on, holds it until the run ends, and
  releases it. macOS also drops a process's assertions when the process exits, however it
  exits.
- on every platform it records how long the machine slept during the run: the growth of a
  clock that counts sleep minus one that doesn't (``_clock_pair``). Where no such pair
  exists, the figure is ``None``.
- elsewhere it takes no assertion and never refuses; it records what it can.

The capability bit values are xnu's ``kIOPMSystemCapabilityCPU`` (0x1) and
``kIOPMSystemCapabilityGraphics`` (0x2) from ``IOPMPrivate.h``; they are not in the public
SDK headers.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any, Final, Protocol

CAPABILITY_CPU: Final = 0x1
CAPABILITY_GRAPHICS: Final = 0x2
FULL_WAKE: Final = CAPABILITY_CPU | CAPABILITY_GRAPHICS
ASSERTION_TYPE: Final = "PreventUserIdleSystemSleep"
ASSERTION_LEVEL_ON: Final = 255
SLEEP_TOLERANCE_NS: Final = 1_000_000_000
"""Sleep during a run below this is treated as clock noise."""


class PowerStateError(RuntimeError):
    """The machine is not in a state a benchmark may run in."""


class Backend(Protocol):
    def capabilities(self) -> int | None: ...

    def create_assertion(self, name: str) -> int: ...

    def assertion_properties(self, assertion: int) -> dict[str, Any]: ...

    def release_assertion(self, assertion: int) -> None: ...


class MacBackend:
    """IOKit and CoreFoundation through ctypes."""

    _UTF8: Final = 0x08000100
    _SINT64: Final = 4

    def __init__(self) -> None:
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        io = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, u32]
        cf.CFRelease.argtypes = [vp]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFGetTypeID.restype = ctypes.c_ulong
        cf.CFGetTypeID.argtypes = [vp]
        cf.CFNumberGetTypeID.restype = ctypes.c_ulong
        cf.CFStringGetTypeID.restype = ctypes.c_ulong
        cf.CFNumberGetValue.restype = ctypes.c_bool
        cf.CFNumberGetValue.argtypes = [vp, ctypes.c_int, vp]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, u32]
        io.IOServiceMatching.restype = vp
        io.IOServiceMatching.argtypes = [ctypes.c_char_p]
        io.IOServiceGetMatchingService.restype = u32
        io.IOServiceGetMatchingService.argtypes = [u32, vp]
        io.IORegistryEntryCreateCFProperty.restype = vp
        io.IORegistryEntryCreateCFProperty.argtypes = [u32, vp, vp, u32]
        io.IOObjectRelease.argtypes = [u32]
        io.IOPMAssertionCreateWithName.restype = ctypes.c_int
        io.IOPMAssertionCreateWithName.argtypes = [vp, u32, vp, ctypes.POINTER(u32)]
        io.IOPMAssertionRelease.restype = ctypes.c_int
        io.IOPMAssertionRelease.argtypes = [u32]
        io.IOPMAssertionCopyProperties.restype = vp
        io.IOPMAssertionCopyProperties.argtypes = [u32]
        self._cf, self._io = cf, io

    def _string(self, text: str) -> int:
        ref = self._cf.CFStringCreateWithCString(None, text.encode(), self._UTF8)
        if not ref:
            raise PowerStateError(f"could not create a CFString for {text!r}")
        return int(ref)

    def _value(self, ref: int | None) -> Any:
        if not ref:
            return None
        kind = self._cf.CFGetTypeID(ref)
        if kind == self._cf.CFNumberGetTypeID():
            out = ctypes.c_int64()
            return out.value if self._cf.CFNumberGetValue(ref, self._SINT64, ctypes.byref(out)) else None
        if kind == self._cf.CFStringGetTypeID():
            buf = ctypes.create_string_buffer(256)
            return buf.value.decode() if self._cf.CFStringGetCString(ref, buf, 256, self._UTF8) else None
        return None

    def capabilities(self) -> int | None:
        matching = self._io.IOServiceMatching(b"IOPMrootDomain")
        service = self._io.IOServiceGetMatchingService(0, matching)  # consumes `matching`
        if not service:
            return None
        key = self._string("System Capabilities")
        try:
            ref = self._io.IORegistryEntryCreateCFProperty(service, key, None, 0)
            try:
                value = self._value(ref)
            finally:
                if ref:
                    self._cf.CFRelease(ref)
        finally:
            self._cf.CFRelease(key)
            self._io.IOObjectRelease(service)
        return int(value) if isinstance(value, int) else None

    def create_assertion(self, name: str) -> int:
        kind, label = self._string(ASSERTION_TYPE), self._string(name)
        assertion = ctypes.c_uint32(0)
        try:
            status = self._io.IOPMAssertionCreateWithName(kind, ASSERTION_LEVEL_ON, label, ctypes.byref(assertion))
        finally:
            self._cf.CFRelease(kind)
            self._cf.CFRelease(label)
        if status != 0 or not assertion.value:
            raise PowerStateError(f"IOPMAssertionCreateWithName failed with IOReturn {status:#x}")
        return assertion.value

    def assertion_properties(self, assertion: int) -> dict[str, Any]:
        props = self._io.IOPMAssertionCopyProperties(assertion)
        if not props:
            return {}
        try:
            out = {}
            for key in ("AssertType", "AssertLevel", "AssertName"):
                name = self._string(key)
                try:
                    out[key] = self._value(self._cf.CFDictionaryGetValue(props, name))
                finally:
                    self._cf.CFRelease(name)
            return out
        finally:
            self._cf.CFRelease(props)

    def release_assertion(self, assertion: int) -> None:
        status = self._io.IOPMAssertionRelease(assertion)
        if status != 0:
            raise PowerStateError(f"IOPMAssertionRelease failed with IOReturn {status:#x}")


def _clock_pair() -> tuple[int, int] | None:
    """A clock that counts sleep and one that doesn't: macOS ``CLOCK_MONOTONIC_RAW`` and
    ``CLOCK_UPTIME_RAW``, Linux ``CLOCK_BOOTTIME`` and ``CLOCK_MONOTONIC``."""
    for counts_sleep, excludes_sleep in (("CLOCK_MONOTONIC_RAW", "CLOCK_UPTIME_RAW"),
                                         ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC")):
        a, b = getattr(time, counts_sleep, None), getattr(time, excludes_sleep, None)
        if a is not None and b is not None:
            return time.clock_gettime_ns(a), time.clock_gettime_ns(b)
    return None


def _pmset_settings() -> dict[str, int]:
    try:
        text = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    settings = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("sleep", "displaysleep", "standby", "powernap") and parts[1].isdigit():
            settings[parts[0]] = int(parts[1])
    return settings


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def state(backend: Backend | None = None, platform: str = sys.platform) -> dict[str, Any]:
    """What can be said about the machine's power state now. Unknowns are ``None``."""
    if platform == "darwin":
        caps = (backend or MacBackend()).capabilities()
        return {
            "system_capabilities": caps,
            "full_wake": None if caps is None else (caps & FULL_WAKE) == FULL_WAKE,
            "pmset": _pmset_settings(),
        }
    if platform.startswith("linux"):
        governors = sorted({g for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
                            if (g := _read(p))})
        supplies = {p.split("/")[-2]: _read(p) for p in glob.glob("/sys/class/power_supply/*/online")}
        return {"cpu_governors": governors or None, "power_supply_online": supplies or None}
    return {}


@contextlib.contextmanager
def hold_awake(name: str = "frames2py benchmark", backend: Backend | None = None,
               platform: str = sys.platform) -> Iterator[dict[str, Any]]:
    """Guard a benchmark run; yields a record that is complete once the block exits.

    Raises ``PowerStateError`` before the block runs if, on macOS, the machine isn't in
    full wake or the assertion can't be taken and confirmed.
    """
    record: dict[str, Any] = {"platform": platform, "assertion": None}
    mac = platform == "darwin"
    if mac:
        backend = backend or MacBackend()
    record["start"] = state(backend, platform)
    if mac:
        if record["start"]["full_wake"] is not True:
            raise PowerStateError(
                "refusing to benchmark: the machine is not in full wake "
                f"(system capabilities {record['start']['system_capabilities']!r}; DarkWake clears graphics)"
            )
        assert backend is not None
        assertion = backend.create_assertion(name)
        props = backend.assertion_properties(assertion)
        if props.get("AssertType") != ASSERTION_TYPE or props.get("AssertLevel") != ASSERTION_LEVEL_ON:
            backend.release_assertion(assertion)
            raise PowerStateError(f"the sleep assertion could not be confirmed: {props!r}")
        record["assertion"] = {"type": ASSERTION_TYPE, "level": ASSERTION_LEVEL_ON, "name": props.get("AssertName")}
    clocks = _clock_pair()
    try:
        yield record
    finally:
        after = _clock_pair()
        record["slept_ns"] = (
            None if clocks is None or after is None
            else max(0, (after[0] - clocks[0]) - (after[1] - clocks[1]))
        )
        record["slept"] = None if record["slept_ns"] is None else record["slept_ns"] > SLEEP_TOLERANCE_NS
        try:
            record["end"] = state(backend, platform)
        finally:
            if mac:
                assert backend is not None
                backend.release_assertion(assertion)
                record["assertion"]["released"] = True


def slept_during(record: dict[str, Any] | None) -> bool:
    """Whether a run's power record shows the machine slept during it."""
    return bool(record and record.get("slept"))

