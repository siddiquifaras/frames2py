"""The environment a result was measured in.

Anything that can't be determined is recorded as ``None``, never guessed.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import platform
import resource
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import numpy as np

import frames2py

_REPO = Path(__file__).resolve().parent.parent


def _run(*args: str, cwd: Path = _REPO, strip: bool = True) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() if strip else result.stdout


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def git_state(repo: Path = _REPO) -> dict[str, Any]:
    """The commit and working-tree state of *repo*.

    ``tracked_changes`` covers modified, staged and deleted tracked files.
    ``untracked_files`` lists every untracked file git doesn't ignore, with its
    SHA-256, so code that exists only in the working tree is visible and
    identifiable. Each field is ``None`` if git couldn't report it.
    """
    status = _run("git", "status", "--porcelain", "--untracked-files=no", cwd=repo)
    others = _run("git", "ls-files", "--others", "--exclude-standard", "-z", cwd=repo, strip=False)
    untracked = (
        None
        if others is None
        else [
            {"path": name, "sha256": _sha256(repo / name)}
            for name in sorted(others.split("\0"))
            if name
        ]
    )
    return {
        "commit": _run("git", "rev-parse", "HEAD", cwd=repo),
        "tracked_changes": None if status is None else bool(status),
        "untracked_files": untracked,
    }


def working_tree_clean(env: dict[str, Any]) -> bool | None:
    """Whether the recorded commit alone identifies the code that ran.

    ``None`` if the record can't say: git was unavailable, or the record predates
    untracked-file tracking.
    """
    tracked = env.get("tracked_changes")
    untracked = env.get("untracked_files")
    if tracked is True or untracked:
        return False
    if tracked is None or untracked is None or env.get("commit") is None:
        return None
    return True


def _chip() -> str | None:
    if sys.platform == "darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = _run("sysctl", "-n", "hw.memsize")
        return int(value) if value and value.isdigit() else None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _power() -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"source": None, "low_power_mode": None}
    batt = _run("pmset", "-g", "batt") or ""
    first = batt.splitlines()[0] if batt else ""
    source = first.split("'")[1] if first.count("'") >= 2 else None
    low_power: bool | None = None
    for line in (_run("pmset", "-g") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "lowpowermode" and parts[1] in ("0", "1"):
            low_power = parts[1] == "1"
    return {"source": source, "low_power_mode": low_power}


def _gil_enabled() -> bool | None:
    check = getattr(sys, "_is_gil_enabled", None)
    return bool(check()) if check is not None else True


def capture() -> dict[str, Any]:
    """Record the machine, software and repository state."""
    return {
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        **git_state(),
        "machine": platform.machine(),
        "chip": _chip(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "free_threaded_build": sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
        "gil_enabled": _gil_enabled(),
        "numpy": np.__version__,
        "frames2py": frames2py.__version__,
        "power": _power(),
    }


def max_rss_bytes(children: bool = False) -> int:
    """Peak resident set size of this process, or of its largest finished child."""
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    peak = resource.getrusage(who).ru_maxrss
    # macOS reports bytes, Linux kibibytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024
