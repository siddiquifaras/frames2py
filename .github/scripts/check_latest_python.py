"""Check that ci.yml's ``latest`` job tests the newest CPython build, and nothing else.

The job gets its interpreter from actions/setup-python with ``check-latest``, which installs
builds from actions/python-versions. These checks fail the job if that interpreter is not the
newest stable build the manifest lists for the runner, or if the virtual environment, or the
pytest run inside it, used any other interpreter.

    python check_latest_python.py newest --requested 3.14t
    python check_latest_python.py venv --base /path/to/setup-python/bin/python
    python check_latest_python.py junit junit.xml --version 3.14.8
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import sysconfig
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

MANIFEST = "https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json"
ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}


def free_threaded() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def linux_release() -> str | None:
    for line in Path("/etc/os-release").read_text().splitlines():
        key, _, value = line.partition("=")
        if key == "VERSION_ID":
            return value.strip('"')
    return None


def newest_in_manifest(minor: str, threaded: bool) -> str:
    """The newest stable version of *minor* with a build for this runner, as setup-python picks."""
    with urllib.request.urlopen(MANIFEST, timeout=60) as response:
        manifest: list[dict[str, Any]] = json.load(response)
    system = {"Linux": "linux", "Darwin": "darwin"}[platform.system()]
    arch = ARCHES[platform.machine().lower()] + ("-freethreaded" if threaded else "")
    release = linux_release() if system == "linux" else None

    def key(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    candidates: list[str] = [
        str(entry["version"]) for entry in manifest
        if entry["stable"] and entry["version"].startswith(minor + ".")
        and any(f["platform"] == system and f["arch"] == arch
                and (release is None or f.get("platform_version") == release) for f in entry["files"])
    ]
    if not candidates:
        raise SystemExit(f"no stable {minor} build for {system} {arch} {release or ''} in the manifest")
    return max(candidates, key=key)


def newest(requested: str) -> list[str]:
    threaded = requested.endswith("t")
    minor = requested.removesuffix("t")
    version = platform.python_version()
    expected = newest_in_manifest(minor, threaded)
    print(f"interpreter {sys.executable}: CPython {version}, free-threaded {free_threaded()}")
    print(f"newest stable {minor}{'t' if threaded else ''} build in actions/python-versions for this runner: {expected}")
    errors = []
    if platform.python_implementation() != "CPython":
        errors.append(f"{platform.python_implementation()} is not CPython")
    if free_threaded() != threaded:
        errors.append(f"free-threaded build is {free_threaded()}, expected {threaded}")
    if version != expected:
        errors.append(f"CPython {version} is not the newest available build, {expected}")
    return errors


def venv(base: Path) -> list[str]:
    script = "import sys, platform; print(sys.prefix); print(platform.python_version())"
    base_prefix, base_version = subprocess.run(
        [str(base), "-c", script], check=True, capture_output=True, text=True).stdout.split()
    print(f"virtual environment {sys.prefix}, on {sys.base_prefix} (CPython {platform.python_version()})")
    print(f"setup-python's interpreter: {base} (prefix {base_prefix}, CPython {base_version})")
    errors = []
    if sys.prefix == sys.base_prefix:
        errors.append("not running in a virtual environment")
    if os.path.realpath(sys.base_prefix) != os.path.realpath(base_prefix):
        errors.append(f"the environment's interpreter {sys.base_prefix} is not setup-python's {base_prefix}")
    if platform.python_version() != base_version:
        errors.append(f"CPython {platform.python_version()} in the environment, {base_version} provided")
    return errors


def junit(report: Path, version: str) -> list[str]:
    root = ET.parse(report).getroot()
    found = {p.get("value") for p in root.iter("property") if p.get("name") == "python"}
    print(f"{report}: the test process ran CPython {', '.join(sorted(v or '' for v in found))}")
    return [] if found == {version} else [f"the tests ran on {found}, not CPython {version}"]


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    newest_parser = commands.add_parser("newest", help="this interpreter is the newest build of the requested version")
    newest_parser.add_argument("--requested", required=True, help="the job's python-version, such as 3.14t")
    venv_parser = commands.add_parser("venv", help="this environment runs on the given base interpreter")
    venv_parser.add_argument("--base", type=Path, required=True)
    junit_parser = commands.add_parser("junit", help="the test process ran this exact version")
    junit_parser.add_argument("report", type=Path)
    junit_parser.add_argument("--version", required=True)
    args = parser.parse_args()

    if args.command == "newest":
        errors = newest(args.requested)
    elif args.command == "venv":
        errors = venv(args.base)
    else:
        errors = junit(args.report, args.version)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
