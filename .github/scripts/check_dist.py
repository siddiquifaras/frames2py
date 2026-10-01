"""Audit the built wheel and sdist before anything installs them.

    python .github/scripts/check_dist.py dist/

Fails unless:
- there is exactly one wheel, tagged ``py3-none-any`` and purelib, and one sdist;
- the wheel's ``frames2py/`` files are exactly the tracked files under ``src/frames2py/``;
- the sdist holds only those files plus PKG-INFO, README.md, LICENSE, pyproject.toml and
  the ``pyproject.toml.orig`` that uv_build keeps beside its TOML 1.0 rewrite;
- no compiled or native file, no retired prototype module and no development material is
  in either;
- no documentation-site source, tooling, build output or logo asset is in either;
- the metadata carries the project's version, licence, Python floor, dependencies and extras,
  and no package of the ``docs`` dependency group, in the dependencies or in any extra.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = {"aedat4", "evt", "hdf5", "recorder", "viewer"}
RETIRED = ("frames2py/core/", "frames2py/bench/", "frames2py/consumers/", "frames2py/display/",
           "frames2py/kernels/base.py", "frames2py/kernels/numpy_kernels.py",
           "frames2py/kernels/native_kernels.py", "native/")
NATIVE_SUFFIXES = (".so", ".pyd", ".dylib", ".dll", ".c", ".cpp", ".h", ".pyx", ".o")
SDIST_EXTRA = {"PKG-INFO", "README.md", "LICENSE", "pyproject.toml", "pyproject.toml.orig"}
DOCS_DIRECTORIES = {"docs", "site", "node_modules"}
DOCS_FILES = {"mkdocs.yml", "package.json", "pnpm-lock.yaml", "frames2py-logo.svg", "frames2py-logo-dark.svg",
              "frames2py-mark.svg", "frames2py-mark-dark.svg", "frames2py-favicon.svg", "frames2py-social.png"}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def tracked_package_files() -> set[str]:
    listing = subprocess.run(["git", "ls-files", "src/frames2py"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.split()
    return {str(PurePosixPath(p).relative_to("src")) for p in listing}


def source_version() -> str:
    tree = ast.parse((ROOT / "src/frames2py/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__version__" for t in node.targets):
            return str(ast.literal_eval(node.value))
    raise SystemExit("no __version__ in src/frames2py/__init__.py")


def check_common(errors: list[str], kind: str, names: list[str]) -> None:
    for name in names:
        if name.endswith(NATIVE_SUFFIXES):
            fail(errors, f"{kind}: compiled or native file {name}")
        if "__pycache__" in name or name.endswith((".pyc", ".pyo")):
            fail(errors, f"{kind}: bytecode {name}")
        if any(part in name for part in RETIRED):
            fail(errors, f"{kind}: retired prototype path {name}")
        for dev in (".github/", "tests/", "benchmarks/", "verification/", ".venv", ".gitignore", ".DS_Store", "dist/",
                    "uv.lock"):
            if dev in name:
                fail(errors, f"{kind}: development material {name}")
        parts = PurePosixPath(name).parts
        if DOCS_DIRECTORIES & set(parts[:-1]) or parts[-1] in DOCS_FILES:
            fail(errors, f"{kind}: documentation-site material {name}")


def requirement_name(requirement: str) -> str:
    """The normalised project name of a requirement string (PEP 503)."""
    name = re.split(r"[\s;\[<>=!~@(]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def docs_packages() -> set[str]:
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text()).get("dependency-groups", {})
    return {requirement_name(r) for r in groups.get("docs", []) if isinstance(r, str)}


def check_metadata(errors: list[str], kind: str, text: str, version: str, requires_python: str,
                   docs: set[str]) -> None:
    meta = Parser().parsestr(text)
    if meta["Version"] != version:
        fail(errors, f"{kind}: Version {meta['Version']}, expected {version}")
    if meta["License-Expression"] != "MIT" or meta.get_all("License-File") != ["LICENSE"]:
        fail(errors, f"{kind}: licence metadata {meta['License-Expression']!r} {meta.get_all('License-File')!r}")
    if any(c.startswith("License ::") for c in meta.get_all("Classifier") or []):
        fail(errors, f"{kind}: legacy licence classifier present")
    if meta["Requires-Python"] != requires_python:
        fail(errors, f"{kind}: Requires-Python {meta['Requires-Python']}, expected {requires_python}")
    extras = set(meta.get_all("Provides-Extra") or [])
    if extras != EXTRAS:
        fail(errors, f"{kind}: extras {sorted(extras)}, expected {sorted(EXTRAS)}")
    for requirement in meta.get_all("Requires-Dist") or []:
        if requirement_name(requirement) in docs:
            fail(errors, f"{kind}: documentation tooling in Requires-Dist: {requirement}")
    unconditional = [r for r in meta.get_all("Requires-Dist") or [] if "extra ==" not in r]
    if unconditional != ["numpy>=2.4"]:
        fail(errors, f"{kind}: unconditional dependencies {unconditional}, expected ['numpy>=2.4']")


def main() -> int:
    dist = Path(sys.argv[1])
    errors: list[str] = []
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if source_version() != version:
        fail(errors, f"__version__ {source_version()} differs from pyproject.toml {version}")
    expected = tracked_package_files()
    docs = docs_packages()
    if not docs:
        fail(errors, "pyproject.toml has no docs dependency group to check against")

    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if [w.name for w in wheels] != [f"frames2py-{version}-py3-none-any.whl"]:
        fail(errors, f"wheels {[w.name for w in wheels]}, expected one py3-none-any wheel")
    if [s.name for s in sdists] != [f"frames2py-{version}.tar.gz"]:
        fail(errors, f"sdists {[s.name for s in sdists]}, expected one")
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        return 1

    info = f"frames2py-{version}.dist-info/"
    with zipfile.ZipFile(wheels[0]) as whl:
        names = [n for n in whl.namelist() if not n.endswith("/")]
        check_common(errors, "wheel", names)
        package = {n for n in names if n.startswith("frames2py/")}
        if package != expected:
            fail(errors, f"wheel package files differ from git: extra {sorted(package - expected)}, "
                         f"missing {sorted(expected - package)}")
        others = sorted(set(names) - package - {info + n for n in ("METADATA", "WHEEL", "RECORD", "licenses/LICENSE")})
        if others:
            fail(errors, f"wheel: unexpected files {others}")
        wheel_info = Parser().parsestr(whl.read(info + "WHEEL").decode())
        if wheel_info.get_all("Tag") != ["py3-none-any"] or wheel_info["Root-Is-Purelib"] != "true":
            fail(errors, f"wheel: Tag {wheel_info.get_all('Tag')} Root-Is-Purelib {wheel_info['Root-Is-Purelib']}")
        check_metadata(errors, "wheel", whl.read(info + "METADATA").decode(), version, project["requires-python"],
                       docs)
        print(f"{wheels[0].name}: {len(names)} files, {len(package)} in frames2py/, "
              f"generator {wheel_info['Generator']}")

    prefix = f"frames2py-{version}/"
    with tarfile.open(sdists[0]) as sdist:
        members = [m.name for m in sdist.getmembers() if m.isfile()]
        if any(not m.startswith(prefix) for m in members):
            fail(errors, "sdist: member outside the top-level directory")
        relative = {m[len(prefix):] for m in members}
        check_common(errors, "sdist", sorted(relative - {"pyproject.toml.orig"}))
        package = {m[len("src/"):] for m in relative if m.startswith("src/")}
        if package != expected:
            fail(errors, f"sdist package files differ from git: extra {sorted(package - expected)}, "
                         f"missing {sorted(expected - package)}")
        top_level = relative - {f"src/{p}" for p in package}
        if top_level != SDIST_EXTRA:
            fail(errors, f"sdist: top-level files {sorted(top_level)}, expected {sorted(SDIST_EXTRA)}")
        original = sdist.extractfile(prefix + "pyproject.toml.orig")
        if original is None or original.read() != (ROOT / "pyproject.toml").read_bytes():
            fail(errors, "sdist: pyproject.toml.orig is not the project's pyproject.toml")
        pkg_info = sdist.extractfile(prefix + "PKG-INFO")
        assert pkg_info is not None
        check_metadata(errors, "sdist", pkg_info.read().decode(), version, project["requires-python"], docs)
        print(f"{sdists[0].name}: {len(members)} files, {len(package)} under src/frames2py/")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not errors:
        print("artifact audit passed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
