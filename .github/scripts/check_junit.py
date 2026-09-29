"""Check a pytest JUnit XML file against what a CI job claims about its run.

The suite's conftest records the runtime as test-suite properties from inside the pytest
process (Python, NumPy, frames2py's import path, and the GIL state at start and end).
This script fails the job when those, or the named tests, don't match the job's claim.

    python .github/scripts/check_junit.py junit.xml --installed --numpy 2.4.1 \\
        --gil disabled --ran tests.contract.test_concurrency::test_stats_read_in_parallel_with_ingest
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit", type=Path)
    parser.add_argument("--installed", action="store_true",
                        help="frames2py must have been imported from site-packages, not the source tree")
    parser.add_argument("--numpy", help="the exact NumPy version the tests must have imported")
    parser.add_argument("--gil", choices=["enabled", "disabled"],
                        help="the GIL state required at the start and at the end of the session")
    parser.add_argument("--ran", action="append", default=[], metavar="CLASSNAME::NAME",
                        help="a test that must have run and passed, not been skipped")
    parser.add_argument("--forbid-skip", action="append", default=[], metavar="TEXT",
                        help="fail if any test was skipped with a reason containing TEXT")
    args = parser.parse_args()

    root = ET.parse(args.junit).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    properties = {p.get("name", ""): p.get("value", "") for s in suites for p in s.iter("property")}
    errors: list[str] = []
    for name in ("python", "numpy", "frames2py_file", "gil_enabled_at_start", "gil_enabled_at_end"):
        if name not in properties:
            errors.append(f"property {name!r} missing: was the suite's conftest loaded?")
    for suite in suites:
        print(f"{suite.get('name')}: tests={suite.get('tests')} failures={suite.get('failures')} "
              f"errors={suite.get('errors')} skipped={suite.get('skipped')} time={suite.get('time')}")
    for name, value in sorted(properties.items()):
        print(f"  {name} = {value}")

    location = Path(properties.get("frames2py_file") or "")
    if args.installed:
        source = Path.cwd().resolve() / "src"
        if "site-packages" not in location.parts:
            errors.append(f"frames2py was imported from {location}, not from site-packages")
        if source in location.resolve().parents:
            errors.append(f"frames2py was imported from the source tree {source}")
    if args.numpy and properties.get("numpy") != args.numpy:
        errors.append(f"NumPy {properties.get('numpy')} was imported, expected {args.numpy}")
    if args.gil:
        expected = str(args.gil == "enabled")
        for name in ("gil_enabled_at_start", "gil_enabled_at_end"):
            if properties.get(name) != expected:
                errors.append(f"{name} = {properties.get(name)}, expected {expected}")

    cases = {f"{c.get('classname')}::{c.get('name')}": c for s in suites for c in s.iter("testcase")}
    for test in args.ran:
        case = cases.get(test)
        if case is None:
            errors.append(f"{test} is not in the report")
            continue
        outcome = [child.tag for child in case if child.tag in ("skipped", "failure", "error")]
        if outcome:
            errors.append(f"{test} did not pass: {outcome[0]}")
        else:
            print(f"  ran and passed: {test}")

    for text in args.forbid_skip:
        skipped = [test for test, case in cases.items()
                   if any(child.tag == "skipped" and text in (child.get("message") or "") for child in case)]
        if skipped:
            errors.append(f"{len(skipped)} tests skipped with {text!r}, e.g. {skipped[0]}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
