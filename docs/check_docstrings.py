"""Fail if griffe warns about any docstring in the package.

    uv run --only-group docs python docs/check_docstrings.py

The API reference renders docstrings with griffe's Google-style parser. A malformed section
(a parameter that isn't in the signature, two parameters on one line) renders wrongly but
doesn't always fail the site build, so every docstring in ``src/frames2py`` is parsed here
with the same parser and any warning is an error.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import griffe

SRC = Path(__file__).resolve().parents[1] / "src"


class Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def objects(obj: griffe.Object) -> list[griffe.Object]:
    found = [obj]
    for member in obj.members.values():
        if not member.is_alias:
            found.extend(objects(member))
    return found


def main() -> int:
    collect = Collect()
    logging.getLogger("griffe").addHandler(collect)
    package = griffe.load("frames2py", search_paths=[str(SRC)], docstring_parser="google")
    parsed = 0
    for obj in objects(package):
        if obj.docstring is not None:
            obj.docstring.parse("google")
            parsed += 1
    for message in collect.messages:
        print(f"ERROR: {message}", file=sys.stderr)
    print(f"{parsed} docstrings parsed, {len(collect.messages)} warnings")
    return 1 if collect.messages or parsed == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
