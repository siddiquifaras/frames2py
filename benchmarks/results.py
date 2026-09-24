"""Result records: one JSON document per suite run.

A document holds the environment, the policy, the target, and one record per
cell. A cell record keeps every raw call time, so any summary can be recomputed.
Unsupported cells are recorded as such, with no numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "frames2py-benchmark/1"


def write(path: Path, document: dict[str, Any], *, overwrite: bool = False) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite a result")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n")


def read(path: Path) -> dict[str, Any]:
    document: Any = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} document")
    return document
