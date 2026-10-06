"""The end-to-end notebook, ``examples/live_observation.ipynb``: running it and checking what is committed.

The notebook needs the ``notebook`` dependency group (nbclient, ipykernel, matplotlib); the core package and the
other groups don't include it. ``tests/test_notebook.py`` checks the committed file in every test run, and runs the
notebook only where the group is installed (the notebook CI workflow). Set ``FRAMES2PY_REQUIRE_NOTEBOOK=1`` to fail
instead of skipping when it isn't.

Re-run the notebook and rewrite its outputs in place, from the repository root:

    uv run --group notebook python -m tests.notebook
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "live_observation.ipynb"
REQUIRE_ENV = "FRAMES2PY_REQUIRE_NOTEBOOK"
CELL_TIMEOUT_S = 300
MAX_NOTEBOOK_BYTES = 600_000
MAX_IMAGE_BYTES = 120_000
# Anything that would tie the file to one machine or one account.
MACHINE_SPECIFIC = re.compile(r"/Users/|/home/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\\\|site-packages|\.venv")


def require_nbclient() -> ModuleType:
    try:
        return importlib.import_module("nbclient")
    except ImportError as exc:
        reason = f"nbclient is not installed (the notebook dependency group): {exc}"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{reason} ({REQUIRE_ENV}=1)", pytrace=False)
        pytest.skip(reason)


def execute(notebook: dict[str, Any], workdir: Path) -> Any:
    """*notebook* run top to bottom in a fresh kernel on this interpreter, with *workdir* as its working directory.

    The kernel is this ``sys.executable``, through a kernel spec written for the run, so that a ``python3`` kernel
    spec installed elsewhere on the machine can't take its place.
    """
    nbclient = require_nbclient()
    nbformat = importlib.import_module("nbformat")
    with tempfile.TemporaryDirectory() as jupyter:
        spec = Path(jupyter) / "kernels" / "python3"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3", "language": "python"}))
        saved = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = jupyter
        try:
            node = nbformat.reads(json.dumps(notebook), as_version=4)
            nbclient.NotebookClient(node, timeout=CELL_TIMEOUT_S, kernel_name="python3",
                                    resources={"metadata": {"path": str(workdir)}}).execute()
        finally:
            if saved is None:
                del os.environ["JUPYTER_PATH"]
            else:
                os.environ["JUPYTER_PATH"] = saved
    return node


def clean(notebook: dict[str, Any]) -> dict[str, Any]:
    """*notebook* without what an execution adds that belongs to one machine: timings and interpreter details."""
    notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python"}}
    for cell in notebook["cells"]:
        cell["metadata"] = {}
    return notebook


def problems(notebook: dict[str, Any], *, committed: bool) -> list[str]:
    """What is wrong with *notebook* as an executed example; empty if nothing.

    Both kinds: it was run top to bottom in one kernel, no cell raised, no code imports a private ``frames2py``
    module, and no output names a local path. A *committed* notebook also has no stderr output (a first run's
    one-off warnings, such as matplotlib building its font cache, are not kept), its images and its total size
    stay small, and it carries no per-machine metadata.
    """
    found = []
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    counts = [c.get("execution_count") for c in code]
    if counts != list(range(1, len(code) + 1)):
        found.append(f"execution counts {counts}: not one run, top to bottom")
    for number, cell in enumerate(code, 1):
        if re.search(r"frames2py(\.\w+)*\._(?!_)", "".join(cell["source"])):  # dunders are public
            found.append(f"code cell {number} uses a private frames2py name")
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                found.append(f"code cell {number} raised {output.get('ename')}: {output.get('evalue')}")
            if output["output_type"] == "stream" and output.get("name") == "stderr" and committed:
                found.append(f"code cell {number} wrote to stderr: {''.join(output['text'])[:200]!r}")
            data = output.get("data") or {}
            for mime, value in data.items():
                if committed and mime.startswith("image/") and len(base64.b64decode(value)) > MAX_IMAGE_BYTES:
                    found.append(f"code cell {number} has an image over {MAX_IMAGE_BYTES} bytes")
            text = json.dumps({**output, "data": {m: v for m, v in data.items() if not m.startswith("image/")}})
            for match in MACHINE_SPECIFIC.finditer(text):
                found.append(f"code cell {number} output names a local path: "
                             f"{text[max(0, match.start() - 40):match.end() + 40]!r}")
    if committed:
        size = len(json.dumps(notebook, indent=1))
        if size > MAX_NOTEBOOK_BYTES:
            found.append(f"{size} bytes, over {MAX_NOTEBOOK_BYTES}")
        if set(notebook["metadata"]) - {"kernelspec", "language_info"} or \
                set(notebook["metadata"].get("language_info", {})) - {"name"}:
            found.append(f"notebook metadata {notebook['metadata']} holds more than the kernel and language")
        if any(c.get("metadata") for c in notebook["cells"]):
            found.append("cell metadata is not empty (execution timings?)")
    return found


def load(path: Path = NOTEBOOK) -> dict[str, Any]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return notebook


def main() -> int:
    notebook = clean(execute(load(), NOTEBOOK.parent))
    found = problems(notebook, committed=True)
    for problem in found:
        print(f"ERROR: {problem}", file=sys.stderr)
    if found:
        print(f"{NOTEBOOK.relative_to(ROOT)} not rewritten", file=sys.stderr)
        return 1
    NOTEBOOK.write_text(importlib.import_module("nbformat").writes(notebook), encoding="utf-8")
    print(f"{NOTEBOOK.relative_to(ROOT)}: executed and rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
