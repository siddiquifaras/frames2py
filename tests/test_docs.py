"""The documentation tells the truth about the installed package.

Every Python block in README.md and docs/content/ is one of three kinds:

- an included example: its only line is ``--8<-- "name.py"``, pulling in docs/snippets/name.py,
  which runs as its own process and must print exactly docs/snippets/name.out;
- a sketch: its first line starts with ``# Sketch (not runnable)``; shown, never run;
- an inline example: anything else. A page's inline blocks run in order in one namespace, and
  a block followed by a ``text`` block titled ``Output`` must print exactly that text.

The API reference must document every name in every public module's ``__all__`` and nothing
that isn't public (apart from the classes public functions return).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import pkgutil
import re
import shutil
import subprocess
import sys
import typing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import frames2py
from tests.adapters.backends import require_backend
from tests.temporal_cases import CASES, RESET, Case
from tests.temporal_cases import SENSOR as TEMPORAL_SENSOR
from tests.temporal_oracle import voxel_value
from tests.torch_support import require_torch

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "docs" / "content"
SNIPPETS = ROOT / "docs" / "snippets"
API_PAGES = CONTENT / "reference" / "api"
FIXTURES = ROOT / "tests" / "data"
PAGES_URL = "https://siddiquifaras.github.io/frames2py/"
PAGES = [ROOT / "README.md", *sorted(CONTENT.rglob("*.md"))]
SKETCH = "# Sketch (not runnable)"
INCLUDE = re.compile(r'^--8<-- "([^"]+)"$')
BACKENDS = {"aedat4": ("dv_processing",), "hdf5": ("h5py", "hdf5plugin"),
            "recorder": ("h5py", "hdf5plugin"), "viewer": ("pyglet",)}
NAMESPACES = {"frames2py.adapters"}  # packages that group public modules and export nothing


@dataclass(frozen=True)
class Block:
    page: Path
    line: int
    language: str
    info: str
    body: str

    @property
    def where(self) -> str:
        return f"{self.page.relative_to(ROOT)}:{self.line}"

    @property
    def include(self) -> str | None:
        match = INCLUDE.match(self.body.strip())
        return match.group(1) if match else None

    @property
    def is_output(self) -> bool:
        return self.language == "text" and 'title="Output"' in self.info


def blocks(page: Path) -> list[Block]:
    """The fenced code blocks of a Markdown page, indented ones (inside lists) included."""
    found, lines, i = [], page.read_text().splitlines(), 0
    while i < len(lines):
        opening = re.match(r"^(\s*)(`{3,})(\S*)(.*)$", lines[i])
        if not opening:
            i += 1
            continue
        indent, fence, language, info = opening.groups()
        body, j = [], i + 1
        while j < len(lines) and not re.match(rf"^\s*{fence}\s*$", lines[j]):
            body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j])
            j += 1
        found.append(Block(page, i + 1, language, info.strip(), "\n".join(body) + "\n"))
        i = j + 1
    return found


def python_blocks() -> list[Block]:
    return [b for page in PAGES for b in blocks(page) if b.language in ("python", "py", "pycon")]


def following_output(block: Block) -> Block | None:
    page_blocks = blocks(block.page)
    index = page_blocks.index(block)
    if index + 1 < len(page_blocks) and page_blocks[index + 1].is_output:
        return page_blocks[index + 1]
    return None


def fixture_directory(tmp_path: Path) -> Path:
    for fixture in FIXTURES.glob("*"):
        if fixture.is_file() and fixture.suffix in (".raw", ".aedat4", ".h5"):
            shutil.copy(fixture, tmp_path / fixture.name)
    return tmp_path


def run_python(code: str, cwd: Path) -> str:
    done = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stderr
    return done.stdout


# ---------------------------------------------------------------- block classification


def test_every_python_block_is_an_example_an_inline_example_or_a_labelled_sketch() -> None:
    for block in python_blocks():
        if block.include is not None:
            assert (SNIPPETS / block.include).is_file(), f"{block.where}: no docs/snippets/{block.include}"
            assert block.include.endswith(".py"), f"{block.where}: includes {block.include}, not a .py snippet"
        elif block.body.lstrip().startswith("#") and "not runnable" in block.body.splitlines()[0].lower():
            assert block.body.startswith(SKETCH), f"{block.where}: a sketch's first line must start {SKETCH!r}"


def test_every_snippet_is_shown_with_its_output() -> None:
    included, outputs = set(), set()
    for page in PAGES:
        page_blocks = blocks(page)
        for block in page_blocks:
            if block.include and block.include.endswith(".py"):
                included.add(block.include)
            if block.is_output and block.include:
                outputs.add(block.include)
    snippets = {p.name for p in SNIPPETS.glob("*.py")}
    assert snippets == included, f"snippets never shown: {snippets - included}"
    assert {p.name for p in SNIPPETS.glob("*.out")} == {s[:-3] + ".out" for s in snippets}
    assert outputs == {s[:-3] + ".out" for s in snippets}, f"outputs never shown: {snippets}"


# ---------------------------------------------------------------- running the examples


@pytest.mark.parametrize("snippet", sorted(p.name for p in SNIPPETS.glob("*.py")))
def test_snippet_prints_its_documented_output(snippet: str, tmp_path: Path) -> None:
    source = (SNIPPETS / snippet).read_text()
    header = source.split("\n\n", 1)[0]
    for extra in re.findall(r"frames2py\[(\w+)\]", header):
        require_backend(*BACKENDS.get(extra, ()))
    if "Needs PyTorch" in header:
        require_torch()
    expected = (SNIPPETS / snippet.replace(".py", ".out")).read_text()
    assert run_python(source, fixture_directory(tmp_path)) == expected


MARKER = "=== frames2py docs block {} ==="


def inline_pages() -> list[Path]:
    return [page for page in PAGES if any(b.page == page for b in inline_blocks())]


def inline_blocks() -> list[Block]:
    return [b for b in python_blocks() if b.include is None and not b.body.startswith(SKETCH)]


@pytest.mark.parametrize("page", inline_pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_inline_examples_run_in_page_order_and_print_their_output(page: Path, tmp_path: Path) -> None:
    page_blocks = [b for b in inline_blocks() if b.page == page]
    program = ["import sys", "namespace = {'__name__': '__main__'}"]
    for i, block in enumerate(page_blocks):
        program.append(f"exec(compile({block.body!r}, {block.where!r}, 'exec'), namespace)")
        program.append(f"print({MARKER.format(i)!r}); sys.stdout.flush()")
    printed = run_python("\n".join(program), fixture_directory(tmp_path))
    outputs = re.split(r"=== frames2py docs block \d+ ===\n", printed)[:-1]
    assert len(outputs) == len(page_blocks)
    for block, output in zip(page_blocks, outputs):
        expected = following_output(block)
        if expected is not None:
            assert output == expected.body, f"{block.where}: printed {output!r}"


def test_readme_quickstart_is_the_tested_quickstart_snippet() -> None:
    readme = [b for b in blocks(ROOT / "README.md") if b.language == "python"]
    assert readme[0].body == (SNIPPETS / "quickstart.py").read_text()
    output = following_output(readme[0])
    assert output is not None and output.body == (SNIPPETS / "quickstart.out").read_text()


# ---------------------------------------------------------------- the API reference


def public_modules() -> list[str]:
    names = ["frames2py"]
    for info in pkgutil.walk_packages(frames2py.__path__, "frames2py."):
        if not any(part.startswith("_") for part in info.name.split(".")):
            names.append(info.name)
    return names


def public_objects() -> dict[str, object]:
    found: dict[str, object] = {}
    for name in public_modules():
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        if exported is None:
            assert name in NAMESPACES, f"{name} is public but has no __all__"
            continue
        for attribute in exported:
            found[f"{name}.{attribute}"] = getattr(module, attribute)
    return found


def returned_types(objects: dict[str, object]) -> set[type]:
    """Classes that public functions return without exporting them."""
    classes = set()
    for obj in objects.values():
        if inspect.isfunction(obj):
            returned = typing.get_type_hints(obj).get("return")
            if inspect.isclass(returned) and returned.__module__.startswith("frames2py."):
                classes.add(returned)
    return classes


def resolve(target: str) -> object:
    parts = target.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(target)


def directives() -> list[tuple[str, str]]:
    found = []
    for page in sorted(API_PAGES.glob("*.md")):
        for line in page.read_text().splitlines():
            if line.startswith(":::"):
                found.append((str(page.relative_to(ROOT)), line[3:].strip()))
    return found


def test_the_api_reference_documents_the_whole_public_api_and_nothing_else() -> None:
    public = public_objects()
    documented = [(page, target, resolve(target)) for page, target in directives()]
    targets = [target for _, target, _ in documented]
    assert len(targets) == len(set(targets)), "an object is documented twice"
    missing = [name for name, obj in public.items() if not any(obj is d for _, _, d in documented)]
    assert not missing, f"public names with no API directive: {missing}"
    allowed = returned_types(public)
    for page, target, obj in documented:
        is_public = any(obj is p for p in public.values())
        assert is_public or obj in allowed, f"{page}: {target} is not public and no public function returns it"


def test_no_other_page_carries_api_directives() -> None:
    for page in PAGES:
        if page.parent != API_PAGES:
            assert not any(line.startswith(":::") for line in page.read_text().splitlines()), page


# ---------------------------------------------------------------- claims


INSTALL_PAGES = [ROOT / "README.md", CONTENT / "getting-started" / "installation.md"]


def shell_lines(page: Path) -> list[str]:
    return [line.split("#")[0].strip() for b in blocks(page) if b.language == "sh" for line in b.body.splitlines()]


@pytest.mark.parametrize("page", INSTALL_PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_installation_is_from_pypi_with_extras_in_brackets(page: Path) -> None:
    commands = shell_lines(page)
    assert "pip install frames2py" in commands, f"{page.relative_to(ROOT)}: no plain pip install from PyPI"
    extras = [c for c in commands if re.fullmatch(r'pip install "frames2py\[[a-z0-9]+(,[a-z0-9]+)*\]"', c)]
    assert extras, f"{page.relative_to(ROOT)}: no PyPI install with extras"


@pytest.mark.parametrize("page", INSTALL_PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_a_release_candidate_is_installed_explicitly(page: Path) -> None:
    version = frames2py.__version__
    text = page.read_text()
    if "rc" not in version:
        assert "--pre" not in text, f"{page.relative_to(ROOT)}: pre-release instructions for final release {version}"
        return
    assert f"pip install frames2py=={version}" in text
    assert "pip install --pre frames2py" in text


def test_git_installs_are_only_offered_as_the_development_version() -> None:
    for page in PAGES:
        for number, line in enumerate(page.read_text().splitlines(), 1):
            if "git+https://" in line:
                assert page == CONTENT / "getting-started" / "installation.md", f"{page.relative_to(ROOT)}:{number}"
    installation = (CONTENT / "getting-started" / "installation.md").read_text()
    development = installation.split("## Development version and source", 1)
    assert len(development) == 2 and "git+https://" not in development[0]


def test_the_readme_links_to_the_documentation_near_its_start_and_its_end() -> None:
    lines = (ROOT / "README.md").read_text().splitlines()
    assert any(PAGES_URL in line for line in lines[:15])
    assert any(PAGES_URL in line for line in lines[-25:])


def test_the_readme_logo_files_exist() -> None:
    readme = (ROOT / "README.md").read_text()
    logos = re.findall(r"raw\.githubusercontent\.com/siddiquifaras/frames2py/main/([^\"]+)", readme)
    assert {Path(path).name for path in logos} == {"frames2py-logo.svg", "frames2py-logo-dark.svg"}
    for path in logos:
        assert (ROOT / path).is_file(), path


@pytest.mark.parametrize("phrase", ["lock-free", "zero-copy", "zero-allocation", "SeqlockPublisher",
                                    "OverflowPolicy", "DROP_OLDEST", "events_dropped", "chunks_dropped",
                                    "buffer_fill_ratio", "ring buffer", "latest_snapshot", "frame_dtype",
                                    "chunk_size", "Telemetry"])
def test_no_page_describes_retired_or_forbidden_behaviour(phrase: str) -> None:
    for page in PAGES:
        assert phrase.lower() not in page.read_text().lower(), f"{page.relative_to(ROOT)} mentions {phrase!r}"


# ---------------------------------------------------------------- the temporal semantics table


SEMANTICS = CONTENT / "core" / "temporal-semantics.md"
TABLE_KERNELS = {"StackedHistogram": "stacked_histogram", "VoxelGrid": "voxel_grid"}
EVENT = re.compile(r"^([+-])(\d+)(?:\(p=(\d+)\))?(?:@(\d+)(?:,(\d+))?)?$")


def documented_cases() -> list[Case]:
    """The rows of the semantics page's two tables, read with the page's own notation."""
    cases, kernel = [], None
    for line in SEMANTICS.read_text().splitlines():
        if line.startswith("## "):
            kernel = TABLE_KERNELS.get(line[3:].strip())
        if kernel is None or not line.startswith("| ") or line.startswith(("| case ", "|---")):
            continue
        cells = re.findall(r"((?:`[^`]*`|[^|`])+)\|", line[1:])  # a | inside a code span is text
        name, params, calls, watermark, frame = (cell.strip().strip("`") for cell in cells[:5])
        bins, bin_us = (int(p) for p in params.split(","))
        parsed: list[tuple[tuple[int, int, int, int], ...] | str] = []
        for call in calls.split("|"):
            call = call.strip()
            if call in ("reset", "()"):
                parsed.append(RESET if call == "reset" else ())
                continue
            events = []
            for token in call.split():
                sign, t, p, x, y = EVENT.fullmatch(token).groups()  # type: ignore[union-attr]
                events.append((int(t), int(x or 0), int(y or 0), int(p) if p else int(sign == "+")))
            parsed.append(tuple(events))
        expected: dict[tuple[int, ...], int] = {}
        if frame != "all 0":
            for entry in frame.split(";"):
                label, values = entry.strip().split(":")
                *channel, pixel = label.split()
                x = int(pixel.removeprefix("x"))
                for j, value in enumerate(values.split()):
                    number = Fraction(value) * (bin_us if kernel == "voxel_grid" else 1)  # voxel: N / bin_us
                    assert number.denominator == 1, f"{name}: {value} is not exact"
                    if number:
                        key = (j, x) if kernel == "voxel_grid" else (int(channel == ["ON"]), j, x)
                        expected[key] = int(number)
        prefix = "histogram" if kernel == "stacked_histogram" else "voxel"
        cases.append(Case(f"{prefix}: {name}", kernel, tuple(parsed), expected,
                          None if watermark == "None" else int(watermark), bins, bin_us))
    return cases


def test_the_semantics_table_shows_exactly_the_hand_computed_cases() -> None:
    documented = {case.name: case for case in documented_cases()}
    assert sorted(documented) == sorted(case.name for case in CASES)
    for case in CASES:
        assert dataclasses.replace(documented[case.name], note=case.note) == case, case.name


@pytest.mark.parametrize("case", documented_cases(), ids=lambda c: c.name)
def test_the_installed_kernels_give_the_documented_frames(case: Case) -> None:
    kernel = (frames2py.StackedHistogram if case.kernel == "stacked_histogram" else frames2py.VoxelGrid)(
        bins=case.bins, bin_us=case.bin_us)
    accumulator = frames2py.Accumulator(TEMPORAL_SENSOR, kernel)
    for call in case.calls:
        if call == RESET:
            accumulator.reset()
        else:
            accumulator.accumulate(np.array(list(call), dtype=frames2py.EVENT_DTYPE))
    frame = accumulator.read()
    expected = np.zeros_like(frame)
    for key, numerator in case.expected.items():
        if case.kernel == "stacked_histogram":
            channel, j, x = key
            expected[channel, j, 0, x] = numerator
        else:
            j, x = key
            expected[j, 0, x] = voxel_value(numerator, case.bin_us)
    assert accumulator.watermark == case.watermark
    assert frame.dtype == expected.dtype and frame.tobytes() == expected.tobytes()
