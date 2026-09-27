"""Real recordings for the opt-in tests and the adapter benchmark.

None of these files is in the repository. ``download`` fetches each into the recordings
directory (``$FRAMES2PY_RECORDINGS_DIR``, default ``~/.cache/frames2py/recordings``),
checks its size and SHA-256, and only then moves it into place. ``path`` returns a
recording's file after checking its hash again, or raises ``FileNotFoundError`` saying how
to get it.

    uv run python -m tests.recordings list
    uv run python -m tests.recordings download [NAME ...]

Licences are recorded as their sources state them. A recording whose licence is unknown is
downloaded from its source for local use only; it is never redistributed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

ENV_DIR: Final = "FRAMES2PY_RECORDINGS_DIR"
_PROPHESEE: Final = "https://kdrive.infomaniak.com/2/app/975517/share/{share}/files/{file}/download"
_PROPHESEE_PAGE: Final = "Prophesee sample recordings, https://docs.prophesee.ai/stable/datasets.html"
_CC0: Final = "CC0 1.0 (stated on the Prophesee datasets page)"


@dataclasses.dataclass(frozen=True)
class Recording:
    name: str
    format: str
    url: str
    size: int
    sha256: str
    licence: str
    source: str
    adapter: str
    """The ``frames2py.adapters`` module that reads it, and the ``open()`` arguments it needs."""
    open_kwargs: dict[str, Any]
    events: int
    """CD events in the recording, as independent decoders count them."""
    member: str | None = None
    """For an archive: the file to extract, whose hash is ``member_sha256``."""
    member_sha256: str | None = None

    @property
    def file_sha256(self) -> str:
        return self.member_sha256 or self.sha256


RECORDINGS: Final = {
    r.name: r
    for r in (
        Recording(
            "sparklers.raw", "evt2",
            _PROPHESEE.format(share="9e4f6a2b-7f56-4e14-b613-5b238c7add44", file=12350),
            2_109_142, "e84afbecdc07d2910ae846a4ae0ee246f5b9c97a53816c637d4f85c023d7c234",
            _CC0, f"{_PROPHESEE_PAGE}, 'sparklers' RAW EVT2; Gen3.0 640x480, header without geometry",
            "evt", {"sensor_size": (640, 480)}, 521_252,
        ),
        Recording(
            "200_jets_at_200hz.raw", "evt2",
            _PROPHESEE.format(share="884fa2f7-d856-49df-9eb5-5681e1937201", file=12264),
            2_338_354, "785520a25aa733bd66006588c2d09e539931958793844ee56a919b6abf6e2c71",
            _CC0, f"{_PROPHESEE_PAGE}, '200_jets_at_200hz' RAW EVT2; Gen3.1 640x480, header without geometry",
            "evt", {"sensor_size": (640, 480)}, 407_365,
        ),
        Recording(
            "active_marker.raw", "evt3",
            _PROPHESEE.format(share="d39fd010-8d56-4ce1-980d-224d5654b478", file=147192),
            108_741_633, "700e9c6dd5df7ff4b36c4167b5fb6d64124e8050cbf5b52b9736e9a1368028c5",
            _CC0, f"{_PROPHESEE_PAGE}, 'active_marker' RAW EVT3; IMX636 1280x720",
            "evt", {}, 22_316_758,
        ),
        Recording(
            "faery_evt3.raw", "evt3",
            "https://raw.githubusercontent.com/aestream/faery/ed2524eeab1f1eb70c625db3585675a2e2c440fb/tests/data/evt3.raw",
            9_363_577, "8eca19cd8f0580ae939490484a023d858192a4c102b6649d8f5e19a62cb77ad3",
            "unknown: the faery repository is LGPL-3.0 and states no licence for its test data",
            "faery tests/data/evt3.raw at commit ed2524e; IMX636 EVK4 1280x720",
            "evt", {}, 1_218_618,
        ),
        Recording(
            "dvp_test-minimal.aedat4", "aedat4",
            "https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8497ae38562201c209799b6cd5d1ebafd/tests/io/test_files/test-minimal.aedat4",
            2_024_458, "67798bc4726099a5e67334bc5ea5b81980817cc4afc22201d92a4395bd747480",
            "unknown: the dv-processing repository is Apache-2.0 and states no licence for its test files",
            "dv-processing tests/io/test_files/test-minimal.aedat4 at commit bf8c0ee; DVXplorer 640x480",
            "aedat4", {}, 255_283,
        ),
        Recording(
            "dvp_sample_data.aedat4", "aedat4",
            "https://gitlab.com/inivation/dv/dv-processing/-/raw/bf8c0ee8497ae38562201c209799b6cd5d1ebafd/tests/io/test_files/sample_data.aedat4",
            1_613_647, "627a9cfb3a42d7107d9df6b3c835fac7c18d95126a2f7d7c8a615e61982f9312",
            "unknown: the dv-processing repository is Apache-2.0 and states no licence for its test files",
            "dv-processing tests/io/test_files/sample_data.aedat4 at commit bf8c0ee; DAVIS346 346x260",
            "aedat4", {}, 9_193,
        ),
        Recording(
            "faery_davis346.aedat4", "aedat4",
            "https://raw.githubusercontent.com/aestream/faery/ed2524eeab1f1eb70c625db3585675a2e2c440fb/tests/data/davis346.aedat4",
            6_144_209, "b383462d72122c69fd96ebf095c384e3f1a2f092cc1b2681be52ebc1d260cf8a",
            "unknown: the faery repository is LGPL-3.0 and states no licence for its test data",
            "faery tests/data/davis346.aedat4 at commit ed2524e; DAVIS346 346x260",
            "aedat4", {}, 78_830,
        ),
        Recording(
            "dsec_thun_01_a_events_left.h5", "hdf5",
            "https://download.ifi.uzh.ch/rpg/DSEC/test/thun_01_a/thun_01_a_events_left.zip",
            267_754_002, "87a1d817e9c1a67c6795174a07dada0a2a4c629b6e60169304409dbbd17fa1d8",
            "CC BY-SA 4.0 (stated on the DSEC download page)",
            "DSEC, https://dsec.ifi.uzh.ch, test/thun_01_a left event camera; 640x480, Blosc-compressed HDF5",
            "hdf5", {"group": "events", "t_offset": "/t_offset", "sensor_size": (640, 480)}, 131_482_728,
            member="events.h5",
            member_sha256="ee2f8808cdc714db42a22e97ef1fd9a56e02a7a24d723621915cf9a1a609e4da",
        ),
    )
}


def directory() -> Path:
    return Path(os.environ.get(ENV_DIR) or Path.home() / ".cache" / "frames2py" / "recordings")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def path(name: str) -> Path:
    """The verified local file of recording *name*."""
    recording = RECORDINGS[name]
    target = directory() / name
    if not target.is_file():
        raise FileNotFoundError(
            f"recording {name!r} is not in {directory()}; run: uv run python -m tests.recordings download {name}"
        )
    actual = _sha256(target)
    if actual != recording.file_sha256:
        raise ValueError(f"{target} has SHA-256 {actual}, expected {recording.file_sha256}; delete it and download again")
    return target


def download(name: str) -> Path:
    """Fetch recording *name* unless a verified copy is already there."""
    recording = RECORDINGS[name]
    target = directory() / name
    if target.is_file() and _sha256(target) == recording.file_sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(recording.url, headers={"User-Agent": "frames2py-tests"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        while block := response.read(1 << 20):
            digest.update(block)
            size += len(block)
            out.write(block)
    if size != recording.size or digest.hexdigest() != recording.sha256:
        part.unlink()
        raise ValueError(
            f"{recording.url} gave {size} bytes with SHA-256 {digest.hexdigest()}; "
            f"expected {recording.size} bytes, {recording.sha256}"
        )
    if recording.member is None:
        part.replace(target)
        return target
    extracted = target.with_name(target.name + ".member")
    with zipfile.ZipFile(part) as archive, archive.open(recording.member) as src, extracted.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    part.unlink()
    if _sha256(extracted) != recording.file_sha256:
        extracted.unlink()
        raise ValueError(f"{recording.member} in {recording.url} does not have SHA-256 {recording.file_sha256}")
    extracted.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.recordings")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="print every recording and whether it is present")
    fetch = commands.add_parser("download", help="download and verify recordings (default: all)")
    fetch.add_argument("names", nargs="*", metavar="NAME")
    args = parser.parse_args(argv)
    unknown = sorted(set(getattr(args, "names", ())) - set(RECORDINGS))
    if unknown:
        parser.error(f"unknown recording(s) {unknown}; choose from {sorted(RECORDINGS)}")
    if args.command == "list":
        for r in RECORDINGS.values():
            present = (directory() / r.name).is_file()
            print(f"{r.name:32} {r.format:7} {r.size / 1e6:8.1f} MB  {'present' if present else 'absent'}  {r.licence}")
        print(f"directory: {directory()}")
        return 0
    for name in args.names or list(RECORDINGS):
        print(f"{name}: {download(name)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
