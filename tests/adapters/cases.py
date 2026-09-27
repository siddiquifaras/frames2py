"""Small files for each adapter, with the events each must yield, for the shared contract tests.

A case writes its file into a directory and returns the ``open`` to call, the path and the
expected events. Cases for an adapter whose backend is missing skip (``backends``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tests.adapters import evt_words as w
from tests.adapters.backends import require_backend


@dataclasses.dataclass(frozen=True)
class Written:
    open: Callable[..., Any]
    path: Path
    expected: np.ndarray
    open_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


def _evt3(directory: Path) -> Written:
    from frames2py.adapters import evt

    words = [w.evt3_time_high(3), w.evt3_time_low(1), w.evt3_y(5)]
    for i in range(40):
        words += [w.evt3_x(i, on=bool(i % 2)), w.evt3_time_low(1 + i)]
        if i % 5 == 0:
            words += [w.evt3_base(100 + i, on=True), w.evt3_vect12(0b1011), w.evt3_vect8(0b1), w.evt3_y(i + 6)]
    data = w.body(words, "3.0")
    path = w.write_raw(directory / "case.raw", words, "3.0", geometry=(640, 480))
    return Written(evt.open, path, w.reference_decode(data, "3.0"))


def _evt2(directory: Path) -> Written:
    from frames2py.adapters import evt

    words = [w.evt2_time_high(7)]
    for i in range(60):
        words.append(w.evt2_cd(i, 60 - i, on=bool(i % 3), low=i))
        if i % 17 == 0:
            words.append(w.evt2_time_high(8 + i))
    data = w.body(words, "2.0")
    path = w.write_raw(directory / "case.raw", words, "2.0", geometry=(640, 480))
    return Written(evt.open, path, w.reference_decode(data, "2.0"))


def _aedat4(directory: Path) -> Written:
    require_backend("dv_processing")
    from frames2py.adapters import aedat4
    from tests.adapters import aedat4_files

    packets = [[(1_000 + 10 * k + i, (7 * k + i) % 640, (3 * k) % 480, (k + i) % 2) for i in range(k % 5 + 1)] for k in range(20)]
    path = aedat4_files.write(directory / "case.aedat4", {"events": packets})
    return Written(aedat4.open, path, w.events([e for packet in packets for e in packet]))


CASES: dict[str, Callable[[Path], Written]] = {
    "aedat4": _aedat4,
    "evt2": _evt2,
    "evt3": _evt3,
}
