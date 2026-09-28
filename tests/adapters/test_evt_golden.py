"""Crafted EVT inputs against OpenEB 5.2.0's recorded output (``tests/data/evt_golden``).

Where the Frames2Py rules agree with OpenEB's default decoder, the adapter must give exactly
OpenEB's events. Where they differ on purpose, the Frames2Py events are written out below,
worked by hand from each input's words, with the reason; OpenEB's output stays in the record
as the comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from frames2py.adapters import evt

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "evt_golden"
RECORD = json.loads((GOLDEN / "openeb-5.2.0.json").read_text())["cases"]

WRAP_RULE = "EVT3 wrap: Frames2Py wraps only 4095 -> 0 or a backward step over 3840 (OpenEB's validator); OpenEB's decoder wraps at 2048"
VECTORS = "vectors follow the format page per word; OpenEB assumes a 12 + 12 + 8 triple and drops a triple past the right edge"
HEIGHT = "rows at or beyond the height are yielded and counted out of bounds by the core; OpenEB drops them"

DIFFERENCES: dict[str, tuple[list[tuple[int, int, int, int]], str]] = {
    "evt3_B2_backward_over_half": ([(12_288_100, 0, 7, 1), (2_048_100, 1, 7, 1)], WRAP_RULE),
    "evt3_H_hal_3000_to_952": ([(12_288_001, 0, 7, 1), (3_899_393, 1, 7, 1)], WRAP_RULE),
    "evt3_N6_backstep_3840_from_3840": ([(15_728_641, 0, 7, 1), (1, 1, 7, 1)], WRAP_RULE),
    "evt3_N8_backstep_3840_from_4095": ([(16_773_121, 0, 7, 1), (1_044_481, 1, 7, 1)], WRAP_RULE),
    "evt3_V2_lone_vect12_then_addr_x": ([(40_965, 100, 7, 1), (40_965, 300, 7, 0), (40_966, 301, 7, 0)], VECTORS),
    "evt3_V3_triple_at_right_edge": ([(40_965, 620, 7, 1), (40_965, 632, 7, 1), (40_965, 644, 7, 1)], VECTORS),
    "evt3_V5_standalone_vect8": ([(40_965, 100, 7, 1), (40_965, 400, 7, 0)], VECTORS),
    "evt3_V6_y_beyond_height": ([(40_965, 3, 500, 1), (40_965, 4, 7, 1)], HEIGHT),
    "evt3_V7_reserved_row_type1": ([(40_965, 1, 7, 1), (40_965, 4, 8, 0), (40_965, 100, 8, 1)], VECTORS),
}


def decode(name: str) -> list[tuple[int, int, int, int]]:
    with evt.open(GOLDEN / f"{name}.raw") as reader:
        return [tuple(int(v) for v in e) for batch in reader for e in batch.tolist()]


def test_record_covers_exactly_the_committed_inputs() -> None:
    assert sorted(RECORD) == sorted(p.stem for p in GOLDEN.glob("*.raw"))
    for name, case in RECORD.items():
        assert hashlib.sha256((GOLDEN / f"{name}.raw").read_bytes()).hexdigest() == case["sha256"], name
    assert set(DIFFERENCES) <= set(RECORD)


@pytest.mark.parametrize("name", sorted(set(RECORD) - set(DIFFERENCES)))
def test_agrees_with_openeb(name: str) -> None:
    assert decode(name) == [tuple(e) for e in RECORD[name]["events"]]


@pytest.mark.parametrize("name", sorted(DIFFERENCES))
def test_differs_from_openeb_as_documented(name: str) -> None:
    expected, reason = DIFFERENCES[name]
    assert decode(name) == expected, reason
    assert [tuple(e) for e in RECORD[name]["events"]] != expected


@pytest.mark.parametrize(("name", "flagged"), [
    ("evt3_N5_backstep_3841_from_3841", False), ("evt3_N6_backstep_3840_from_3840", True),
    ("evt3_N7_backstep_3841_from_4095", False), ("evt3_N8_backstep_3840_from_4095", True),
    ("evt3_B2_backward_over_half", True), ("evt3_H_hal_3000_to_952", True), ("evt3_A_wrap_4095_to_0", False),
])
def test_wraps_exactly_where_openebs_validator_reports_no_violation(name: str, flagged: bool) -> None:
    # A backward TIME_HIGH step is a wrap in Frames2Py iff OpenEB's validator doesn't flag it.
    assert ("NonMonotonicTimeHigh" in RECORD[name]["violations"]) is flagged
    (t_before, *_), (t_after, *_) = decode(name)[:2]
    wrapped = t_before < 1 << 24 <= t_after
    assert wrapped is not flagged
