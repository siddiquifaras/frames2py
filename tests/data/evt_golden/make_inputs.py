"""Write the crafted EVT inputs added after the first OpenEB comparison (the ``N`` cases).

    uv run python -m tests.data.evt_golden.make_inputs [--check]

Words come from the EVT 2.0 / 3.0 format pages (``tests.adapters.evt_words``). Where vectors
appear, they come as full VECT_12, VECT_12, VECT_8 triples, because OpenEB's default decoder
always reads VECT_12 as the start of one; otherwise that known difference would hide the
rule a case is about. The other inputs here were written by the same kind of generator
during the adapter research and are committed as they were decoded by OpenEB.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from tests.adapters import evt_words as w

HERE: Final = Path(__file__).resolve().parent


def _triple(mask: int = 1) -> list[int]:
    return [w.evt3_vect12(mask), w.evt3_vect12(mask), w.evt3_vect8(mask)]


def _warm(first: int) -> list[int]:
    """TIME_HIGH words stepping forward by 1000 up to *first*, so every decoder reaches it by forward steps."""
    return [w.evt3_time_high(v) for v in range(1000, first, 1000)]


def _step(prev: int, new: int) -> list[int]:
    return [*_warm(prev), w.evt3_time_high(prev), w.evt3_time_low(1), w.evt3_y(7), w.evt3_x(0, on=True),
            w.evt3_time_high(new), w.evt3_time_low(1), w.evt3_x(1, on=True)]


CASES: Final[dict[str, tuple[str, list[int]]]] = {
    "evt2_N1_cd_before_first_time_high": ("2.0", [
        w.evt2_cd(0, 7, on=True, low=5), w.evt2_time_high(10), w.evt2_cd(1, 7, on=True, low=2)]),
    "evt3_N1_events_before_first_row": ("3.0", [
        w.evt3_time_high(10), w.evt3_time_low(5), w.evt3_x(1, on=True), w.evt3_y(7), w.evt3_x(2, on=True)]),
    "evt3_N2_vectors_before_first_base": ("3.0", [
        w.evt3_time_high(10), w.evt3_time_low(5), w.evt3_y(7), *_triple(0b111),
        w.evt3_base(100, on=True), *_triple(0b1)]),
    "evt3_N3_words_before_first_time_high": ("3.0", [
        w.evt3_y(7), w.evt3_x(1, on=True), w.evt3_time_high(10), w.evt3_x(2, on=True), w.evt3_y(8), w.evt3_x(3, on=True)]),
    "evt3_N4_time_low_reset_on_time_high_change": ("3.0", [
        w.evt3_time_high(10), w.evt3_time_low(100), w.evt3_y(7), w.evt3_x(0, on=True),
        w.evt3_time_high(10), w.evt3_x(1, on=True), w.evt3_time_high(11), w.evt3_x(2, on=True)]),
    "evt3_N5_backstep_3841_from_3841": ("3.0", _step(3841, 0)),
    "evt3_N6_backstep_3840_from_3840": ("3.0", _step(3840, 0)),
    "evt3_N7_backstep_3841_from_4095": ("3.0", _step(4095, 254)),
    "evt3_N8_backstep_3840_from_4095": ("3.0", _step(4095, 255)),
    "evt3_N9_reserved_row_with_triples": ("3.0", [
        w.evt3_time_high(10), w.evt3_time_low(5), w.evt3_y(7), w.evt3_x(1, on=True),
        w.evt3_reserved_row(9), w.evt3_x(2, on=True), w.evt3_base(100, on=True), *_triple(0b1),
        w.evt3_y(8), w.evt3_x(4, on=False), *_triple(0b1)]),
}


def build(name: str) -> bytes:
    version, words = CASES[name]
    return w.header(version, (640, 480)) + w.body(words, version)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.data.evt_golden.make_inputs")
    parser.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    args = parser.parse_args(argv)
    status = 0
    for name in CASES:
        data, target = build(name), HERE / f"{name}.raw"
        if args.check:
            same = target.is_file() and target.read_bytes() == data
            status |= not same
            print(f"{name}: {'identical' if same else 'DIFFERENT'}")
        else:
            target.write_bytes(data)
    return status


if __name__ == "__main__":
    sys.exit(main())
