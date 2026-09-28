"""EVT 2.0 / 3.0 decoding: known word streams, the per-word reference, and read-size invariance."""

from __future__ import annotations

import numpy as np
import pytest

from frames2py.adapters._evt_decode import EVT3_SLICE_WORDS, Evt2Decoder, Evt3Decoder
from tests.adapters import evt_words as w

WRAP2 = 1 << 34
WRAP3 = 1 << 24


def decode(words: list[int], version: str, chunks: list[int] | None = None) -> np.ndarray:
    """Vectorised decode of a body, fed whole or in pieces of the given sizes."""
    data = w.body(words, version)
    return feed(data, version, chunks)


def feed(data: bytes, version: str, chunks: list[int] | None = None) -> np.ndarray:
    decoder = Evt2Decoder() if version == "2.0" else Evt3Decoder()
    parts, at = [], 0
    for size in chunks or [len(data)]:
        parts += decoder.feed(data[at : at + size])
        at += size
    if at < len(data):
        parts += decoder.feed(data[at:])
    return np.concatenate(parts) if parts else np.empty(0, dtype=w.EVENT_DTYPE)


def check(words: list[int], version: str, expected: list[tuple[int, int, int, int]]) -> None:
    """Both the decoder and the per-word reference give exactly *expected*."""
    want = w.events(expected)
    got = decode(words, version)
    assert got.dtype == w.EVENT_DTYPE
    assert got.tolist() == want.tolist()
    assert w.reference_decode(w.body(words, version), version).tolist() == want.tolist()


class TestEvt2:
    def test_time_is_time_high_with_the_event_low_bits(self) -> None:
        check([w.evt2_time_high(1000), w.evt2_cd(3, 4, on=True, low=5), w.evt2_cd(639, 479, on=False, low=63)],
              "2.0", [(64005, 3, 4, 1), (64063, 639, 479, 0)])

    def test_events_before_the_first_time_high_are_dropped(self) -> None:
        check([w.evt2_cd(1, 1, on=True), w.evt2_time_high(10), w.evt2_cd(2, 2, on=True, low=1)],
              "2.0", [(641, 2, 2, 1)])

    def test_no_time_high_at_all_gives_no_events(self) -> None:
        check([w.evt2_cd(1, 1, on=True)] * 5, "2.0", [])

    def test_counter_wrap_adds_2_pow_34(self) -> None:
        top = (1 << 28) - 1
        check([w.evt2_time_high(top), w.evt2_cd(0, 7, on=True, low=1), w.evt2_time_high(3), w.evt2_cd(1, 7, on=True, low=1)],
              "2.0", [(top * 64 + 1, 0, 7, 1), (WRAP2 + 3 * 64 + 1, 1, 7, 1)])

    @pytest.mark.parametrize(("new", "wraps"), [(156, True), (157, False)])
    def test_wrap_threshold(self, new: int, wraps: bool) -> None:
        # A backward step of at least (2**28 - 1) * 64 - 10000 us is a wrap: from the top,
        # 156 * 64 = 9984 us is within 10000 us of zero and 157 * 64 = 10048 is not.
        top = (1 << 28) - 1
        t = new * 64 + (WRAP2 if wraps else 0)
        check([w.evt2_time_high(top), w.evt2_cd(0, 7, on=True), w.evt2_time_high(new), w.evt2_cd(1, 7, on=True)],
              "2.0", [(top * 64, 0, 7, 1), (t, 1, 7, 1)])

    def test_other_backward_steps_are_kept(self) -> None:
        check([w.evt2_time_high(1 << 27), w.evt2_cd(0, 7, on=True), w.evt2_time_high(1 << 26), w.evt2_cd(1, 7, on=True),
               w.evt2_time_high(0), w.evt2_cd(2, 7, on=True)],
              "2.0", [((1 << 27) * 64, 0, 7, 1), ((1 << 26) * 64, 1, 7, 1), (0, 2, 7, 1)])

    def test_forward_jump_is_kept(self) -> None:
        check([w.evt2_time_high(100), w.evt2_cd(0, 7, on=True), w.evt2_time_high(1 << 27), w.evt2_cd(1, 7, on=True)],
              "2.0", [(6400, 0, 7, 1), ((1 << 27) * 64, 1, 7, 1)])

    def test_low_bits_out_of_order_are_kept(self) -> None:
        check([w.evt2_time_high(1000), w.evt2_cd(0, 7, on=True, low=40), w.evt2_cd(1, 7, on=True, low=10)],
              "2.0", [(64040, 0, 7, 1), (64010, 1, 7, 1)])

    def test_other_word_types_are_ignored(self) -> None:
        check([w.evt2_time_high(10), w.evt2_other(0xA, 0x123), w.evt2_other(0xE, 0xFFFFFFF), w.evt2_other(0xF, 1),
               w.evt2_other(0x5, 0xFFFF), w.evt2_cd(5, 6, on=False, low=2)],
              "2.0", [(642, 5, 6, 0)])


class TestEvt3Time:
    def test_time_is_time_high_and_time_low(self) -> None:
        check([w.evt3_time_high(10), w.evt3_time_low(5), w.evt3_y(7), w.evt3_x(1, on=True)],
              "3.0", [(10 << 12 | 5, 1, 7, 1)])

    def test_before_the_first_time_low_the_low_bits_are_zero(self) -> None:
        check([w.evt3_time_high(10), w.evt3_y(7), w.evt3_x(1, on=True), w.evt3_time_low(9), w.evt3_x(2, on=True)],
              "3.0", [(10 << 12, 1, 7, 1), (10 << 12 | 9, 2, 7, 1)])

    def test_words_before_the_first_time_high_are_skipped(self) -> None:
        # The row set before it doesn't count: the event after it has no row.
        check([w.evt3_time_low(3), w.evt3_y(7), w.evt3_x(1, on=True), w.evt3_time_high(10), w.evt3_x(2, on=True),
               w.evt3_y(8), w.evt3_x(3, on=True)],
              "3.0", [(10 << 12, 3, 8, 1)])

    def test_value_changing_time_high_resets_time_low(self) -> None:
        check([w.evt3_time_high(10), w.evt3_time_low(100), w.evt3_y(7), w.evt3_x(0, on=True),
               w.evt3_time_high(10), w.evt3_x(1, on=True),
               w.evt3_time_high(11), w.evt3_x(2, on=True)],
              "3.0", [(10 << 12 | 100, 0, 7, 1), (10 << 12 | 100, 1, 7, 1), (11 << 12, 2, 7, 1)])

    def test_time_low_going_back_within_a_time_high_is_kept(self) -> None:
        check([w.evt3_time_high(10), w.evt3_time_low(400), w.evt3_y(7), w.evt3_x(0, on=True),
               w.evt3_time_low(0), w.evt3_x(1, on=True)],
              "3.0", [(10 << 12 | 400, 0, 7, 1), (10 << 12, 1, 7, 1)])

    def test_4095_to_0_is_a_wrap(self) -> None:
        check([w.evt3_time_high(4095), w.evt3_time_low(4000), w.evt3_y(7), w.evt3_x(0, on=True),
               w.evt3_time_high(0), w.evt3_time_low(10), w.evt3_x(1, on=True)],
              "3.0", [(4095 << 12 | 4000, 0, 7, 1), (WRAP3 + 10, 1, 7, 1)])

    @pytest.mark.parametrize(("prev", "new"), [(3841, 0), (4095, 254), (4000, 100)])
    def test_backward_step_over_3840_is_a_wrap(self, prev: int, new: int) -> None:
        check([w.evt3_time_high(prev), w.evt3_y(7), w.evt3_x(0, on=True), w.evt3_time_high(new), w.evt3_x(1, on=True)],
              "3.0", [(prev << 12, 0, 7, 1), (WRAP3 + (new << 12), 1, 7, 1)])

    @pytest.mark.parametrize(("prev", "new"), [(3840, 0), (4095, 255), (3000, 952), (3000, 500), (2048, 0), (2000, 1999)])
    def test_other_backward_steps_are_kept(self, prev: int, new: int) -> None:
        # Steps of 2048-3840 are kept too: they are not wraps here, although OpenEB's decoder wraps them.
        check([w.evt3_time_high(prev), w.evt3_y(7), w.evt3_x(0, on=True), w.evt3_time_high(new), w.evt3_x(1, on=True)],
              "3.0", [(prev << 12, 0, 7, 1), (new << 12, 1, 7, 1)])

    def test_forward_jumps_are_kept(self) -> None:
        check([w.evt3_time_high(0), w.evt3_y(7), w.evt3_x(0, on=True), w.evt3_time_high(4095), w.evt3_x(1, on=True)],
              "3.0", [(0, 0, 7, 1), (4095 << 12, 1, 7, 1)])

    def test_first_time_high_is_never_a_wrap(self) -> None:
        check([w.evt3_time_high(4095), w.evt3_y(7), w.evt3_x(0, on=True), w.evt3_time_high(0), w.evt3_x(1, on=True)],
              "3.0", [(4095 << 12, 0, 7, 1), (WRAP3, 1, 7, 1)])

    def test_wraps_accumulate(self) -> None:
        words = [w.evt3_time_high(0), w.evt3_y(7)]
        for k in range(3):
            words += [w.evt3_time_high(2048), w.evt3_time_high(4095), w.evt3_time_high(0), w.evt3_x(k, on=True)]
        check(words, "3.0", [((k + 1) * WRAP3, k, 7, 1) for k in range(3)])


class TestEvt3Rows:
    def test_events_before_the_first_row_are_dropped(self) -> None:
        check([w.evt3_time_high(1), w.evt3_x(1, on=True), w.evt3_base(10, on=True), w.evt3_vect12(0b1),
               w.evt3_y(3), w.evt3_x(2, on=False)],
              "3.0", [(1 << 12, 2, 3, 0)])

    def test_reserved_row_emits_nothing_and_does_not_move_the_vector_base(self) -> None:
        check([w.evt3_time_high(10), w.evt3_time_low(5), w.evt3_y(7), w.evt3_x(1, on=True),
               w.evt3_reserved_row(9), w.evt3_x(2, on=True), w.evt3_base(100, on=True), w.evt3_vect12(0b1), w.evt3_vect8(0b1),
               w.evt3_y(8), w.evt3_x(4, on=False), w.evt3_vect12(0b11)],
              "3.0", [(40965, 1, 7, 1), (40965, 4, 8, 0), (40965, 100, 8, 1), (40965, 101, 8, 1)])

    def test_rows_beyond_any_sensor_height_are_decoded(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(2047), w.evt3_x(3, on=True), w.evt3_y(7), w.evt3_x(4, on=True)],
              "3.0", [(1 << 12, 3, 2047, 1), (1 << 12, 4, 7, 1)])

    def test_system_type_bit_is_not_part_of_y(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(5) | 1 << 11, w.evt3_x(3, on=True)], "3.0", [(1 << 12, 3, 5, 1)])


class TestEvt3Vectors:
    def test_vectors_emit_set_bits_from_the_base_and_move_it_on(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(100, on=True),
               w.evt3_vect12(0b1000_0000_0101), w.evt3_vect12(0b1), w.evt3_vect8(0b1000_0001)],
              "3.0", [(4096, 100, 7, 1), (4096, 102, 7, 1), (4096, 111, 7, 1), (4096, 112, 7, 1),
                      (4096, 124, 7, 1), (4096, 131, 7, 1)])

    def test_vect8_ignores_its_upper_bits(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(0, on=False), 0x5F01, w.evt3_vect8(0b10)],
              "3.0", [(4096, 0, 7, 0), (4096, 9, 7, 0)])

    def test_empty_vectors_still_move_the_base(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(0, on=True), w.evt3_vect12(0), w.evt3_vect8(0), w.evt3_vect12(1)],
              "3.0", [(4096, 20, 7, 1)])

    def test_single_events_do_not_move_the_base(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(100, on=True), w.evt3_x(500, on=False), w.evt3_vect12(0b1)],
              "3.0", [(4096, 500, 7, 0), (4096, 100, 7, 1)])

    def test_lone_vectors_and_the_right_edge_follow_the_format_page(self) -> None:
        # No 12 + 12 + 8 grouping is assumed, and nothing is dropped near the right edge.
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(630, on=True), w.evt3_vect12(0b1), w.evt3_vect12(0b1),
               w.evt3_time_low(1), w.evt3_vect8(0b1)],
              "3.0", [(4096, 630, 7, 1), (4096, 642, 7, 1), (4097, 654, 7, 1)])

    def test_vectors_before_any_base_emit_nothing(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_vect12(0xFFF), w.evt3_base(5, on=False), w.evt3_vect8(0b1)],
              "3.0", [(4096, 5, 7, 0)])

    def test_row_change_keeps_the_base(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(5, on=True), w.evt3_vect12(0b1), w.evt3_y(9), w.evt3_vect8(0b1)],
              "3.0", [(4096, 5, 7, 1), (4096, 17, 9, 1)])

    def test_vector_x_beyond_65535_is_malformed(self) -> None:
        words = [w.evt3_time_high(1), w.evt3_y(7), w.evt3_base(2047, on=True)] + [w.evt3_vect12(0)] * 5500
        words.append(w.evt3_vect12(0xFFF))
        with pytest.raises(ValueError, match="malformed"):
            decode(words, "3.0")
        with pytest.raises(ValueError):
            w.reference_decode(w.body(words, "3.0"), "3.0")

    def test_other_word_types_are_ignored(self) -> None:
        check([w.evt3_time_high(1), w.evt3_y(7), w.evt3_other(0xA, 0x7FF), w.evt3_other(0xE, 0x1), w.evt3_other(0xF, 0xFFF),
               w.evt3_other(0x7, 0xF), w.evt3_other(0x9, 0x123), w.evt3_x(3, on=True)],
              "3.0", [(4096, 3, 7, 1)])


def random_evt3(rng: np.random.Generator, n: int) -> list[int]:
    """A word stream mixing every type, wraps, backward steps and reserved rows."""
    makers = [
        lambda: w.evt3_time_high(int(rng.integers(0, 4096))),
        lambda: w.evt3_time_high(4095), lambda: w.evt3_time_high(0),
        lambda: w.evt3_time_low(int(rng.integers(0, 4096))),
        lambda: w.evt3_y(int(rng.integers(0, 2048))),
        lambda: w.evt3_reserved_row(int(rng.integers(0, 2048))),
        lambda: w.evt3_x(int(rng.integers(0, 2048)), on=bool(rng.integers(2))),
        lambda: w.evt3_base(int(rng.integers(0, 1800)), on=bool(rng.integers(2))),
        lambda: w.evt3_vect12(int(rng.integers(0, 4096))),
        lambda: w.evt3_vect8(int(rng.integers(0, 256))),
        lambda: w.evt3_other(int(rng.choice([0x7, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF])), int(rng.integers(0, 4096))),
    ]
    weights = np.array([3, 0.2, 0.2, 4, 3, 0.5, 8, 3, 6, 3, 1])
    picks = rng.choice(len(makers), size=n, p=weights / weights.sum())
    return [makers[i]() for i in picks]


def random_evt2(rng: np.random.Generator, n: int) -> list[int]:
    top = (1 << 28) - 1
    makers = [
        lambda: w.evt2_time_high(int(rng.integers(0, top + 1))),
        lambda: w.evt2_time_high(top), lambda: w.evt2_time_high(int(rng.integers(0, 200))),
        lambda: w.evt2_cd(int(rng.integers(0, 2048)), int(rng.integers(0, 2048)), on=bool(rng.integers(2)),
                          low=int(rng.integers(0, 64))),
        lambda: w.evt2_other(int(rng.choice([0x2, 0x5, 0xA, 0xE, 0xF])), int(rng.integers(0, 1 << 28))),
    ]
    weights = np.array([2, 0.3, 0.3, 20, 1])
    picks = rng.choice(len(makers), size=n, p=weights / weights.sum())
    return [makers[i]() for i in picks]


class TestAgainstTheReference:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_evt3_streams(self, seed: int) -> None:
        words = random_evt3(np.random.default_rng(seed), 3000)
        assert decode(words, "3.0").tolist() == w.reference_decode(w.body(words, "3.0"), "3.0").tolist()

    @pytest.mark.parametrize("seed", range(20))
    def test_random_evt2_streams(self, seed: int) -> None:
        words = random_evt2(np.random.default_rng(seed), 3000)
        assert decode(words, "2.0").tolist() == w.reference_decode(w.body(words, "2.0"), "2.0").tolist()

    def test_state_carries_across_internal_slices(self) -> None:
        # A stream longer than one slice, with time, row and vector state crossing its end.
        rng = np.random.default_rng(99)
        words = random_evt3(rng, 2 * EVT3_SLICE_WORDS + 5000)
        assert decode(words, "3.0").tolist() == w.reference_decode(w.body(words, "3.0"), "3.0").tolist()


def boundaries_around(words: list[int], version: str, kinds: set[int]) -> list[int]:
    """Chunk sizes splitting the body just before, inside and just after every word of *kinds*."""
    size = 4 if version == "2.0" else 2
    shift = 28 if version == "2.0" else 12
    cuts = set()
    for i, word in enumerate(words):
        if word >> shift in kinds:
            cuts |= {i * size, i * size + 1, (i + 1) * size}
    edges = sorted(c for c in cuts if 0 < c < len(words) * size)
    return [b - a for a, b in zip([0, *edges], [*edges, len(words) * size])]


class TestReadSizeInvariance:
    @pytest.mark.parametrize("version", ["2.0", "3.0"])
    @pytest.mark.parametrize("seed", range(6))
    def test_any_split_gives_the_same_events(self, version: str, seed: int) -> None:
        rng = np.random.default_rng(1000 + seed)
        words = (random_evt2 if version == "2.0" else random_evt3)(rng, 2500)
        whole = decode(words, version)
        n_bytes = len(w.body(words, version))
        one_byte = decode(words, version, [1] * n_bytes)
        random_sizes = decode(words, version, list(rng.integers(1, 97, size=n_bytes)))
        assert one_byte.tolist() == whole.tolist()
        assert random_sizes.tolist() == whole.tolist()

    @pytest.mark.parametrize(
        ("version", "kinds"),
        [("2.0", {0x8}), ("2.0", {0x0, 0x1}), ("3.0", {0x8}), ("3.0", {0x6}), ("3.0", {0x3, 0x4, 0x5}), ("3.0", {0x0, 0x1})],
        ids=["evt2-time-high", "evt2-cd", "evt3-time-high", "evt3-time-low", "evt3-vectors", "evt3-rows"],
    )
    def test_splits_around_state_words(self, version: str, kinds: set[int]) -> None:
        rng = np.random.default_rng(7)
        words = (random_evt2 if version == "2.0" else random_evt3)(rng, 2500)
        assert decode(words, version, boundaries_around(words, version, kinds)).tolist() == decode(words, version).tolist()

    def test_partial_word_is_carried_to_the_next_read(self) -> None:
        data = w.body([w.evt3_time_high(1), w.evt3_y(7), w.evt3_x(3, on=True)], "3.0")
        decoder = Evt3Decoder()
        assert decoder.feed(data[:5]) == []
        assert decoder.pending_bytes == 1
        (events,) = decoder.feed(data[5:])
        assert events.tolist() == [(4096, 3, 7, 1)]
        assert decoder.pending_bytes == 0

    @pytest.mark.parametrize("version", ["2.0", "3.0"])
    def test_trailing_partial_word_is_not_decoded(self, version: str) -> None:
        if version == "2.0":
            words = [w.evt2_time_high(1), w.evt2_cd(3, 4, on=True)]
        else:
            words = [w.evt3_time_high(1), w.evt3_y(4), w.evt3_x(3, on=True)]
        data = w.body(words, version)
        truncated = data + data[-3:] if version == "2.0" else data + data[-1:]
        decoder = Evt2Decoder() if version == "2.0" else Evt3Decoder()
        events = np.concatenate(decoder.feed(truncated))
        assert events.tolist() == decode(words, version).tolist()
        assert decoder.pending_bytes == (3 if version == "2.0" else 1)
