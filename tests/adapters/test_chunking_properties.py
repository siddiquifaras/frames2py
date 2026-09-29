"""Adapter output does not depend on how the input is cut: properties over generated inputs.

- The EVT 2.0 / 3.0 decoders, fed any split of any word stream, give what the per-word
  reference decoder gives for the whole body.
- Every adapter, on its committed real-data fixture, gives the same events for any
  ``batch_size``, in arrays of exactly that size except the last.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.adapters import evt_words as w
from tests.adapters.backends import require_backend
from tests.adapters.test_evt_decoding import feed

DATA = Path(__file__).resolve().parent.parent / "data"


def evt3_word() -> st.SearchStrategy[int]:
    return st.one_of(
        st.builds(w.evt3_time_high, st.integers(0, 4095)),
        st.builds(w.evt3_time_low, st.integers(0, 4095)),
        st.builds(w.evt3_y, st.integers(0, 2047)),
        st.builds(w.evt3_reserved_row, st.integers(0, 2047)),
        st.builds(lambda x, on: w.evt3_x(x, on=on), st.integers(0, 2047), st.booleans()),
        st.builds(lambda x, on: w.evt3_base(x, on=on), st.integers(0, 2047), st.booleans()),
        st.builds(w.evt3_vect12, st.integers(0, 4095)),
        st.builds(w.evt3_vect8, st.integers(0, 255)),
        st.integers(0, 0xFFFF),  # any word, other and reserved types included
    )


def evt3_segment() -> st.SearchStrategy[list[int]]:
    """A TIME_HIGH, a row and its events: consecutive segments step time forward and back by
    any amount, so wraps and backward discontinuities are followed by events."""
    return st.builds(lambda high, row, rest: [w.evt3_time_high(high), w.evt3_y(row), *rest],
                     st.integers(0, 4095), st.integers(0, 2047), st.lists(evt3_word(), max_size=12))


def evt2_word() -> st.SearchStrategy[int]:
    top = (1 << 28) - 1
    return st.one_of(
        st.builds(w.evt2_time_high, st.one_of(st.integers(0, top), st.integers(top - 300, top), st.integers(0, 300))),
        st.builds(lambda x, y, on, low: w.evt2_cd(x, y, on=on, low=low),
                  st.integers(0, 2047), st.integers(0, 2047), st.booleans(), st.integers(0, 63)),
        st.integers(0, 0xFFFFFFFF),
    )


def words(version: str) -> st.SearchStrategy[list[int]]:
    """Arbitrary words, and for EVT 3.0 structured segments among them."""
    if version == "2.0":
        return st.lists(evt2_word(), max_size=400)
    pieces = st.lists(st.one_of(evt3_word().map(lambda word: [word]), evt3_segment()), max_size=60)
    return pieces.map(lambda parts: [word for part in parts for word in part])


@st.composite
def split_body(draw: st.DrawFn, version: str) -> tuple[bytes, list[int]]:
    """A body of generated words plus a trailing partial word, and read sizes that cut it anywhere."""
    size = 4 if version == "2.0" else 2
    data = w.body(draw(words(version)), version) + draw(st.binary(max_size=size - 1))
    cuts = sorted(set(draw(st.lists(st.integers(1, max(len(data) - 1, 1)), max_size=40))))
    return data, [b - a for a, b in zip([0, *cuts], [*cuts, len(data)])]


@pytest.mark.parametrize("version", ["2.0", "3.0"])
@given(data=st.data())
def test_evt_decoding_under_any_split_matches_the_per_word_reference(version: str, data: st.DataObject) -> None:
    body, sizes = data.draw(split_body(version))
    assert feed(body, version, sizes).tolist() == w.reference_decode(body, version).tolist()


def _open(name: str) -> Callable[..., Any]:
    if name == "aedat4":
        require_backend("dv_processing")
        from frames2py.adapters import aedat4

        return lambda **kw: aedat4.open(DATA / "sparklers_100k.aedat4", **kw)
    if name == "hdf5":
        require_backend("h5py", "hdf5plugin")
        from frames2py.adapters import hdf5

        return lambda **kw: hdf5.open(DATA / "sparklers_100k.h5", group="events", t_offset="t_offset", **kw)
    from frames2py.adapters import evt

    if name == "evt2":
        return lambda **kw: evt.open(DATA / "sparklers_100k.evt2.raw", sensor_size=(640, 480), **kw)
    return lambda **kw: evt.open(DATA / "active_marker_head.evt3.raw", **kw)


@pytest.fixture(scope="module", params=["aedat4", "evt2", "evt3", "hdf5"])
def fixture_reader(request: pytest.FixtureRequest) -> tuple[Callable[..., Any], np.ndarray]:
    """How to open one fixture, and every event it holds, read with the decoder's own boundaries."""
    opener = _open(request.param)
    with opener() as reader:
        whole = np.concatenate(list(reader))
    return opener, whole


@settings(max_examples=25)
@given(batch_size=st.one_of(st.integers(1, 5_000), st.integers(1, 150_000)))
def test_batch_size_changes_only_the_boundaries(fixture_reader: tuple[Callable[..., Any], np.ndarray],
                                                batch_size: int) -> None:
    opener, whole = fixture_reader
    with opener(batch_size=batch_size) as reader:
        batches = list(reader)
    assert all(len(b) == batch_size for b in batches[:-1])
    assert 1 <= len(batches[-1]) <= batch_size
    assert all(b.flags.c_contiguous for b in batches)
    assert np.concatenate(batches).tobytes() == whole.tobytes()
