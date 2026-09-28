"""Public API shape: constructors, exports, kernel classes and the event layout."""

from __future__ import annotations

import inspect

import pytest

from tests.contract.api import EVENT_DTYPE, impl
from tests.contract.helpers import SENSOR

TOP_LEVEL = {
    "Accumulator",
    "Engine",
    "EVENT_DTYPE",
    "EventCount",
    "Polarity",
    "TimeSurface",
    "ExpDecay",
    "TimestampDecay",
    "EngineStats",
    "SnapshotMeta",
}


def _parameters(obj: object) -> list[tuple[str, inspect._ParameterKind, object]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(obj).parameters.values()]  # type: ignore[arg-type]


def test_top_level_exports() -> None:
    assert set(impl.__all__) == TOP_LEVEL
    for name in TOP_LEVEL:
        assert hasattr(impl, name), name


def test_event_dtype_layout() -> None:
    assert impl.EVENT_DTYPE == EVENT_DTYPE
    assert impl.EVENT_DTYPE.itemsize == 13


def test_engine_signature() -> None:
    P = inspect.Parameter
    assert _parameters(impl.Engine) == [
        ("sensor_size", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("kernel", P.POSITIONAL_OR_KEYWORD, "event_count"),
        ("snapshot_interval_ms", P.KEYWORD_ONLY, 16.0),
    ]


@pytest.mark.parametrize(
    ("method", "params"),
    [("ingest", ["events"]), ("snapshot", []), ("start", []), ("stop", []), ("reset", [])],
)
def test_engine_methods(method: str, params: list[str]) -> None:
    assert [name for name, _, _ in _parameters(getattr(impl.Engine, method))][1:] == params


@pytest.mark.parametrize(("method", "params"), [("accumulate", ["events"]), ("read", []), ("reset", [])])
def test_accumulator_methods(method: str, params: list[str]) -> None:
    assert [name for name, _, _ in _parameters(getattr(impl.Accumulator, method))][1:] == params


def test_accumulator_signature() -> None:
    P = inspect.Parameter
    assert _parameters(impl.Accumulator) == [
        ("sensor_size", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("kernel", P.POSITIONAL_OR_KEYWORD, P.empty),
    ]


def test_chunk_size_is_not_accepted() -> None:
    with pytest.raises(TypeError):
        impl.Engine(SENSOR, "event_count", chunk_size=1024)


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        ("EventCount", []),
        ("Polarity", []),
        ("TimeSurface", []),
        ("ExpDecay", ["decay"]),
        ("TimestampDecay", ["tau_us"]),
    ],
)
def test_kernel_class_parameters(cls: str, params: list[str]) -> None:
    P = inspect.Parameter
    signature = _parameters(getattr(impl, cls))
    assert [name for name, _, _ in signature] == params
    assert all(default is P.empty for _, _, default in signature)


@pytest.mark.parametrize("name", ["event_count", "polarity", "time_surface"])
def test_parameterless_kernels_by_name(name: str) -> None:
    impl.Accumulator(SENSOR, name)
    impl.Engine(SENSOR, name)


@pytest.mark.parametrize("name", ["exp_decay", "timestamp_decay", "no_such_kernel"])
def test_other_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        impl.Accumulator(SENSOR, name)
    with pytest.raises(ValueError):
        impl.Engine(SENSOR, name)


@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf"), float("nan"), pytest.param(10**400, id="10**400")])
def test_tau_must_be_finite_and_positive(tau: float) -> None:
    with pytest.raises(ValueError):
        impl.TimestampDecay(tau)


@pytest.mark.parametrize(
    "decay", [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), float("-inf"), pytest.param(10**400, id="10**400")]
)
def test_decay_must_be_finite_and_strictly_between_0_and_1(decay: float) -> None:
    with pytest.raises(ValueError):
        impl.ExpDecay(decay)


@pytest.mark.parametrize("decay", [1e-300, 0.5, 1 - 1e-12])
def test_decay_inside_the_domain_is_accepted(decay: float) -> None:
    impl.ExpDecay(decay)


def test_tau_has_no_default() -> None:
    with pytest.raises(TypeError):
        impl.TimestampDecay()
