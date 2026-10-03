"""The temporal-kernel gate's harness (``benchmarks/temporal_gate_preregistration.md``), not performance."""

from __future__ import annotations

import copy
import functools
import json
from typing import Any

import numpy as np
import pytest

import frames2py.kernels._temporal as temporal_module
from frames2py import EVENT_DTYPE

from benchmarks import gate, results
from benchmarks.__main__ import main
from benchmarks.matrix import (
    GATE_BATCH_INTERVALS,
    GATE_DISTRIBUTIONS,
    GATE_RESOLUTIONS,
    REFERENCE_MACHINE,
    TEMPORAL_KERNEL_CONFIGS,
    Cell,
    default_seed,
    temporal_gate_cells,
)
from benchmarks.measure import Policy
from benchmarks.runner import run_suite
from benchmarks.targets import TARGETS
from benchmarks.targets.temporal import plane_share, reference
from tests.temporal_oracle import TemporalReference
from tests.test_benchmarks import _FakeTarget, _measure, _set_rates, _uniform


class TestDefinition:
    def test_150_distinct_cells_over_the_preregistered_dimensions(self) -> None:
        cells = temporal_gate_cells()
        assert len(cells) == len({c.condition for c in cells}) == 150
        assert TEMPORAL_KERNEL_CONFIGS == {
            "stacked_histogram_5x10000": ("stacked_histogram", 5, 10_000),
            "stacked_histogram_15x3333": ("stacked_histogram", 15, 3_333),
            "stacked_histogram_10x5000": ("stacked_histogram", 10, 5_000),
            "voxel_grid_5x12500": ("voxel_grid", 5, 12_500),
            "voxel_grid_15x3571": ("voxel_grid", 15, 3_571),
        }
        assert {c.kernel for c in cells} == set(TEMPORAL_KERNEL_CONFIGS)
        assert {c.sensor_size for c in cells} == set(GATE_RESOLUTIONS)
        assert {(c.batch_size, c.interval_ms) for c in cells} == set(GATE_BATCH_INTERVALS)
        assert {c.distribution for c in cells} == set(GATE_DISTRIBUTIONS)
        assert all(c.in_gate for c in cells)

    def test_streams_run_at_20_events_per_us_with_the_v1_seed_rule(self) -> None:
        cell = temporal_gate_cells()[0]
        assert cell.seed == default_seed(cell.sensor_size, cell.batch_size, cell.distribution)
        workload = cell.workload()
        assert workload.event_rate_hz == 20_000_000
        first, second = workload.batches(2)
        index = np.arange(2 * cell.batch_size)
        assert np.array_equal(np.concatenate([first["t"], second["t"]]), index // 20)
        assert int(first["x"].max()) < cell.sensor_size[0] and int(first["y"].max()) < cell.sensor_size[1]


SMALL = (7, 5)


def _stream(config: str, seed: int) -> list[np.ndarray]:
    """Events over a few bins of *config*, split into calls, timestamps in arrival order."""
    _, bins, bin_us = TEMPORAL_KERNEL_CONFIGS[config]
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 300))
    events = np.zeros(n, dtype=EVENT_DTYPE)
    events["t"] = np.sort(rng.integers(0, (bins + 3) * bin_us, size=n))
    events["x"] = rng.integers(0, SMALL[0], size=n)
    events["y"] = rng.integers(0, SMALL[1], size=n)
    events["p"] = rng.integers(0, 3, size=n)
    cuts = np.sort(rng.integers(0, n + 1, size=3))
    return np.split(events, cuts)


@pytest.mark.parametrize("config", list(TEMPORAL_KERNEL_CONFIGS))
@pytest.mark.parametrize("seed", range(6))
def test_the_reference_agrees_with_the_independent_oracle(config: str, seed: int) -> None:
    kind, bins, bin_us = TEMPORAL_KERNEL_CONFIGS[config]
    calls = _stream(config, seed)
    oracle = TemporalReference(kind, SMALL, bins=bins, bin_us=bin_us)
    for call in calls:
        oracle.accumulate(call)
    expected = oracle.read()
    got = reference(config, SMALL, calls)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert (got.view(np.uint8) == expected.view(np.uint8)).all()


SMALL_CELLS = [
    Cell(config, (33, 21), batch, interval, distribution, seed=3)
    for config in TEMPORAL_KERNEL_CONFIGS
    for batch, interval in ((10_000, 16.0), (100_000, 0.0))
    for distribution in GATE_DISTRIBUTIONS
]


class TestTargets:
    @pytest.mark.parametrize("cell", SMALL_CELLS, ids=lambda c: c.label)
    @pytest.mark.parametrize("name", ["temporal-kernel", "temporal-engine", "temporal-planes"])
    def test_real_kernels_pass_their_result_checks(self, name: str, cell: Cell) -> None:
        _, _, checks = _measure(name, cell, Policy(timed_calls=12))
        assert checks["valid"], checks["failures"]
        assert set(checks["power"]) == {"start", "end"}

    @pytest.mark.parametrize("name", ["temporal-kernel", "temporal-engine"])
    def test_a_call_that_did_not_accumulate_is_caught(self, name: str) -> None:
        cell = Cell("voxel_grid_5x12500", (40, 30), 100_000, 0.0, "uniform", seed=2)
        batches = cell.workload().batches(5)
        prepared = TARGETS[name]().prepare(cell, batches)
        assert prepared.before_call and prepared.after_call and prepared.finish
        for i, batch in enumerate(batches):
            prepared.before_call()
            if i != 2:
                prepared.call(batch)
            prepared.after_call()
        assert not prepared.finish()["valid"]

    def test_the_plane_hook_times_each_call_and_is_removed_after_it(self) -> None:
        original = temporal_module._clear_planes
        cell = Cell("stacked_histogram_15x3333", (40, 30), 100_000, 0.0, "uniform", seed=2)
        document = run_suite("temporal-gate", (cell,), TARGETS["temporal-planes"](),
                             Policy(runs=1, timed_calls=4, memory_calls=0), progress=lambda _: None,
                             process_per_run=False)
        assert temporal_module._clear_planes is original
        (record,) = document["cells"]
        (checks,) = record["checks"]
        assert checks["valid"] and checks["plane_clearing_function"] == "frames2py.kernels._temporal._clear_planes"
        assert len(checks["plane_clearing_ns"]) == 5 and all(ns > 0 for ns in checks["plane_clearing_ns"])
        (share,) = plane_share(record, warmup_calls=1)
        assert 0 < share < 1

    def test_state_size_is_recorded(self) -> None:
        cell = Cell("voxel_grid_15x3571", (40, 30), 10_000, 16.0, "uniform", seed=2)
        for name in ("temporal-kernel", "temporal-engine"):
            prepared = TARGETS[name]().prepare(cell, cell.workload().batches(2))
            assert prepared.details["state_bytes"] == 2 * 15 * 40 * 30 * 8


AC = {"source": "AC Power", "low_power_mode": False}


@functools.cache
def _one_run_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = Policy(runs=1, timed_calls=1, memory_calls=0)
    return tuple(  # type: ignore[return-value]
        run_suite("temporal-gate", temporal_gate_cells(), _FakeTarget(level, every_kernel=True), policy,
                  progress=lambda _: None, process_per_run=False)
        for level in ("kernel", "engine")
    )


def _documents(rate: float = 30e6) -> tuple[dict[str, Any], dict[str, Any]]:
    """Five-run documents for every temporal gate cell, all environment rules met."""
    out = []
    for template in _one_run_documents():
        document = copy.deepcopy(template)
        document["policy"]["runs"] = 5
        for env in ("environment", "environment_end"):
            document[env].update(tracked_changes=False, untracked_files=[], commit="c" * 40, power=dict(AC),
                                 **REFERENCE_MACHINE)
        for record in document["cells"]:
            record["checks"] = [{"valid": True, "power": {"start": dict(AC), "end": dict(AC)}}] * 5
            record["runtime"] = [{**record["runtime"][0], "src_tree": "t" * 40}] * 5
            record["workload_sha256"] = record["workload_sha256"] * 5
            record["call_ns"] = record["call_ns"] * 5
        _set_rates(document, {c.condition: _uniform(rate) for c in temporal_gate_cells()})
        out.append(document)
    return out[0], out[1]


class TestClassification:
    def test_documents_are_classified_against_the_temporal_cells(self) -> None:
        kernel, engine = _documents()
        result = gate.verdicts(kernel, engine)
        assert [Cell.from_record(c) for c in result["cells"]] == list(temporal_gate_cells())
        assert result["gate"] == "MET" and result["src_tree"] == "t" * 40

    def test_a_miss_is_reported_by_cell(self) -> None:
        kernel, engine = _documents()
        target = temporal_gate_cells()[7]
        _set_rates(engine, {target.condition: _uniform(15e6)})
        result = gate.verdicts(kernel, engine)
        assert [c["label"] for c in result["cells"] if c["verdict"] == "NOT MET"] == [target.label]

    @pytest.mark.parametrize(
        ("where", "power"),
        [("environment", {"source": "Battery Power", "low_power_mode": False}),
         ("environment_end", {"source": "AC Power", "low_power_mode": True})],
    )
    def test_a_document_off_ac_or_in_low_power_mode_is_invalid_for_its_environment(
        self, where: str, power: dict[str, Any]
    ) -> None:
        kernel, engine = _documents()
        engine[where]["power"] = power
        result = gate.verdicts(kernel, engine)
        assert {c["verdict"] for c in result["cells"]} == {"INVALID"}
        assert all(c["engine_level"]["environment_only"] for c in result["cells"])

    def test_a_run_off_ac_invalidates_only_its_cell(self) -> None:
        kernel, engine = _documents()
        record = kernel["cells"][4]
        record["checks"] = [dict(c) for c in record["checks"]]
        record["checks"][3] = {"valid": True, "power": {"start": dict(AC), "end": {"source": "Battery Power",
                                                                                   "low_power_mode": False}}}
        result = gate.verdicts(kernel, engine)
        assert [c["verdict"] for c in result["cells"]].count("INVALID") == 1
        assert result["cells"][4]["kernel_level"]["environment_only"] is True

    def test_a_failed_result_check_is_not_an_environment_problem(self) -> None:
        kernel, engine = _documents()
        kernel["cells"][0]["checks"] = [dict(c) for c in kernel["cells"][0]["checks"]]
        kernel["cells"][0]["checks"][0]["valid"] = False
        result = gate.verdicts(kernel, engine)
        assert result["cells"][0]["kernel_level"]["environment_only"] is False

    def test_runs_must_agree_on_the_src_tree(self) -> None:
        kernel, engine = _documents()
        record = engine["cells"][9]
        record["runtime"] = [dict(r) for r in record["runtime"]]
        record["runtime"][2]["src_tree"] = "u" * 40
        with pytest.raises(ValueError, match="src/ tree"):
            gate.verdicts(kernel, engine)

    def test_a_document_must_end_on_a_clean_tree(self) -> None:
        kernel, engine = _documents()
        kernel["environment_end"]["untracked_files"] = [{"path": "x.py", "sha256": "0"}]
        with pytest.raises(ValueError, match="clean working tree"):
            gate.verdicts(kernel, engine)

    def test_the_gate_command_classifies_temporal_documents(self, tmp_path: Any, capsys: Any) -> None:
        kernel, engine = _documents()
        paths = {}
        for name, document in (("kernel", kernel), ("engine", engine)):
            paths[name] = tmp_path / f"{name}.json"
            results.write(paths[name], document)
        assert main(["gate", "--kernel-level", str(paths["kernel"]), "--engine-level", str(paths["engine"]),
                     "--out", str(tmp_path / "gate.json")]) == 0
        verdicts = json.loads((tmp_path / "gate.json").read_text())
        assert verdicts["suite"] == "temporal-gate" and verdicts["gate"] == "MET"
        assert len(verdicts["cells"]) == 150 and "Gate: MET" in capsys.readouterr().out
