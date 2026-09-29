"""The benchmark suite's own behaviour: definitions, workloads, measurement,
results and interpretation. No timing assertions: these must pass on any machine.
"""

from __future__ import annotations

import copy
import functools
import gc
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frames2py import EVENT_DTYPE

from benchmarks import environment, gate, power, report, results
from benchmarks.__main__ import main
from benchmarks.matrix import (
    GATE_THRESHOLD_EVENTS_PER_S,
    V1_KERNELS,
    Cell,
    gate_cells,
    prototype_baseline_cells,
)
from benchmarks.measure import Policy, measure_memory, nearest_rank, summarize, time_calls
from benchmarks.runner import run_suite, timed_calls_for
from benchmarks.targets import TARGETS, Level, Prepared
from benchmarks.targets.v1 import engine_timed_calls, publication_schedule
from benchmarks.workloads import Workload
from tests.fake_power import FakeMac


class TestGateDefinition:
    def test_150_distinct_cells(self) -> None:
        cells = gate_cells()
        assert len(cells) == 150
        assert len({cell.condition for cell in cells}) == 150
        assert all(cell.in_gate for cell in cells)

    def test_dimensions_match_the_gate_definition(self) -> None:
        cells = gate_cells()
        assert {c.kernel for c in cells} == set(V1_KERNELS) == {
            "event_count", "polarity", "time_surface", "exp_decay", "timestamp_decay"
        }
        assert {c.sensor_size for c in cells} == {(346, 260), (640, 480), (1280, 720)}
        assert {(c.batch_size, c.interval_ms) for c in cells} == {
            (10_000, 16.0), (100_000, 0.0), (100_000, 16.0), (1_000_000, 0.0), (1_000_000, 16.0)
        }
        assert {c.distribution for c in cells} == {"uniform", "clustered"}
        assert GATE_THRESHOLD_EVENTS_PER_S == 20_000_000

    def test_10k_at_0ms_is_not_a_gate_condition(self) -> None:
        cell = Cell("event_count", (346, 260), 10_000, 0.0, "uniform", seed=1)
        assert not cell.in_gate
        assert not any(c.batch_size == 10_000 and c.interval_ms == 0.0 for c in gate_cells())

    def test_membership_ignores_the_seed(self) -> None:
        cell = gate_cells()[0]
        other = Cell(cell.kernel, cell.sensor_size, cell.batch_size, cell.interval_ms,
                     cell.distribution, seed=cell.seed + 1)
        assert other.in_gate

    def test_baseline_suite_marks_its_non_gate_cells(self) -> None:
        cells = prototype_baseline_cells()
        outside = [c for c in cells if not c.in_gate]
        assert len(cells) == 72
        assert {(c.batch_size, c.interval_ms) for c in outside} == {(10_000, 0.0)}
        assert "timestamp_decay" not in {c.kernel for c in cells}

    def test_same_input_for_every_kernel_and_interval(self) -> None:
        seeds: dict[Any, set[int]] = {}
        for c in gate_cells():
            seeds.setdefault((c.sensor_size, c.batch_size, c.distribution), set()).add(c.seed)
        assert all(len(s) == 1 for s in seeds.values())
        assert len({s.pop() for s in seeds.values()}) == len(seeds)


class TestWorkloads:
    @pytest.mark.parametrize("distribution", ["uniform", "clustered"])
    def test_same_parameters_same_batches(self, distribution: str) -> None:
        first = Workload(distribution, (346, 260), 5_000, seed=3).batches(3)
        second = Workload(distribution, (346, 260), 5_000, seed=3).batches(3)
        other = Workload(distribution, (346, 260), 5_000, seed=4).batches(3)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        assert not np.array_equal(first[0], other[0])

    @pytest.mark.parametrize("distribution", ["uniform", "clustered"])
    @pytest.mark.parametrize("sensor", [(346, 260), (1280, 720), (1, 1), (3, 700)])
    def test_valid_in_bounds_events(self, distribution: str, sensor: tuple[int, int]) -> None:
        for batch in Workload(distribution, sensor, 20_000, seed=9).batches(2):
            assert batch.dtype == EVENT_DTYPE
            assert batch.flags["C_CONTIGUOUS"] and batch.ndim == 1
            assert int(batch["x"].max()) < sensor[0]
            assert int(batch["y"].max()) < sensor[1]
            assert set(np.unique(batch["p"]).tolist()) <= {0, 1}

    def test_timestamps_continue_across_batches(self) -> None:
        batches = Workload("uniform", (346, 260), 1_000, seed=1).batches(3)
        t = np.concatenate([b["t"] for b in batches])
        np.testing.assert_array_equal(t, np.arange(3_000, dtype=np.uint64))
        fast = Workload("uniform", (346, 260), 1_000, seed=1, event_rate_hz=20_000_000).batches(3)
        t = np.concatenate([b["t"] for b in fast])
        assert np.all(np.diff(t.astype(np.int64)) >= 0)
        assert int(t[-1]) == (2_999 * 1_000_000) // 20_000_000

    def test_clustered_events_are_localized(self) -> None:
        sensor, n = (346, 260), 100_000
        uniform = Workload("uniform", sensor, n, seed=5).batches(4)
        clustered = Workload("clustered", sensor, n, seed=5).batches(4)

        def pixels(batches: list[np.ndarray]) -> int:
            return len({(int(x), int(y)) for b in batches for x, y in zip(b["x"], b["y"])})

        assert pixels(clustered) < 0.1 * pixels(uniform)

    def test_batch_size_does_not_move_the_clusters(self) -> None:
        small = Workload("clustered", (640, 480), 1_000, seed=2)
        large = Workload("clustered", (640, 480), 50_000, seed=2)
        assert np.array_equal(small._centres(), large._centres())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"distribution": "gaussian"},
            {"sensor_size": (0, 10)},
            {"batch_size": -1},
            {"event_rate_hz": 0},
            {"distribution": "clustered", "clusters": 0},
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs: dict[str, Any]) -> None:
        base: dict[str, Any] = {"distribution": "uniform", "sensor_size": (10, 10), "batch_size": 1, "seed": 0}
        with pytest.raises(ValueError):
            Workload(**{**base, **kwargs})


class TestMeasurement:
    def test_nearest_rank(self) -> None:
        samples = [15, 20, 35, 40, 50]
        assert [nearest_rank(samples, q) for q in (5, 30, 40, 50, 100)] == [15, 20, 20, 35, 50]
        assert nearest_rank([3, 1, 2, 7, 5, 4, 6], 99) == 7
        with pytest.raises(ValueError):
            nearest_rank([], 50)

    def test_summary_is_the_median_of_run_medians(self) -> None:
        runs = [[1_000, 3_000, 2_000], [4_000, 4_000, 5_000], [1_000, 1_000, 1_000]]
        summary = summarize(10_000, runs)
        assert summary["run_median_ns"] == [2_000, 4_000, 1_000]
        assert summary["median_ns"] == 2_000
        assert summary["events_per_s"] == pytest.approx(10_000 / 2e-6)
        assert summary["events_per_s_run_range"] == pytest.approx([10_000 / 4e-6, 10_000 / 1e-6])
        assert summary["latency_ns"]["samples"] == 9
        assert summary["latency_ns"]["p50"] == 2_000
        assert summary["latency_ns"]["max"] == 5_000

    def test_sustained_throughput_counts_every_call(self) -> None:
        # One slow call in four: invisible to the median, not to the sustained rate.
        runs = [[1_000, 1_000, 1_000, 9_000]]
        assert summarize(10, runs)["events_per_s"] == pytest.approx(10 / 1e-6)
        sustained = summarize(10, runs, "sustained")
        assert sustained["events_per_s"] == pytest.approx(40 / 12e-6)
        assert sustained["statistic"] == "sustained"
        with pytest.raises(ValueError):
            summarize(10, runs, "mean")

    def test_statistic_is_the_median_of_per_run_values(self) -> None:
        runs = [[1_000], [2_000], [4_000], [8_000]]
        assert summarize(1, runs)["events_per_s"] == pytest.approx((1 / 2e-6 + 1 / 4e-6) / 2)

    def test_hooks_run_around_every_call_outside_the_timed_interval(self) -> None:
        events: list[str] = []
        batches = [np.full(1, i, dtype=EVENT_DTYPE) for i in range(4)]

        def call(batch: np.ndarray) -> None:
            events.append(f"call {int(batch['t'][0])}")

        span: list[int] = []
        samples, _ = time_calls(call, batches, 1, 3, before_call=lambda: events.append("before"),
                                after_call=lambda: events.append("after"), span=span)
        assert events == ["before", "call 0", "after"] + [
            e for i in range(1, 4) for e in ("before", f"call {i}", "after")
        ]
        assert len(samples) == 3 and len(span) == 1 and span[0] >= sum(samples)

    def test_warmup_is_discarded_and_counters_cover_timed_calls_only(self) -> None:
        seen: list[int] = []
        batches = [np.full(1, i, dtype=EVENT_DTYPE) for i in range(6)]

        def call(batch: np.ndarray) -> None:
            seen.append(int(batch["t"][0]))

        samples, deltas = time_calls(call, batches, warmup_calls=2, timed_calls=3,
                                     counters=lambda: {"calls": len(seen)})
        assert seen == [0, 1, 2, 3, 4]
        assert len(samples) == 3
        assert deltas == {"calls": 3}
        assert gc.isenabled()

    def test_too_few_batches_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            time_calls(lambda b: None, [np.empty(0, dtype=EVENT_DTYPE)], 1, 1)

    def test_policy_needs_a_timed_call_count_for_unknown_sizes(self) -> None:
        assert Policy().timed_calls_for(1_000_000) == 7
        assert Policy(timed_calls=3).timed_calls_for(12_345) == 3
        with pytest.raises(ValueError):
            Policy().timed_calls_for(12_345)

    def test_memory_sees_temporary_and_retained_allocations(self) -> None:
        batches = [np.empty(0, dtype=EVENT_DTYPE)] * 4
        kept: list[np.ndarray] = []

        def temporary(batch: np.ndarray) -> None:
            np.ones(2**20, dtype=np.float64).sum()  # 8 MiB, freed before return

        def retaining(batch: np.ndarray) -> None:
            kept.append(np.ones(2**17, dtype=np.float64))  # 1 MiB kept

        temp = measure_memory(temporary, batches, warmup_calls=1, memory_calls=3)
        assert min(temp["peak_temporary_bytes"]) >= 8 * 2**20
        assert abs(temp["retained_growth_bytes"]) < 2**20
        grow = measure_memory(retaining, batches, warmup_calls=1, memory_calls=3)
        assert grow["retained_growth_bytes"] >= 3 * 2**20


class _FakeTarget:
    """Sums x over each batch; supports every kernel except timestamp_decay, unless
    *every_kernel*."""

    name = "fake"

    def __init__(self, level: Level = "kernel", every_kernel: bool = False) -> None:
        self.level = level
        self.statistic = "median_call" if level == "kernel" else "sustained"
        self.every_kernel = every_kernel
        self.prepared = 0

    def supports(self, cell: Cell) -> bool:
        return self.every_kernel or cell.kernel != "timestamp_decay"

    def timed_calls(self, cell: Cell, default: int, warmup_calls: int) -> int:
        return default

    def prepare(self, cell: Cell, batches: Any = ()) -> Prepared:
        self.prepared += 1
        total = [0]

        def call(batch: np.ndarray) -> None:
            total[0] += int(batch["x"].sum())

        def counters() -> Mapping[str, int]:
            return {"calls": total[0]}

        return Prepared(call=call, counters=counters, details={"fresh": True})

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "level": self.level, "statistic": self.statistic}


def _document(
    level: Level, cells: tuple[Cell, ...], policy: Policy, every_kernel: bool = False
) -> dict[str, Any]:
    return run_suite("test", cells, _FakeTarget(level, every_kernel), policy, progress=lambda _: None,
                     process_per_run=False)


class TestRunnerAndResults:
    CELLS = (
        Cell("event_count", (32, 24), 1_000, 0.0, "uniform", seed=1),
        Cell("timestamp_decay", (32, 24), 1_000, 0.0, "clustered", seed=2),
    )

    def test_unsupported_cells_are_recorded_without_numbers(self) -> None:
        document = _document("kernel", self.CELLS, Policy(runs=2, timed_calls=4, memory_calls=2))
        measured, unsupported = document["cells"]
        assert measured["status"] == "measured"
        assert [len(run) for run in measured["call_ns"]] == [4, 4]
        assert measured["summary"]["latency_ns"]["samples"] == 8
        assert measured["memory"]["calls"] == 2
        assert measured["workload"]["distribution"] == "uniform"
        assert unsupported == {**self.CELLS[1].to_record(), "in_gate": False, "status": "unsupported"}

    def test_every_run_gets_fresh_state(self) -> None:
        target = _FakeTarget()
        run_suite("test", self.CELLS, target, Policy(runs=3, timed_calls=2, memory_calls=1),
                  progress=lambda _: None, process_per_run=False)
        assert target.prepared == 3 + 1  # one per run, one for the memory pass

    def test_unregistered_targets_cannot_run_in_subprocesses(self) -> None:
        with pytest.raises(ValueError):
            run_suite("test", self.CELLS, _FakeTarget(), Policy(runs=1, timed_calls=1))

    def test_one_process_per_run(self) -> None:
        cells = (Cell("event_count", (32, 24), 1_000, 0.0, "uniform", seed=1),)
        document = run_suite("test", cells, TARGETS["v1-kernel"](),
                             Policy(runs=2, timed_calls=3, memory_calls=1), progress=lambda _: None)
        (measured,) = document["cells"]
        assert document["policy"]["process_per_run"] is True
        assert [len(run) for run in measured["call_ns"]] == [3, 3]
        assert measured["memory"]["calls"] == 1
        assert measured["details"]["output_dtype"] == "uint32"

    def test_environment_is_recorded(self) -> None:
        env = _document("kernel", self.CELLS[:1], Policy(runs=1, timed_calls=1, memory_calls=0))["environment"]
        for key in ("commit", "tracked_changes", "untracked_files", "chip", "memory_bytes", "os",
                    "python", "numpy", "power", "free_threaded_build"):
            assert key in env

    def test_documents_round_trip_and_are_never_overwritten(self, tmp_path: Path) -> None:
        document = _document("kernel", self.CELLS, Policy(runs=1, timed_calls=2, memory_calls=0))
        path = tmp_path / "out.json"
        results.write(path, document)
        assert results.read(path) == json.loads(json.dumps(document))
        with pytest.raises(FileExistsError):
            results.write(path, document)
        path.write_text("{}")
        with pytest.raises(ValueError):
            results.read(path)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "tracked.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


class TestProvenance:
    def test_clean_checkout(self, repo: Path) -> None:
        state = environment.git_state(repo)
        assert state["tracked_changes"] is False
        assert state["untracked_files"] == []
        assert state["commit"] is not None and len(state["commit"]) == 40
        assert environment.working_tree_clean(state) is True
        assert "clean working tree" in report._tree_state(state)

    def test_untracked_directory_is_not_clean(self, repo: Path) -> None:
        # The failure mode: new, never-committed code next to an unchanged commit.
        (repo / "benchmarks" / "targets").mkdir(parents=True)
        (repo / "benchmarks" / "targets" / "new.py").write_text("y = 2\n")
        (repo / "notes with space.txt").write_text("z\n")
        state = environment.git_state(repo)
        assert state["tracked_changes"] is False
        assert state["untracked_files"] == [
            {"path": "benchmarks/targets/new.py", "sha256": hashlib.sha256(b"y = 2\n").hexdigest()},
            {"path": "notes with space.txt", "sha256": hashlib.sha256(b"z\n").hexdigest()},
        ]
        assert environment.working_tree_clean(state) is False
        text = report._tree_state(state)
        assert "not reproducible from the commit alone" in text and "2 untracked file(s)" in text

    def test_ignored_files_do_not_count(self, repo: Path) -> None:
        (repo / "ignored").mkdir()
        (repo / "ignored" / "result.json").write_text("{}")
        state = environment.git_state(repo)
        assert state["untracked_files"] == []
        assert environment.working_tree_clean(state) is True

    def test_modified_tracked_file_is_not_clean(self, repo: Path) -> None:
        (repo / "tracked.py").write_text("x = 3\n")
        state = environment.git_state(repo)
        assert state["tracked_changes"] is True
        assert environment.working_tree_clean(state) is False

    def test_outside_a_repository_nothing_is_claimed(self, tmp_path: Path) -> None:
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        state = environment.git_state(outside)
        if state["commit"] is not None:
            pytest.skip("tmp_path is inside a git repository")
        assert state == {"commit": None, "tracked_changes": None, "untracked_files": None}
        assert environment.working_tree_clean(state) is None
        assert report._tree_state(state).startswith("working tree state unknown")

    @pytest.mark.parametrize(
        "env",
        [
            {"commit": "a" * 40, "tracked_changes": False},  # predates untracked-file tracking
            {"commit": "a" * 40, "tracked_changes": None, "untracked_files": []},
            {"commit": None, "tracked_changes": False, "untracked_files": []},
        ],
    )
    def test_incomplete_records_are_never_called_clean(self, env: dict[str, Any]) -> None:
        assert environment.working_tree_clean(env) is None
        assert "clean working tree" not in report._tree_state(env)


@functools.cache
def _one_run_gate_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = Policy(runs=1, timed_calls=1, memory_calls=0)
    kernel, engine = (_document(level, gate_cells(), policy, every_kernel=True) for level in ("kernel", "engine"))
    return kernel, engine


def _gate_documents(runs: int = 5) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kernel- and engine-level documents for every gate cell, from a clean tree."""
    documents = []
    for template in _one_run_gate_documents():
        document = copy.deepcopy(template)
        document["policy"]["runs"] = runs
        document["environment"].update(tracked_changes=False, untracked_files=[])
        for record in document["cells"]:
            for key in ("checks", "runtime", "workload_sha256"):
                record[key] = record[key] * runs
            record["call_ns"] = record["call_ns"] * runs
        documents.append(document)
    return documents[0], documents[1]


def _set_rates(document: dict[str, Any], rates: dict[Any, list[float]]) -> None:
    """Give cells per-run throughputs, as one-call runs with the matching time."""
    for record in document["cells"]:
        condition = Cell.from_record(record).condition
        if condition in rates:
            record["call_ns"] = [[round(record["batch_size"] / r * 1e9)] for r in rates[condition]]
            n = len(rates[condition])
            for key in ("checks", "runtime", "workload_sha256"):
                record[key] = record[key][:1] * n


def _uniform(rate: float, runs: int = 5) -> list[float]:
    return [rate] * runs


class TestGateClassification:
    def test_stage_one_band_edges(self) -> None:
        assert gate.stage1(22.0e6) == "clear_pass"
        assert gate.stage1(21.99e6) == "borderline"
        assert gate.stage1(18.0e6) == "borderline"
        assert gate.stage1(17.99e6) == "clear_miss"

    def test_stage_two_needs_the_median_and_18_of_20_runs(self) -> None:
        assert gate.stage2([20.5e6] * 18 + [19e6] * 2) == "stage2_pass"
        assert gate.stage2([20.5e6] * 17 + [19e6] * 3) == "not_met"
        assert gate.stage2([19.9e6] * 20) == "not_met"
        assert gate.stage2([20.0e6] * 20) == "stage2_pass"

    def test_a_cell_passes_only_when_both_levels_pass(self) -> None:
        kernel, engine = _gate_documents()
        cells = gate_cells()
        everything = {c.condition: _uniform(30e6) for c in cells}
        _set_rates(kernel, everything)
        _set_rates(engine, {**everything, cells[0].condition: _uniform(10e6)})
        result = gate.verdicts(kernel, engine)
        assert result["cells"][0]["verdict"] == "NOT MET"
        assert result["counts"] == {"PASS": 149, "NOT MET": 1, "INVALID": 0, "INCOMPLETE": 0}
        assert result["gate"] == "NOT MET"

    def test_borderline_is_not_a_pass_until_stage_two(self) -> None:
        kernel, engine = _gate_documents()
        cells = gate_cells()
        everything = {c.condition: _uniform(30e6) for c in cells}
        _set_rates(kernel, everything)
        _set_rates(engine, {**everything, cells[3].condition: _uniform(21e6)})
        result = gate.verdicts(kernel, engine)
        assert result["cells"][3]["engine_level"]["stage1"] == "borderline"
        assert result["cells"][3]["verdict"] == "INCOMPLETE"
        assert result["gate"] == "INCONCLUSIVE"
        assert [c.condition for c in gate.borderline_cells(engine)] == [cells[3].condition]

        second, _ = _gate_documents(runs=20)
        second.update(target=engine["target"], policy={**engine["policy"], "runs": 20})
        second["cells"] = [r for r in second["cells"] if Cell.from_record(r).condition == cells[3].condition]
        _set_rates(second, {cells[3].condition: [20.5e6] * 18 + [19e6] * 2})
        passed = gate.verdicts(kernel, engine, engine_stage2=second)
        assert passed["cells"][3]["verdict"] == "PASS" and passed["gate"] == "MET"
        _set_rates(second, {cells[3].condition: [20.5e6] * 17 + [19e6] * 3})
        assert gate.verdicts(kernel, engine, engine_stage2=second)["cells"][3]["verdict"] == "NOT MET"

    def test_verdict_cells_identify_their_gate_cell(self) -> None:
        kernel, engine = _gate_documents()
        _set_rates(kernel, {c.condition: _uniform(30e6) for c in gate_cells()})
        _set_rates(engine, {c.condition: _uniform(30e6) for c in gate_cells()})
        result = json.loads(json.dumps(gate.verdicts(kernel, engine)))
        assert [Cell.from_record(c) for c in result["cells"]] == list(gate_cells())
        assert all(c["kernel_level"]["result"] == c["engine_level"]["result"] == "pass" for c in result["cells"])

    def test_stage_two_may_only_measure_borderline_cells(self) -> None:
        kernel, engine = _gate_documents()
        _set_rates(engine, {c.condition: _uniform(30e6) for c in gate_cells()})
        second, _ = _gate_documents(runs=20)
        second.update(target=engine["target"], policy={**engine["policy"], "runs": 20})
        with pytest.raises(ValueError, match="not borderline"):
            gate.level_results(engine, second)

    def test_a_failed_check_makes_the_cell_invalid_whatever_its_speed(self) -> None:
        kernel, engine = _gate_documents()
        everything = {c.condition: _uniform(30e6) for c in gate_cells()}
        _set_rates(kernel, everything)
        _set_rates(engine, everything)
        engine["cells"][0]["checks"][2] = {"valid": False, "failures": ["publications at calls ..."]}
        result = gate.verdicts(kernel, engine)
        assert result["cells"][0]["verdict"] == "INVALID"
        assert result["gate"] == "INCONCLUSIVE"

    def test_runs_on_another_runtime_are_invalid(self) -> None:
        kernel, engine = _gate_documents()
        record = kernel["cells"][0]
        record["runtime"] = [dict(r) for r in record["runtime"]]
        record["runtime"][1]["gil_enabled"] = not record["runtime"][1]["gil_enabled"]
        assert gate.level_results(kernel)[Cell.from_record(record).condition]["result"] == "invalid"

    def test_levels_must_come_from_the_same_clean_commit_and_runtime(self) -> None:
        kernel, engine = _gate_documents()
        with pytest.raises(ValueError):
            gate.verdicts(engine, kernel)
        engine["environment"]["untracked_files"] = [{"path": "x.py", "sha256": "0"}]
        with pytest.raises(ValueError, match="clean working tree"):
            gate.verdicts(kernel, engine)

    def test_classification_uses_raw_call_times_not_summaries(self) -> None:
        kernel, engine = _gate_documents()
        _set_rates(kernel, {c.condition: _uniform(10e6) for c in gate_cells()})
        for record in kernel["cells"]:
            record["summary"]["events_per_s"] = 1e12
        assert gate.level_results(kernel)[gate_cells()[0].condition]["stage1"] == "clear_miss"

    def test_off_reference_results_do_not_count(self) -> None:
        kernel, engine = _gate_documents()
        for document in (kernel, engine):
            document["environment"]["chip"] = "Some Other CPU"
        assert gate.verdicts(kernel, engine)["gate"] == "NOT ON THE REFERENCE MACHINE"

    def test_table_reports_every_cell_with_its_conditions(self) -> None:
        cells = tuple(c for c in gate_cells() if c.kernel == "event_count" and c.sensor_size == (346, 260))
        text = report.table(_document("kernel", cells, Policy(runs=1, timed_calls=1, memory_calls=0)))
        assert text.startswith("Conditions: fake (kernel level")
        assert text.count("\n| event_count 346x260") == 10


def _measure(
    target_name: str, cell: Cell, policy: Policy | None = None
) -> tuple[Prepared, list[int], dict[str, Any]]:
    """One timed run of *cell*, as the runner does it, and the target's checks."""
    target = TARGETS[target_name]()
    policy = policy or Policy(timed_calls=6)
    timed = timed_calls_for(cell, target, policy)
    batches = cell.workload().batches(policy.warmup_calls + timed)
    prepared = target.prepare(cell, batches)
    samples, _ = time_calls(prepared.call, batches, policy.warmup_calls, timed, prepared.counters,
                            prepared.before_call, prepared.after_call)
    assert prepared.finish is not None
    return prepared, samples, dict(prepared.finish())


V1_SMALL_CELLS = [
    Cell(kernel, (33, 21), batch, interval, distribution, seed=3)
    for kernel in V1_KERNELS
    for batch, interval in ((1_000, 16.0), (2_000, 0.0))
    for distribution in ("uniform", "clustered")
]


class TestV1Targets:
    @pytest.mark.parametrize("cell", V1_SMALL_CELLS, ids=lambda c: c.label)
    @pytest.mark.parametrize("name", ["v1-kernel", "v1-engine"])
    def test_real_kernels_pass_their_result_checks(self, name: str, cell: Cell) -> None:
        _, _, checks = _measure(name, cell)
        assert checks["valid"], checks["failures"]

    def test_timed_calls_give_at_least_10_publications_at_16_ms(self) -> None:
        def count(batch: int, interval: float) -> int:
            cell = Cell("event_count", (346, 260), batch, interval, "uniform", seed=1)
            return engine_timed_calls(cell, Policy().timed_calls_for(batch), warmup_calls=1)

        assert (count(10_000, 16.0), count(100_000, 16.0), count(1_000_000, 16.0)) == (320, 40, 10)
        assert (count(100_000, 0.0), count(1_000_000, 0.0)) == (20, 7)
        for batch, calls in ((10_000, 320), (100_000, 40), (1_000_000, 10)):
            assert sum(publication_schedule(1 + calls, batch, 16.0)[1:]) == 10

    def test_engine_publishes_on_the_virtual_cadence(self) -> None:
        cell = Cell("polarity", (40, 30), 10_000, 16.0, "uniform", seed=2)
        _, samples, checks = _measure("v1-engine", cell, Policy())
        assert len(samples) == 320
        assert checks["valid"], checks["failures"]
        assert checks["publication_calls"] == list(range(0, 321, 32))
        assert [p["sequence"] for p in checks["publications"]] == list(range(1, 12))

    def test_virtual_clock_does_not_leak_to_other_engines(self) -> None:
        import time as time_module

        import frames2py._engine as engine_module

        _measure("v1-engine", Cell("event_count", (40, 30), 1_000, 16.0, "uniform", seed=2))
        assert engine_module.time is time_module

    def test_a_clock_that_does_not_advance_is_caught(self) -> None:
        cell = Cell("event_count", (40, 30), 10_000, 16.0, "uniform", seed=2)
        target = TARGETS["v1-engine"]()
        batches = cell.workload().batches(41)
        prepared = target.prepare(cell, batches)
        assert prepared.after_call is not None and prepared.finish is not None
        for batch in batches:  # never lets the Engine see the virtual clock
            prepared.call(batch)
            prepared.after_call()
        assert not prepared.finish()["valid"]

    def test_worker_processes_record_their_runtime_and_checks(self) -> None:
        cell = Cell("exp_decay", (32, 24), 1_000, 0.0, "clustered", seed=1)
        document = run_suite("test", (cell,), TARGETS["v1-engine"](),
                             Policy(runs=2, timed_calls=3, memory_calls=1), progress=lambda _: None)
        (record,) = document["cells"]
        assert [c["valid"] for c in record["checks"]] == [True, True]
        assert all(r["gil_enabled"] == document["environment"]["gil_enabled"] for r in record["runtime"])
        assert len(set(record["workload_sha256"])) == 1
        assert record["summary"]["statistic"] == "sustained"
        assert record["memory"]["calls"] == 1

    @pytest.mark.parametrize("name", ["v1-kernel", "v1-engine"])
    def test_a_call_that_did_not_accumulate_is_caught(self, name: str) -> None:
        cell = Cell("time_surface", (40, 30), 1_000, 0.0, "uniform", seed=2)
        batches = cell.workload().batches(4)
        prepared = TARGETS[name]().prepare(cell, batches)
        assert prepared.before_call and prepared.after_call and prepared.finish
        for i, batch in enumerate(batches):
            prepared.before_call()
            if i != 2:
                prepared.call(batch)
            prepared.after_call()
        assert not prepared.finish()["valid"]


class TestPowerGuard:
    @pytest.mark.parametrize("caps", [0x1, 0x9, 0x0, None], ids=["cpu-only", "darkwake-net", "none", "unknown"])
    def test_refuses_outside_full_wake_without_taking_an_assertion(self, caps: int | None) -> None:
        backend = FakeMac(caps)
        with pytest.raises(power.PowerStateError, match="full wake"):
            with power.hold_awake(backend=backend, platform="darwin"):
                pytest.fail("the block must not run")
        assert backend.created == 0

    def test_refuses_when_the_assertion_cannot_be_taken(self) -> None:
        with pytest.raises(power.PowerStateError):
            with power.hold_awake(backend=FakeMac(0x1F, create_fails=True), platform="darwin"):
                pytest.fail("the block must not run")

    def test_refuses_and_releases_when_the_assertion_cannot_be_confirmed(self) -> None:
        backend = FakeMac(0x1F, confirms=False)
        with pytest.raises(power.PowerStateError, match="confirmed"):
            with power.hold_awake(backend=backend, platform="darwin"):
                pytest.fail("the block must not run")
        assert backend.held == set()

    def test_holds_for_the_block_and_releases_after_it_even_on_error(self) -> None:
        backend = FakeMac(0x1F)
        with pytest.raises(ZeroDivisionError):
            with power.hold_awake(backend=backend, platform="darwin") as record:
                assert backend.held == {1}
                1 / 0
        assert backend.held == set()
        assert record["assertion"]["released"] is True and record["end"]["full_wake"] is True

    def test_other_platforms_never_refuse(self) -> None:
        with power.hold_awake(platform="linux") as record:
            pass
        assert record["assertion"] is None and "slept" in record

    def test_sleep_during_the_block_is_recorded_and_voids_gate_cells(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kernel, _ = _gate_documents()
        _set_rates(kernel, {c.condition: _uniform(30e6) for c in gate_cells()})
        clocks = iter([(0, 0), (60_000_000_000, 5_000_000_000)])  # 55 s asleep
        monkeypatch.setattr(power, "_clock_pair", lambda: next(clocks))
        with power.hold_awake(platform="linux") as record:
            pass
        assert record["slept"] is True and record["slept_ns"] == 55_000_000_000
        kernel["power"] = record
        assert all(r["result"] == "invalid" for r in gate.level_results(kernel).values())
        del kernel["power"]  # documents from before power records keep their classification
        assert all(r["result"] == "pass" for r in gate.level_results(kernel).values())

    def test_full_wake_at_start_and_end_keeps_the_classification(self) -> None:
        kernel, _ = _gate_documents()
        _set_rates(kernel, {c.condition: _uniform(30e6) for c in gate_cells()})
        with power.hold_awake(backend=FakeMac(0x1F), platform="darwin") as record:
            pass
        assert record["slept"] is False and not power.ended_outside_full_wake(record)
        kernel["power"] = record
        assert all(r["result"] == "pass" for r in gate.level_results(kernel).values())

    def test_ending_in_darkwake_voids_gate_cells_without_any_recorded_sleep(self) -> None:
        kernel, _ = _gate_documents()
        _set_rates(kernel, {c.condition: _uniform(30e6) for c in gate_cells()})
        backend = FakeMac(0x1F)
        with power.hold_awake(backend=backend, platform="darwin") as record:
            backend.caps = 0x9  # DarkWake: CPU without graphics
        assert record["slept"] is False and power.ended_outside_full_wake(record)
        kernel["power"] = record
        assert all(r["result"] == "invalid" for r in gate.level_results(kernel).values())
        assert any("ended outside full wake" in reason for reason in gate.problems(kernel["cells"][0], kernel, 5))

    def test_platforms_without_a_wake_state_keep_the_classification(self) -> None:
        kernel, _ = _gate_documents()
        _set_rates(kernel, {c.condition: _uniform(30e6) for c in gate_cells()})
        with power.hold_awake(platform="linux") as record:
            pass
        kernel["power"] = record
        assert all(r["result"] == "pass" for r in gate.level_results(kernel).values())

    @pytest.mark.skipif(sys.platform != "darwin", reason="IOKit power assertions are macOS only")
    def test_real_assertion_is_held_and_released(self) -> None:
        backend = power.MacBackend()
        if not power.state(backend)["full_wake"]:
            pytest.skip("not in full wake; the guard would refuse")
        created: list[int] = []
        create = backend.create_assertion

        def spy(name: str) -> int:
            created.append(create(name))
            return created[-1]

        backend.create_assertion = spy  # type: ignore[method-assign]
        with power.hold_awake("frames2py test", backend=backend) as record:
            assert backend.assertion_properties(created[0])["AssertLevel"] == power.ASSERTION_LEVEL_ON
        assert backend.assertion_properties(created[0]) == {}
        assert record["slept"] is False

    def test_documents_record_the_power_state(self) -> None:
        document = _document("kernel", TestRunnerAndResults.CELLS[:1], Policy(runs=1, timed_calls=1, memory_calls=0))
        assert set(document["power"]) >= {"platform", "start", "end", "slept_ns", "slept", "assertion"}
        env_power = document["environment"]["power"]
        expected = {"system_capabilities", "full_wake", "pmset"} if sys.platform == "darwin" else set()
        assert expected <= set(env_power)
        if sys.platform == "darwin":
            assert document["power"]["assertion"]["released"] is True


def test_cli_lists_the_gate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--suite", "gate"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("150 cells, 150 in the hard gate")
    assert main(["list", "--suite", "prototype-baseline", "--batch-size", "10000", "--interval", "0"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("12 cells, 0 in the hard gate")


class TestAdapterCharacterisation:
    """The adapter benchmark measures what it says, on a committed fixture, in this process."""

    def test_document_records_conditions_counts_and_rates(self) -> None:
        from benchmarks import adapters

        fixture = Path(__file__).resolve().parent / "data" / "sparklers_100k.evt2.raw"
        document = adapters.run("sparklers_100k.evt2.raw", runs=2, process_per_run=False, path=fixture, adapter="evt",
                                open_kwargs={"sensor_size": (640, 480)}, expected_events=100_000)
        assert document["schema"] == adapters.SCHEMA and document["policy"] == {"runs": 2, "process_per_run": False}
        assert document["backends"]["numpy"] == np.__version__
        for run in document["runs"]:
            assert run["events"] == 100_000 and run["checks"] == {"events_match": True}
            assert (run["t_min"], run["t_max"]) == (913_716_224, 913_728_417)
            assert run["file_bytes"] == fixture.stat().st_size
            assert run["snapshots_published"]["ingest"] >= 1
        assert "decode_peak_traced_bytes" in document["runs"][-1]
        per_run = document["per_run"][0]
        span_s = (913_728_417 - 913_716_224) / 1e6
        assert per_run["recording_events_per_s"] == pytest.approx(100_000 / span_s)
        assert per_run["real_time_factor"] == pytest.approx(span_s / (document["runs"][0]["decode_ns"] / 1e9))
        assert "via frames2py.adapters.evt" in adapters.report(document)

    def test_a_wrong_event_count_makes_the_document_invalid(self) -> None:
        from benchmarks import adapters

        fixture = Path(__file__).resolve().parent / "data" / "active_marker_head.evt3.raw"
        document = adapters.run("active_marker_head", runs=1, process_per_run=False, path=fixture, adapter="evt",
                                open_kwargs={}, expected_events=1)
        assert document["valid"] is False and "INVALID" in adapters.report(document)

    def test_batch_size_reaches_the_reader(self) -> None:
        from benchmarks import adapters

        fixture = Path(__file__).resolve().parent / "data" / "active_marker_head.evt3.raw"
        document = adapters.run("active_marker_head", runs=1, process_per_run=False, path=fixture, adapter="evt",
                                open_kwargs={}, expected_events=46_893, batch_size=10_000)
        assert document["open_kwargs"]["batch_size"] == 10_000 and document["valid"]
