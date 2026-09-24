"""The benchmark suite's own behaviour: definitions, workloads, measurement,
results and interpretation. No timing assertions: these must pass on any machine.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frames2py import EVENT_DTYPE

from benchmarks import environment, report, results
from benchmarks.__main__ import main
from benchmarks.matrix import (
    GATE_THRESHOLD_EVENTS_PER_S,
    V1_KERNELS,
    Cell,
    gate_cells,
    prototype_baseline_cells,
)
from benchmarks.measure import Policy, measure_memory, nearest_rank, summarize, time_calls
from benchmarks.runner import run_suite
from benchmarks.targets import TARGETS, Level, Prepared
from benchmarks.workloads import Workload


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
    """Sums x over each batch; supports every kernel except timestamp_decay."""

    name = "fake"

    def __init__(self, level: Level = "kernel") -> None:
        self.level = level
        self.prepared = 0

    def supports(self, cell: Cell) -> bool:
        return cell.kernel != "timestamp_decay"

    def prepare(self, cell: Cell) -> Prepared:
        self.prepared += 1
        total = [0]

        def call(batch: np.ndarray) -> None:
            total[0] += int(batch["x"].sum())

        def counters() -> Mapping[str, int]:
            return {"calls": total[0]}

        return Prepared(call=call, counters=counters, details={"fresh": True})

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "level": self.level}


def _document(level: Level, cells: tuple[Cell, ...], policy: Policy) -> dict[str, Any]:
    return run_suite("test", cells, _FakeTarget(level), policy, progress=lambda _: None,
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
        cells = (Cell("event_count", (32, 24), 1_000, 0.0, "uniform", seed=1),
                 Cell("timestamp_decay", (32, 24), 1_000, 0.0, "uniform", seed=1))
        document = run_suite("test", cells, TARGETS["prototype-kernel"](),
                             Policy(runs=2, timed_calls=3, memory_calls=1), progress=lambda _: None)
        measured, unsupported = document["cells"]
        assert document["policy"]["process_per_run"] is True
        assert [len(run) for run in measured["call_ns"]] == [3, 3]
        assert measured["memory"]["calls"] == 1
        assert measured["details"]["state_dtype"] == "float32"
        assert unsupported["status"] == "unsupported"

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


def _with_throughput(document: dict[str, Any], events_per_s: dict[Any, float]) -> dict[str, Any]:
    for record in document["cells"]:
        condition = Cell.from_record(record).condition
        if condition in events_per_s:
            record["summary"]["events_per_s"] = events_per_s[condition]
    return document


class TestGateVerdicts:
    def _documents(self, engine_speed: dict[Any, float]) -> tuple[dict[str, Any], dict[str, Any]]:
        cells = tuple(c for c in gate_cells() if c.kernel == "event_count" and c.sensor_size == (346, 260))
        policy = Policy(runs=1, timed_calls=1, memory_calls=0)
        kernel = _with_throughput(_document("kernel", cells, policy), {c.condition: 25e6 for c in cells})
        engine = _with_throughput(_document("engine", cells, policy),
                                  {c.condition: engine_speed.get(c.condition, 25e6) for c in cells})
        return kernel, engine

    def test_a_cell_passes_only_when_both_levels_reach_20m(self) -> None:
        slow = gate_cells()[0].condition
        kernel, engine = self._documents({slow: 19.9e6})
        verdicts = {cell.condition: verdict for cell, verdict, _, _ in report.gate_verdicts(kernel, engine)}
        assert verdicts[slow] == "fail"
        measured = {c for c, v in verdicts.items() if v != "incomplete"}
        assert len(measured) == 10
        assert all(verdicts[c] == "pass" for c in measured - {slow})
        assert sum(v == "incomplete" for v in verdicts.values()) == 140

    def test_levels_cannot_be_swapped(self) -> None:
        kernel, engine = self._documents({})
        with pytest.raises(ValueError):
            report.gate_verdicts(engine, kernel)

    def test_off_reference_results_are_labelled(self) -> None:
        kernel, engine = self._documents({})
        for document in (kernel, engine):
            document["environment"]["chip"] = "Some Other CPU"
        assert "don't count toward the gate" in report.gate_summary(kernel, engine)

    def test_table_reports_every_cell_with_its_conditions(self) -> None:
        kernel, _ = self._documents({})
        text = report.table(kernel)
        assert text.startswith("Conditions: fake (kernel level)")
        assert text.count("\n| event_count 346x260") == 10


class TestPrototypeTargets:
    @pytest.mark.parametrize("name", ["prototype-kernel", "prototype-engine"])
    @pytest.mark.parametrize("kernel", ["event_count", "polarity", "time_surface", "exp_decay"])
    def test_runs_every_prototype_kernel(self, name: str, kernel: str) -> None:
        target = TARGETS[name]()
        cell = Cell(kernel, (64, 48), 2_000, 0.0, "uniform", seed=1)
        assert target.supports(cell)
        prepared = target.prepare(cell)
        for batch in cell.workload().batches(2):
            prepared.call(batch)
        assert "state_dtype" in prepared.details

    @pytest.mark.parametrize("name", ["prototype-kernel", "prototype-engine"])
    def test_timestamp_decay_is_unsupported(self, name: str) -> None:
        cell = Cell("timestamp_decay", (64, 48), 2_000, 0.0, "uniform", seed=1)
        assert not TARGETS[name]().supports(cell)

    def test_engine_target_uses_the_cell_interval(self) -> None:
        target = TARGETS["prototype-engine"]()
        every_call = target.prepare(Cell("event_count", (64, 48), 100, 0.0, "uniform", seed=1))
        batches = Workload("uniform", (64, 48), 100, seed=1).batches(5)
        _, deltas = time_calls(every_call.call, batches, 1, 4, every_call.counters)
        assert deltas["snapshots_published"] == 4
        hourly = target.prepare(Cell("event_count", (64, 48), 100, 3_600_000.0, "uniform", seed=1))
        _, deltas = time_calls(hourly.call, batches, 1, 4, hourly.counters)
        assert deltas["snapshots_published"] == 0


def test_cli_lists_the_gate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--suite", "gate"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("150 cells, 150 in the hard gate")
    assert main(["list", "--suite", "prototype-baseline", "--batch-size", "10000", "--interval", "0"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("12 cells, 0 in the hard gate")
