"""The consumer characterisation harness (``benchmarks/consumers.py``): document validity, the
power guard, case configuration, the workers' own checks and failure handling. No timing
assertions, and no real measurement durations: workers run on tiny inputs or are replaced.
"""

from __future__ import annotations

import functools
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import consumers, power
from benchmarks.__main__ import main
from frames2py import EVENT_DTYPE
from frames2py.replay import paced
from tests.adapters.backends import require_backend
from tests.fake_power import FakeMac


def events(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=EVENT_DTYPE)
    out["t"] = np.cumsum(rng.integers(1, 4, n), dtype=np.uint64)
    out["x"], out["y"], out["p"] = rng.integers(0, 64, n), rng.integers(0, 48, n), rng.integers(0, 2, n)
    return out


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> FakeMac:
    """A macOS power backend in full wake, used by every ``run()`` in the test."""
    backend = FakeMac(0x1F)
    monkeypatch.setattr(power, "hold_awake", functools.partial(power.hold_awake, backend=backend, platform="darwin"))
    return backend


def canned(checks: dict[str, bool] | None = None) -> Any:
    requests: list[dict[str, Any]] = []

    def spawn(request: dict[str, Any], python: str) -> dict[str, Any]:
        requests.append(request)
        return {"checks": dict(checks or {"ok": True})}

    spawn.requests = requests  # type: ignore[attr-defined]
    return spawn


class TestDocumentValidity:
    def test_every_case_runs_the_given_number_of_times_interleaved(self, mac: FakeMac,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        spawn = canned()
        monkeypatch.setattr(consumers, "_spawn", spawn)
        document = consumers.run("replay", runs=3, seconds=1.0)
        cases = len(consumers.REPLAY_CASES) * 2
        assert document["schema"] == consumers.SCHEMA and document["valid"] is True
        assert [len(entry["runs"]) for entry in document["cases"]] == [3] * cases
        order = [(r["recording"], r["speed"], r["ingest"]) for r in spawn.requests]
        assert order[:cases] == order[cases : 2 * cases]  # run 1 of every case, then run 2, ...
        assert [r["memory"] for r in spawn.requests] == [False] * (2 * cases) + [True] * cases
        assert {r["viewer_first"] for r in spawn.requests[:cases]} == {False}
        assert {r["viewer_first"] for r in spawn.requests[cases : 2 * cases]} == {True}
        assert mac.held == set()

    def test_one_failed_check_makes_the_document_invalid(self, mac: FakeMac, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(consumers, "_spawn", canned({"ok": True, "round_trip": False}))
        assert consumers.run("replay", runs=1)["valid"] is False

    def test_ending_outside_full_wake_makes_the_document_invalid(self, mac: FakeMac,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        def spawn(request: dict[str, Any], python: str) -> dict[str, Any]:
            mac.caps = 0x9  # DarkWake: CPU without graphics
            return {"checks": {"ok": True}}

        monkeypatch.setattr(consumers, "_spawn", spawn)
        document = consumers.run("replay", runs=1)
        assert document["valid"] is False and power.ended_outside_full_wake(document["power"])

    def test_sleep_during_the_run_makes_the_document_invalid(self, mac: FakeMac,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
        clocks = iter([(0, 0), (60_000_000_000, 5_000_000_000)])  # 55 s asleep
        monkeypatch.setattr(power, "_clock_pair", lambda: next(clocks))
        monkeypatch.setattr(consumers, "_spawn", canned())
        document = consumers.run("replay", runs=1)
        assert document["valid"] is False and document["power"]["slept"] is True

    def test_nothing_runs_outside_full_wake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeMac(0x9)
        monkeypatch.setattr(power, "hold_awake", functools.partial(power.hold_awake, backend=backend, platform="darwin"))
        spawn = canned()
        monkeypatch.setattr(consumers, "_spawn", spawn)
        with pytest.raises(power.PowerStateError):
            consumers.run("replay", runs=1)
        assert spawn.requests == [] and backend.created == 0

    def test_a_failed_worker_process_fails_the_run_and_releases_the_guard(self, mac: FakeMac,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        def spawn(request: dict[str, Any], python: str) -> dict[str, Any]:
            raise subprocess.CalledProcessError(1, ["python", "-m", "benchmarks", "consumers-worker"])

        monkeypatch.setattr(consumers, "_spawn", spawn)
        with pytest.raises(subprocess.CalledProcessError):
            consumers.run("replay", runs=1)
        assert mac.created == 1 and mac.held == set()

    def test_report_refuses_other_documents(self) -> None:
        with pytest.raises(ValueError, match=consumers.SCHEMA):
            consumers.report({"schema": "frames2py-adapter-benchmark/1"})


class TestConfiguration:
    def test_viewer_cases(self) -> None:
        cases = list(consumers._cases("viewer", {}, Path(".")))
        renders = [c for c in cases if c["kind"] == "render"]
        impacts = [c for c in cases if c["kind"] == "impact"]
        assert len(renders) == len(consumers.VIEWER_KERNELS) * len(consumers.VIEWER_RESOLUTIONS) * 2
        assert {tuple(c["sensor_size"]) for c in renders} == set(consumers.VIEWER_RESOLUTIONS)
        assert {(c["kernel"], c["batch_size"]) for c in impacts} == {
            (k, b) for k in consumers.IMPACT_KERNELS for b in consumers.IMPACT_BATCHES}

    def test_replay_cases_take_the_seconds_given(self) -> None:
        cases = list(consumers._cases("replay", {"seconds": 2.5}, Path(".")))
        assert {c["seconds"] for c in cases} == {2.5} and {c["ingest"] for c in cases} == {True, False}

    def test_recorder_cases_cover_every_compression_and_write_size(self, monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
        loaded: list[tuple[str, int]] = []

        def load(name: str, limit: int, directory: Path) -> tuple[Path, dict[str, Any]]:
            loaded.append((name, limit))
            return directory / f"{name}.npy", {"sensor_size": [64, 48], "events_sha256": "x"}

        monkeypatch.setattr(consumers, "_load_events", load)
        cases = list(consumers._cases("recorder", {"recordings": ["a", "b"], "events": 123}, tmp_path))
        assert loaded == [("a", 123), ("b", 123)]
        assert {(c["compression"], c["write_size"]) for c in cases} == {
            (comp, size) for comp in consumers.RECORDER_COMPRESSIONS for size in consumers.RECORDER_WRITE_SIZES}
        assert len(cases) == 2 * len(consumers.RECORDER_COMPRESSIONS) * len(consumers.RECORDER_WRITE_SIZES)

    def test_unknown_characterisation(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            list(consumers._cases("gpu", {}, Path(".")))

    def test_an_existing_output_is_never_overwritten(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("the characterisation must not start")

        monkeypatch.setattr(consumers, "run", must_not_run)
        out = tmp_path / "viewer.json"
        out.write_text("keep")
        with pytest.raises(SystemExit):
            main(["viewer", "--out", str(out)])
        assert out.read_text() == "keep"

    def test_statistics_helpers(self) -> None:
        assert consumers._percentiles([]) == {}
        assert consumers._percentiles([5])["max_ns"] == 5.0
        assert math.isnan(consumers._median([None, None])) and consumers._median([3.0, None, 1.0]) == 2.0


class TestWorkers:
    @pytest.mark.parametrize("compression", [None, "gzip", "blosc"])
    def test_recorder_worker_checks_the_round_trip(self, tmp_path: Path, compression: str | None) -> None:
        require_backend("h5py", "hdf5plugin")
        import hashlib

        data = events(3_000)
        np.save(tmp_path / "e.npy", data)
        request = {"events_path": str(tmp_path / "e.npy"), "sensor_size": [64, 48], "compression": compression,
                   "write_size": 700, "scratch": str(tmp_path), "memory": True,
                   "events_sha256": hashlib.sha256(data.tobytes()).hexdigest()}
        record = consumers.recorder_worker(request)
        assert record["checks"] == {"round_trip": True} and record["events"] == 3_000
        assert record["record_peak_traced_bytes"] > 0 and not (tmp_path / "r.h5").exists()
        assert consumers.recorder_worker({**request, "events_sha256": "0" * 64})["checks"] == {"round_trip": False}

    @pytest.mark.parametrize("kernel", consumers.VIEWER_KERNELS)
    def test_render_worker_checks_that_a_given_scale_renders_the_same(self, kernel: str) -> None:
        record = consumers.render_worker({"kernel": kernel, "sensor_size": [32, 24], "distribution": "clustered"})
        assert record["checks"] == {"same_scale_same_image": True}
        assert 0 < record["nonzero_values"] <= record["values"]
        assert (record["auto_scale_only"] == {}) is (kernel == "time_surface")

    def test_impact_worker_runs_both_arms_in_the_requested_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(consumers, "IMPACT_WARMUP_S", 0.02)
        monkeypatch.setattr(consumers, "IMPACT_WINDOW_S", 0.2)
        record = consumers.impact_worker({"kernel": "event_count", "batch_size": 10_000, "viewer_first": True})
        assert record["order"] == ["viewer", "none"] and record["checks"]["no_errors"]
        timed = math.ceil(0.22 * consumers.IMPACT_RATE / 10_000) - int(0.02 * consumers.IMPACT_RATE / 10_000)
        for arm in ("viewer", "none"):
            assert record["arms"][arm]["timed_batches"] == timed and record["arms"][arm]["errors"] == []
        assert record["arms"]["none"]["viewer_renders"] == 0

    def test_the_impact_stream_keeps_the_gate_streams_timestamps_across_pool_wraps(self) -> None:
        from benchmarks.workloads import Workload

        count = 2 * 64 + 1  # two pool wraps
        stream = consumers._impact_stream(500, count)
        gate = Workload("uniform", (1280, 720), 500, seed=3).batches(count)
        previous = -1
        for k in range(count):
            t = stream(k)["t"]
            assert np.array_equal(t, gate[k]["t"]), k
            assert int(t.min()) >= previous, k
            previous = int(t.max())

    def test_replay_due_times_are_an_independent_recomputation_of_paced(self) -> None:
        batches = [np.zeros(0, dtype=EVENT_DTYPE), events(40, 1)]
        tail = events(30, 2)
        tail["t"] += np.uint64(10_000_000)  # forward jump
        back = events(20, 3)  # back to the start
        batches += [tail, back, np.zeros(0, dtype=EVENT_DTYPE)]
        for speed in (0.5, 1.0, 3.0):
            due = consumers._due_offsets(batches, speed)
            clock = [0]
            got = []
            for _ in paced(batches, speed=speed, clock=lambda: clock[0],
                           sleep=lambda s: clock.__setitem__(0, clock[0] + round(s * 1e9))):
                got.append(clock[0])
            assert got == due

    def test_replay_worker_checks_its_fake_clock_and_lateness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        batches = [events(200, i) for i in range(12)]
        for i, b in enumerate(batches):
            b["t"] += np.uint64(i * 1_000)  # 12 ms of events
        monkeypatch.setattr(consumers, "_replay_batches", lambda name, seconds: ([b.copy() for b in batches], (64, 48)))
        record = consumers.replay_worker({"recording": "fake", "speed": 1.0, "ingest": True, "seconds": 1.0})
        assert record["checks"] == {"never_early": True, "fake_clock_exact": True}
        assert record["batches"] == 12 and record["fake_clock_exact"] == 12
        assert record["duration_ns"] >= record["requested_ns"]


def test_report_tabulates_each_kind(mac: FakeMac, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(consumers, "IMPACT_WARMUP_S", 0.02)
    monkeypatch.setattr(consumers, "IMPACT_WINDOW_S", 0.1)
    monkeypatch.setattr(consumers, "_spawn", lambda request, python: consumers.worker(request))
    monkeypatch.setattr(consumers, "VIEWER_KERNELS", ("event_count",))
    monkeypatch.setattr(consumers, "VIEWER_RESOLUTIONS", ((32, 24),))
    monkeypatch.setattr(consumers, "IMPACT_KERNELS", ("event_count",))
    monkeypatch.setattr(consumers, "IMPACT_BATCHES", (10_000,))
    document = consumers.run("viewer", runs=1)
    text = consumers.report(json.loads(json.dumps(document)))
    assert "render event_count" in text and "impact event_count" in text and "32x24" in text
