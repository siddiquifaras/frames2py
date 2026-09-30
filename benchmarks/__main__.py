"""Command line: ``uv run python -m benchmarks {list,run,report,stage2,gate,adapters,adapters-report,
recorder,viewer,replay,consumers-report,observation}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks import adapters, consumers, environment, gate, report, results
from benchmarks.matrix import SUITES, Cell
from benchmarks.measure import Policy
from benchmarks.runner import run_suite, worker
from benchmarks.targets import TARGETS


def _resolution(text: str) -> tuple[int, int]:
    width, _, height = text.partition("x")
    try:
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from None


def _select(cells: Sequence[Cell], args: argparse.Namespace) -> list[Cell]:
    def keep(cell: Cell) -> bool:
        return (
            (not args.kernel or cell.kernel in args.kernel)
            and (not args.resolution or cell.sensor_size in args.resolution)
            and (not args.batch_size or cell.batch_size in args.batch_size)
            and (not args.interval or cell.interval_ms in args.interval)
            and (not args.distribution or cell.distribution in args.distribution)
        )

    return [cell for cell in cells if keep(cell)]


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--kernel", action="append", help="repeatable")
    parser.add_argument("--resolution", action="append", type=_resolution, help="WxH, repeatable")
    parser.add_argument("--batch-size", action="append", type=int, help="repeatable")
    parser.add_argument("--interval", action="append", type=float, help="ms, repeatable")
    parser.add_argument("--distribution", action="append", help="repeatable")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)

    list_cmd = commands.add_parser("list", help="print the cells of a suite")
    _add_filters(list_cmd)

    defaults = Policy()
    run_cmd = commands.add_parser("run", help="measure a suite against a target")
    _add_filters(run_cmd)
    run_cmd.add_argument("--target", choices=sorted(TARGETS), required=True)
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--overwrite", action="store_true")
    run_cmd.add_argument("--runs", type=int, default=defaults.runs)
    run_cmd.add_argument("--warmup-calls", type=int, default=defaults.warmup_calls)
    run_cmd.add_argument("--timed-calls", type=int, default=None)
    run_cmd.add_argument("--memory-calls", type=int, default=defaults.memory_calls)
    run_cmd.add_argument(
        "--same-process", action="store_true", help="run every run in this process, not one process per run"
    )

    report_cmd = commands.add_parser("report", help="tabulate a result document")
    report_cmd.add_argument("result", type=Path)

    stage2_cmd = commands.add_parser(
        "stage2", help="re-measure a primary document's borderline cells, same target and policy"
    )
    stage2_cmd.add_argument("primary", type=Path)
    stage2_cmd.add_argument("--out", type=Path, required=True)

    gate_cmd = commands.add_parser("gate", help="per-cell gate verdicts from both levels")
    gate_cmd.add_argument("--kernel-level", type=Path, required=True)
    gate_cmd.add_argument("--engine-level", type=Path, required=True)
    gate_cmd.add_argument("--kernel-stage2", type=Path)
    gate_cmd.add_argument("--engine-stage2", type=Path)
    gate_cmd.add_argument("--out", type=Path, help="also write the verdicts as JSON")

    commands.add_parser("worker", help="internal: one run, request on stdin, records on stdout")

    adapters_cmd = commands.add_parser("adapters", help="characterise an adapter on a downloaded real recording")
    adapters_cmd.add_argument("--recording", required=True, help="a name from tests.recordings")
    adapters_cmd.add_argument("--out", type=Path, required=True)
    adapters_cmd.add_argument("--runs", type=int, default=5)
    adapters_cmd.add_argument("--kernel", default="event_count")
    adapters_cmd.add_argument("--interval", type=float, default=16.0, help="snapshot interval, ms")
    adapters_cmd.add_argument("--batch-size", type=int, default=None, help="the reader's batch_size (default: None)")
    adapters_report_cmd = commands.add_parser("adapters-report", help="tabulate an adapter characterisation document")
    adapters_report_cmd.add_argument("result", type=Path)
    commands.add_parser("adapters-worker", help="internal: one adapter run, request on stdin, record on stdout")
    recorder_cmd = commands.add_parser("recorder", help="characterise the recorder's write rate on real recordings")
    recorder_cmd.add_argument("--recording", action="append", help="a name from tests.recordings; repeatable")
    recorder_cmd.add_argument("--events", type=int, default=consumers.RECORDER_EVENTS, help="events per recording")
    viewer_cmd = commands.add_parser("viewer", help="characterise rendering and a viewing consumer's effect on ingest")
    replay_cmd = commands.add_parser("replay", help="characterise paced replay's timing on real recordings")
    replay_cmd.add_argument("--seconds", type=float, default=consumers.REPLAY_SECONDS,
                            help="seconds of each recording")
    for sub in (recorder_cmd, viewer_cmd, replay_cmd):
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--runs", type=int, default=5)
    consumers_report_cmd = commands.add_parser("consumers-report", help="tabulate a consumer characterisation document")
    consumers_report_cmd.add_argument("result", type=Path)
    commands.add_parser("consumers-worker", help="internal: one consumer run, request on stdin, record on stdout")
    _add_observation(commands)

    args = parser.parse_args(argv)

    if args.command == "observation":
        return _observation(args)

    if args.command == "worker":
        json.dump(worker(json.load(sys.stdin)), sys.stdout)
        return 0

    if args.command == "consumers-worker":
        json.dump(consumers.worker(json.load(sys.stdin)), sys.stdout)
        return 0
    if args.command in ("recorder", "viewer", "replay"):
        if args.out.exists():
            parser.error(f"{args.out} exists")
        extra = {"recordings": args.recording, "events": args.events} if args.command == "recorder" else (
            {"seconds": args.seconds} if args.command == "replay" else {})
        document = consumers.run(args.command, runs=args.runs, **extra)
        args.out.write_text(json.dumps(document, indent=1))
        print(consumers.report(document))
        return 0 if document["valid"] else 1
    if args.command == "consumers-report":
        print(consumers.report(json.loads(args.result.read_text())))
        return 0
    if args.command == "adapters-worker":
        json.dump(adapters.worker(json.load(sys.stdin)), sys.stdout)
        return 0

    if args.command == "adapters":
        if args.out.exists():
            parser.error(f"{args.out} exists; refusing to overwrite a result")
        document = adapters.run(args.recording, runs=args.runs, kernel=args.kernel, interval_ms=args.interval,
                                batch_size=args.batch_size)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=1) + "\n")
        print(adapters.report(document))
        print(f"wrote {args.out}")
        return 0

    if args.command == "adapters-report":
        print(adapters.report(json.loads(args.result.read_text())))
        return 0

    if args.command == "list":
        cells = _select(SUITES[args.suite](), args)
        print(report.labels(cells))
        print(f"{len(cells)} cells, {sum(c.in_gate for c in cells)} in the hard gate")
        return 0

    if args.command == "run":
        cells = _select(SUITES[args.suite](), args)
        if not cells:
            parser.error("the filters select no cells")
        if args.out.exists() and not args.overwrite:
            parser.error(f"{args.out} exists; pass --overwrite to replace it")
        policy = Policy(
            runs=args.runs,
            warmup_calls=args.warmup_calls,
            timed_calls=args.timed_calls,
            memory_calls=args.memory_calls,
        )
        document = run_suite(
            args.suite, cells, TARGETS[args.target](), policy, process_per_run=not args.same_process
        )
        results.write(args.out, document, overwrite=args.overwrite)
        print(f"wrote {args.out}")
        return 0

    if args.command == "report":
        print(report.table(results.read(args.result)))
        return 0

    if args.command == "stage2":
        return _stage2(parser, args)

    def optional(path: Path | None) -> dict[str, object] | None:
        return results.read(path) if path is not None else None

    verdicts = gate.verdicts(
        results.read(args.kernel_level),
        results.read(args.engine_level),
        optional(args.kernel_stage2),
        optional(args.engine_stage2),
    )
    if args.out is not None:
        if args.out.exists():
            parser.error(f"{args.out} exists; refusing to overwrite a result")
        args.out.write_text(json.dumps(verdicts, indent=1) + "\n")
    print(gate.summary(verdicts))
    return 0


def _add_observation(commands: Any) -> None:
    study = commands.add_parser("observation", help="the observation study (benchmarks/observation_preregistration.md)")
    sub = study.add_subparsers(dest="observation_command", required=True)
    validate = sub.add_parser("validate", help="run a validation stage (22)")
    validate.add_argument("--stage", choices=["V1", "V2", "V3", "V4", "V6"], required=True)
    validate.add_argument("--no-env-gate", action="store_true",
                          help="record the environment checks without gating on them (V4 on a machine in normal use)")
    sub.add_parser("calibrate", help="V5: calibrate K5 and write the calibration record")
    run = sub.add_parser("run", help="run campaign passes (11.4)")
    run.add_argument("--experiment", choices=["P1", "P2", "all"], default="all")
    run.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    run.add_argument("--unattended", action="store_true")
    run.add_argument("--session-retries", type=int, default=0)
    run.add_argument("--retry-wait", type=float, default=600.0, help="seconds between session-check retries")
    analyse = sub.add_parser("analyse", help="tables and figure data from a campaign directory")
    analyse.add_argument("--campaign", type=Path, default=None)
    analyse.add_argument("--out", type=Path, required=True)
    sub.add_parser("worker", help="internal: one run, request on stdin")
    sub.add_parser("validate-worker", help="internal: one validation stage, request on stdin")


def _observation(args: argparse.Namespace) -> int:
    command = args.observation_command
    if command == "worker":
        from benchmarks import observation

        return observation.worker_main()
    if command == "validate-worker":
        from benchmarks import observation_validation

        json.dump(observation_validation.validate_worker(json.load(sys.stdin)), sys.stdout, default=str)
        return 0
    from benchmarks import observation_driver as driver

    if command == "validate":
        out, ok = driver.validate(args.stage, gate_env=not args.no_env_gate)
        print(f"{args.stage}: {'passed' if ok else 'FAILED'}; output in {out}")
        return 0 if ok else 1
    if command == "calibrate":
        out, ok = driver.calibrate()
        print(f"V5: {'passed' if ok else 'FAILED'}; output in {out}")
        return 0 if ok else 1
    if command == "run":
        experiments = ["P1", "P2"] if args.experiment == "all" else [args.experiment]
        return driver.campaign(experiments, args.passes, unattended=args.unattended,
                               session_retries=args.session_retries, retry_wait_s=args.retry_wait)
    from benchmarks import observation_analysis

    summary = observation_analysis.analyse(args.campaign or driver.CAMPAIGN_DIR, args.out)
    print(json.dumps({k: summary[k] for k in ("cells", "p2_cells", "valid_runs", "attempts")}))
    for name, verdict in summary["verdicts"].items():
        print(f"{name}: {verdict['verdict']}")
    return 0


def _stage2(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    primary = results.read(args.primary)
    if args.out.exists():
        parser.error(f"{args.out} exists; refusing to overwrite a result")
    here = environment.capture()
    now = {key: here.get(key) for key in gate.RUNTIME_KEYS}
    if now != gate.runtime_of(primary):
        parser.error(f"this interpreter is {now}; the primary document ran on {gate.runtime_of(primary)}")
    if here.get("commit") != primary["environment"].get("commit") or not environment.working_tree_clean(here):
        parser.error("stage 2 must run from the primary document's commit, on a clean working tree")
    cells = gate.borderline_cells(primary)
    recorded = primary["policy"]
    policy = Policy(
        runs=gate.STAGE2_RUNS,
        warmup_calls=recorded["warmup_calls"],
        timed_calls=recorded["timed_calls"],
        memory_calls=recorded["memory_calls"],
    )
    target = TARGETS[primary["target"]["name"]]()
    print(f"{len(cells)} borderline cell(s)", file=sys.stderr)
    document = run_suite(primary["suite"], cells, target, policy,
                         process_per_run=recorded["process_per_run"])
    results.write(args.out, document)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
