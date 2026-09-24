"""Command line: ``uv run python -m benchmarks {list,run,report,gate}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmarks import report, results
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

    gate_cmd = commands.add_parser("gate", help="per-cell gate verdicts from both levels")
    gate_cmd.add_argument("--kernel-level", type=Path, required=True)
    gate_cmd.add_argument("--engine-level", type=Path, required=True)

    commands.add_parser("worker", help="internal: one run, request on stdin, records on stdout")

    args = parser.parse_args(argv)

    if args.command == "worker":
        json.dump(worker(json.load(sys.stdin)), sys.stdout)
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

    print(report.gate_summary(results.read(args.kernel_level), results.read(args.engine_level)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
