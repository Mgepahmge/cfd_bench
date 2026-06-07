"""Run subcommand for cfd-bench."""

from __future__ import annotations

import argparse

from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.runner import DEFAULT_WORKLOADS, run_all


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser("run", help="Run CFD-Bench workloads W1–W8")
    ap.add_argument(
        "--workloads",
        nargs="+",
        default=list(DEFAULT_WORKLOADS),
        help="Workloads to run (default: w1..w8)",
    )
    add_common_workload_args(ap)
    ap.set_defaults(
        backend=["postgresql", "iotdb", "tiledb"],
        func=run_run,
    )
    for action in ap._actions:
        if action.dest == "ships":
            action.required = True
            action.help = "Ship dataset keys (required, e.g. JBC_615k)"
            break


def run_run(args: argparse.Namespace) -> int:
    cfg = workload_config_from_args(args)
    backends = set(args.backend)
    run_all(cfg, backends, workloads=args.workloads)
    return 0
