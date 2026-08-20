"""Run subcommand for cfd-bench."""

from __future__ import annotations

import argparse

from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.runner import DEFAULT_WORKLOADS, run_all
from cfd_bench.core.results import csv_result_output


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser("run", help="Run CFD-Bench workloads W1-W11")
    ap.add_argument(
        "--workloads",
        nargs="+",
        default=list(DEFAULT_WORKLOADS),
        help="Workloads to run (default: w1..w8; extended: w9 w10 w11)",
    )
    add_common_workload_args(ap)
    ap.add_argument(
        "--output",
        default=None,
        metavar="RESULTS.csv",
        help="Mirror completed benchmark readings to a CSV file (disabled by default)",
    )
    # H5 expansion is PostgreSQL-first.  Other backends remain opt-in and all
    # legacy arguments are still available as overrides.
    ap.set_defaults(
        backend=["postgresql"],
        func=run_run,
    )


def run_run(args: argparse.Namespace) -> int:
    cfg = workload_config_from_args(args)
    backends = set(args.backend)
    print("Runtime configuration:")
    print(f"  backends={sorted(backends)} datasets={cfg.ships}")
    print(f"  progress={cfg.progress} progress_interval={cfg.progress_interval_sec:g}s")
    for ship in cfg.ships:
        print(
            f"  {ship}: zone={cfg.fluid_zone(ship)} "
            f"steps={cfg.valid_steps(ship)} vars={cfg.valid_variables(ship)}"
        )
    with csv_result_output(args.output) as writer:
        if writer is not None:
            print(f"  csv_results={writer.path}")
        run_all(cfg, backends, workloads=args.workloads)
    return 0
