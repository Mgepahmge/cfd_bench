"""Run subcommand for cfd-bench."""

from __future__ import annotations

import argparse

from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.runner import DEFAULT_WORKLOADS, run_all


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser("run", help="Run CFD-Bench workloads W1-W8")
    ap.add_argument(
        "--workloads",
        nargs="+",
        default=list(DEFAULT_WORKLOADS),
        help="Workloads to run (default: w1..w8)",
    )
    add_common_workload_args(ap)
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
    for ship in cfg.ships:
        print(
            f"  {ship}: zone={cfg.fluid_zone(ship)} "
            f"steps={cfg.valid_steps(ship)} vars={cfg.valid_variables(ship)}"
        )
    run_all(cfg, backends, workloads=args.workloads)
    return 0
