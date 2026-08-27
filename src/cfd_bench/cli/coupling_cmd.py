"""CLI for high-throughput structure-to-CFD coupling."""

from __future__ import annotations

import argparse

from cfd_bench.core.warnings_state import preserve_warning_filters


def add_coupling_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser(
        "couple",
        help="Map every structural H5 node to one IoTDB CFD frame by linear interpolation",
    )
    ap.add_argument("--structure-dataset", required=True, help="H5 dataset key already ingested to IoTDB")
    ap.add_argument("--cfd-dataset", required=True, help="CFD dataset key already ingested to IoTDB")
    ap.add_argument("--cfd-step", type=int, required=True, help="CFD timestep/frame to map")
    ap.add_argument("--output", required=True, help="Independent coupling result .h5 path")
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="CFD variables to map (default: all variables in CFD metadata)",
    )
    ap.add_argument(
        "--structure-zone",
        default=None,
        help="Structural mesh zone (default: H5 dataset metadata zone)",
    )
    ap.add_argument(
        "--cfd-zone",
        default=None,
        help="CFD mesh/result zone (default: CFD dataset metadata zone)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Structural points written per output batch (default: 4096)",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also save support node IDs and barycentric weights for every mapped point",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable coupling progress output",
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=0.25,
        help="Minimum TTY progress refresh interval in seconds (default: 0.25)",
    )
    ap.set_defaults(func=run_coupling)


def run_coupling(args: argparse.Namespace) -> int:
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if float(args.progress_interval) <= 0:
        raise ValueError("--progress-interval must be > 0")

    with preserve_warning_filters():
        from cfd_bench.features.structure_cfd_coupling import StructureCfdCouplingEngine
        from cfd_bench.infra.iotdb.config import IoTDBConfig
        from cfd_bench.infra.iotdb.repository import IoTDBRepository

        repo = IoTDBRepository(IoTDBConfig())
        repo.open()
        try:
            engine = StructureCfdCouplingEngine(repo)
            summary = engine.couple_to_h5(
                structure_dataset=str(args.structure_dataset),
                cfd_dataset=str(args.cfd_dataset),
                cfd_step=int(args.cfd_step),
                output_path=str(args.output),
                variables=args.variables,
                structure_zone=args.structure_zone,
                cfd_zone=args.cfd_zone,
                batch_size=int(args.batch_size),
                diagnostics=bool(args.diagnostics),
                progress=not bool(args.no_progress),
                progress_interval=float(args.progress_interval),
            )
        finally:
            repo.close()

    print("Structure/CFD coupling result")
    print(f"  structure={summary.structure_dataset} zone={summary.structure_zone}")
    print(f"  cfd={summary.cfd_dataset} step={summary.cfd_step} zone={summary.cfd_zone}")
    print(f"  variables={list(summary.variables)}")
    print(
        "  nodes="
        f"{summary.node_count} pass={summary.success_count} outside={summary.outside_count} "
        f"no_cell={summary.no_containing_cell_count} failed={summary.failed_count}"
    )
    print(f"  output={summary.output_path}")
    return 0
