"""Ingest subcommand for cfd-bench."""

from __future__ import annotations

import argparse
import sys

from cfd_bench.core.paths import resolve_tiledb_root, resolve_vtk_dir
from cfd_bench.ingest.orchestrator import ingest_all


def add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser("ingest", help="Load CFD data into storage backends")
    ap.add_argument("--dat", required=True, help="Path to a .dat file or directory of .dat files")
    ap.add_argument(
        "--datasets",
        required=True,
        metavar="DATASET",
        help="Dataset key, e.g. JBC_615k",
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["postgresql", "iotdb", "tiledb"],
        help="Backends to ingest (default: postgresql iotdb tiledb)",
    )
    ap.add_argument("--zone-indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tiledb-root", default=resolve_tiledb_root())
    ap.add_argument("--vtk-root", default=resolve_vtk_dir(), help="VTK backend root directory")
    ap.add_argument(
        "--init-pg-schema",
        dest="init_pg_schema",
        action="store_true",
        default=True,
        help="Apply PostgreSQL DDL before load (default: true)",
    )
    ap.add_argument(
        "--no-init-pg-schema",
        dest="init_pg_schema",
        action="store_false",
        help="Skip PostgreSQL DDL",
    )
    ap.add_argument(
        "--build-pg-spatial",
        dest="build_pg_spatial",
        action="store_true",
        default=True,
        help="Build PostGIS spatial layers after PG load (default: true)",
    )
    ap.add_argument(
        "--no-build-pg-spatial",
        dest="build_pg_spatial",
        action="store_false",
        help="Skip PostGIS spatial layer build",
    )
    ap.add_argument("--iotdb-host", default="127.0.0.1")
    ap.add_argument("--iotdb-port", default="6667")
    ap.add_argument("--iotdb-user", default="root")
    ap.add_argument("--iotdb-password", default="root")
    ap.set_defaults(func=run_ingest)


def run_ingest(args: argparse.Namespace) -> int:
    report = ingest_all(
        args.dat,
        args.datasets,
        args.backends,
        zone_indices=args.zone_indices,
        tiledb_root=args.tiledb_root,
        vtk_root=args.vtk_root,
        init_pg_schema=args.init_pg_schema,
        build_pg_spatial=args.build_pg_spatial,
        iotdb_host=args.iotdb_host,
        iotdb_port=args.iotdb_port,
        iotdb_user=args.iotdb_user,
        iotdb_password=args.iotdb_password,
    )

    print("\n[ingest] Summary")
    for backend in report.success:
        print(f"  OK   {backend}")
    for backend, err in report.failed:
        print(f"  FAIL {backend}: {err}")

    return 0 if report.ok else 1
