"""Shared argparse helpers for workloads."""

from __future__ import annotations

import argparse

from cfd_bench.core.paths import (
    resolve_max_range_dir,
    resolve_tiledb_root,
    resolve_vtk_dir,
    resolve_vtk_hull_dir,
)
from cfd_bench.workloads.common.config import WorkloadConfig


def add_common_workload_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ships", nargs="+", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument(
        "--backend",
        nargs="+",
        default=["postgresql", "iotdb", "tiledb", "vtk"],
        help="Storage backends to benchmark",
    )
    ap.add_argument(
        "--geom-engine",
        choices=["db", "vtk"],
        default="db",
        help="Geometry engine: db=use data backend (default); vtk=use VTK files",
    )
    ap.add_argument("--vtk-dir", default=resolve_vtk_dir())
    ap.add_argument("--vtk-hull-dir", default=resolve_vtk_hull_dir())
    ap.add_argument("--tiledb-root", default=resolve_tiledb_root())
    ap.add_argument("--max-range-dir", default=resolve_max_range_dir())


def workload_config_from_args(args, ships=None) -> WorkloadConfig:
    return WorkloadConfig(
        ships=args.ships or ships or WorkloadConfig().ships,
        duration_sec=args.duration,
        geom_engine=args.geom_engine,
        vtk_dir=args.vtk_dir,
        vtk_hull_dir=getattr(args, "vtk_hull_dir", resolve_vtk_hull_dir()),
        tiledb_root=args.tiledb_root,
        max_range_dir=getattr(args, "max_range_dir", resolve_max_range_dir()),
    )
