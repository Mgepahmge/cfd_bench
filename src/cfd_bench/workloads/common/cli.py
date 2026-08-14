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
    ap.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset key(s) to benchmark, e.g. JBC_615k beam_static",
    )
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--steps", type=int, nargs="+", default=None, help="Override auto-discovered timesteps/frames")
    ap.add_argument("--variables", nargs="+", default=None, help="Override auto-discovered variables")
    ap.add_argument("--zone-fluid", default=None, help="Override auto-discovered fluid/result zone")
    ap.add_argument("--zone-hull", default="0_Wall_hull")
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
    ap.add_argument(
        "--max-range-dir",
        default=resolve_max_range_dir(),
        help="Legacy W3 sidecar directory for non-PostgreSQL backends",
    )


def workload_config_from_args(args, datasets=None, ships=None) -> WorkloadConfig:
    requested_ships = list(args.datasets or datasets or ships or [])
    explicit_steps = getattr(args, "steps", None)
    explicit_variables = getattr(args, "variables", None)
    explicit_zone = getattr(args, "zone_fluid", None)
    backends = set(getattr(args, "backend", []) or [])

    discovered_steps = {}
    discovered_variables = {}
    discovered_zones = {}

    needs_pg_discovery = "postgresql" in backends and (
        explicit_steps is None
        or explicit_variables is None
        or explicit_zone is None
    )
    if needs_pg_discovery:
        try:
            from cfd_bench.infra.postgresql.discovery import discover_postgresql_datasets

            infos = discover_postgresql_datasets(
                preferred_zone=explicit_zone,
                selected_datasets=requested_ships,
            )
        except Exception:
            # Dataset identity is always explicit.  If metadata discovery is
            # unavailable, retain the legacy fallback for steps/variables/zone.
            infos = []

        for info in infos:
            discovered_steps[info.dataset_key] = list(info.timesteps)
            discovered_variables[info.dataset_key] = list(info.variables)
            discovered_zones[info.dataset_key] = info.zone_type

    needs_iotdb_discovery = "iotdb" in backends and (
        explicit_steps is None
        or explicit_variables is None
        or explicit_zone is None
    )
    if needs_iotdb_discovery:
        try:
            from cfd_bench.infra.iotdb.discovery import discover_iotdb_datasets

            infos = discover_iotdb_datasets(requested_ships)
        except Exception:
            infos = []
        for info in infos:
            # PostgreSQL discovery wins when both backends are selected.  This
            # keeps the stable PostgreSQL runtime configuration unchanged.
            discovered_steps.setdefault(info.dataset_key, list(info.timesteps))
            discovered_variables.setdefault(info.dataset_key, list(info.variables))
            discovered_zones.setdefault(info.dataset_key, info.zone_type)

    return WorkloadConfig(
        ships=requested_ships,
        duration_sec=args.duration,
        steps=explicit_steps,
        variables=(
            [str(v).upper() for v in explicit_variables]
            if explicit_variables is not None
            else None
        ),
        geom_engine=args.geom_engine,
        vtk_dir=args.vtk_dir,
        vtk_hull_dir=getattr(args, "vtk_hull_dir", resolve_vtk_hull_dir()),
        tiledb_root=args.tiledb_root,
        max_range_dir=getattr(args, "max_range_dir", resolve_max_range_dir()),
        zone_fluid=explicit_zone,
        zone_hull=getattr(args, "zone_hull", "0_Wall_hull"),
        discovered_steps=discovered_steps,
        discovered_variables=discovered_variables,
        discovered_zones=discovered_zones,
    )
