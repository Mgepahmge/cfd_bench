"""W1: Point / line / plane intersection + scalar query."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.metrics import aggregation
from cfd_bench.workloads.common.random_geom import (
    random_line_in_bbox,
    random_plane_in_bbox,
    random_points_in_bbox,
)


def _run_point_queries(label, query_fn, scalar_fn, bounds, duration, variables):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(variables)
        while True:
            pts = random_points_in_bbox(bounds)
            cells = query_fn(pts)
            if len(cells) > 0:
                break
        vals = scalar_fn(cells, var)
        aggregation(vals)
        txn += 1
    print(f"{label} point intersection: {txn} txns in {duration}s")


def _run_line_queries(label, query_fn, scalar_fn, bounds, duration, variables):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(variables)
        start, end = random_line_in_bbox(bounds)
        cells = query_fn(start, end)
        if len(cells) > 0:
            scalar_fn(cells, var)
        txn += 1
    print(f"{label} line intersection: {txn} txns in {duration}s")


def _run_plane_queries(label, query_fn, scalar_fn, bounds, duration, variables):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(variables)
        origin, normal = random_plane_in_bbox(bounds)
        cells = query_fn(origin, normal)
        if len(cells) > 0:
            scalar_fn(cells, var)
        txn += 1
    print(f"{label} plane intersection: {txn} txns in {duration}s")


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    steps = cfg.valid_steps(ship)

    for step in steps:
        if cfg.skip_step(ship, step):
            print(f"skip {ship} step {step}")
            continue
        print(f"\n=== W1 ship={ship} step={step} ===")
        shared_bounds = None

        if "postgresql" in backends:
            pg = make_pg(ship, step, zone=cfg.zone_fluid)
            geom = make_geom_client(cfg, ship, step, pg, cfg.zone_fluid)
            bounds = mesh_bounds(cfg, ship, step, pg, geom, cfg.zone_fluid)
            if bounds is None:
                print("PG: no bounds, skip")
            else:
                shared_bounds = shared_bounds or bounds

                def pg_scalar(cells, var):
                    return pg.point_query(cells, var)

                _run_point_queries("PG", geom.point_intersection, pg_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_line_queries("PG", geom.line_intersection, pg_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_plane_queries("PG", geom.plane_intersection, pg_scalar, bounds, cfg.duration_sec, cfg.variables)
            pg.close()

        if "iotdb" in backends:
            iotdb = make_iotdb(ship, step, cfg.zone_fluid)
            geom = make_geom_client(cfg, ship, step, iotdb, cfg.zone_fluid)
            bounds = mesh_bounds(cfg, ship, step, iotdb, geom, cfg.zone_fluid) or shared_bounds
            if bounds is None:
                print("IoTDB: no bounds, skip")
            else:

                def iotdb_scalar(cells, var):
                    return iotdb.point_query(cells, var)

                _run_point_queries("IoTDB", geom.point_intersection, iotdb_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_line_queries("IoTDB", geom.line_intersection, iotdb_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_plane_queries("IoTDB", geom.plane_intersection, iotdb_scalar, bounds, cfg.duration_sec, cfg.variables)
            iotdb.close()

        if "tiledb" in backends:
            tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.zone_fluid)
            geom = make_geom_client(cfg, ship, step, tiledb, cfg.zone_fluid)
            bounds = mesh_bounds(cfg, ship, step, tiledb, geom, cfg.zone_fluid) or shared_bounds
            if bounds is None:
                print("TileDB: no bounds, skip")
            else:

                def tdb_scalar(cells, var):
                    return tiledb.point_query(cells, var)

                _run_point_queries("TileDB", geom.point_intersection, tdb_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_line_queries("TileDB", geom.line_intersection, tdb_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_plane_queries("TileDB", geom.plane_intersection, tdb_scalar, bounds, cfg.duration_sec, cfg.variables)
            tiledb.close()

        if "vtk" in backends:
            vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.zone_fluid)
            bounds = mesh_bounds(cfg, ship, step, vtk, vtk, cfg.zone_fluid) or shared_bounds
            if bounds is None:
                print("VTK: no bounds, skip")
            else:

                def vtk_scalar(cells, var):
                    return vtk.point_query(cells, var)

                _run_point_queries("VTK", vtk.point_intersection, vtk_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_line_queries("VTK", vtk.line_intersection, vtk_scalar, bounds, cfg.duration_sec, cfg.variables)
                _run_plane_queries("VTK", vtk.plane_intersection, vtk_scalar, bounds, cfg.duration_sec, cfg.variables)
            vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W1: intersection + point query")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    backends = set(args.backend)
    for ship in cfg.ships:
        run_ship(cfg, ship, backends)


if __name__ == "__main__":
    main()
