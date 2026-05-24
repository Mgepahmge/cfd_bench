"""W1: Point / line / plane intersection + scalar query."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk, mesh_bounds_from_client
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.metrics import aggregation
from cfd_bench.workloads.common.random_geom import (
    random_line_in_bbox,
    random_plane_in_bbox,
    random_points_in_bbox,
)


def _run_point_queries(label, query_fn, scalar_fn, bounds, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(VARIABLES)
        while True:
            pts = random_points_in_bbox(bounds)
            cells = query_fn(pts)
            if len(cells) > 0:
                break
        vals = scalar_fn(cells, var)
        aggregation(vals)
        txn += 1
    print(f"{label} point intersection: {txn} txns in {duration}s")


def _run_line_queries(label, query_fn, scalar_fn, bounds, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(VARIABLES)
        start, end = random_line_in_bbox(bounds)
        cells = query_fn(start, end)
        if len(cells) > 0:
            scalar_fn(cells, var)
        txn += 1
    print(f"{label} line intersection: {txn} txns in {duration}s")


def _run_plane_queries(label, query_fn, scalar_fn, bounds, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(VARIABLES)
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
        geom = make_vtk(cfg.vtk_dir, ship, step)
        bounds = mesh_bounds_from_client(geom)
        if bounds is None:
            print("no bounds, skip")
            continue

        if "postgresql" in backends:
            pg = make_pg(ship, "fluid")
            pg.set_step(step)

            def pg_scalar(cells, var):
                return pg.point_query(cells, var)

            _run_point_queries("PG", lambda p: geom.point_intersection(p), pg_scalar, bounds, cfg.duration_sec)
            _run_line_queries("PG", geom.line_intersection, pg_scalar, bounds, cfg.duration_sec)
            _run_plane_queries("PG", geom.plane_intersection, pg_scalar, bounds, cfg.duration_sec)
            pg.close()

        if "iotdb" in backends:
            iotdb = make_iotdb(ship, step)
            db_bounds = mesh_bounds_from_client(iotdb) or bounds

            def iotdb_scalar(cells, var):
                return iotdb.point_query(cells, var)

            _run_point_queries("IoTDB", iotdb.point_intersection, iotdb_scalar, db_bounds, cfg.duration_sec)
            _run_line_queries("IoTDB", iotdb.line_intersection, iotdb_scalar, db_bounds, cfg.duration_sec)
            _run_plane_queries("IoTDB", iotdb.plane_intersection, iotdb_scalar, db_bounds, cfg.duration_sec)
            iotdb.close()

        if "tiledb" in backends:
            tiledb = make_tiledb(ship, step, cfg.tiledb_root)
            db_bounds = mesh_bounds_from_client(tiledb) or bounds

            def tdb_scalar(cells, var):
                return tiledb.point_query(cells, var)

            _run_point_queries("TileDB", tiledb.point_intersection, tdb_scalar, db_bounds, cfg.duration_sec)
            _run_line_queries("TileDB", tiledb.line_intersection, tdb_scalar, db_bounds, cfg.duration_sec)
            _run_plane_queries("TileDB", tiledb.plane_intersection, tdb_scalar, db_bounds, cfg.duration_sec)
            tiledb.close()

        if "vtk" in backends:
            vtk = make_vtk(cfg.vtk_dir, ship, step)

            def vtk_scalar(cells, var):
                return vtk.point_query(cells, var)

            _run_point_queries("VTK", vtk.point_intersection, vtk_scalar, bounds, cfg.duration_sec)
            _run_line_queries("VTK", vtk.line_intersection, vtk_scalar, bounds, cfg.duration_sec)
            _run_plane_queries("VTK", vtk.plane_intersection, vtk_scalar, bounds, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W1: intersection + point query")
    ap.add_argument("--ships", nargs="+", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--backend", nargs="+", default=["postgresql", "iotdb", "tiledb", "vtk"])
    ap.add_argument("--vtk-dir", default="../vtk_dir")
    ap.add_argument("--tiledb-root", default="../TileDB_Instances")
    args = ap.parse_args()

    cfg = WorkloadConfig(
        ships=args.ships or ships or WorkloadConfig().ships,
        duration_sec=args.duration,
        vtk_dir=args.vtk_dir,
        tiledb_root=args.tiledb_root,
    )
    backends = set(args.backend)
    for ship in cfg.ships:
        run_ship(cfg, ship, backends)


if __name__ == "__main__":
    main()
