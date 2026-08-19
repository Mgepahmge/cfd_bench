"""W1: Point / line / plane intersection + scalar query."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import is_h5_client, make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.metrics import aggregation
from cfd_bench.core.observability import benchmark_progress
from cfd_bench.workloads.common.random_geom import (
    random_line_in_bbox,
    random_plane_in_bbox,
    random_points_in_bbox,
)


def _run_point_queries(label, query_fn, scalar_fn, bounds, duration, variables, *, max_hit_attempts=None, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W1/point", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            var = random.choice(variables)
            attempts = 0
            cells = np.zeros((0,), dtype=np.int32)
            while True:
                prog.set_phase("sample points + point intersection")
                pts = random_points_in_bbox(bounds)
                cells = query_fn(pts)
                attempts += 1
                if len(cells) > 0 or (max_hit_attempts is not None and attempts >= int(max_hit_attempts)):
                    break
            if len(cells) > 0:
                prog.set_phase(f"scalar query {var}")
                vals = scalar_fn(cells, var)
                prog.set_phase("aggregation")
                aggregation(vals)
            txn += 1
            prog.transaction()
    print(f"{label} point intersection: {txn} txns in {duration}s")


def _run_line_queries(label, query_fn, scalar_fn, bounds, duration, variables, *, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W1/line", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            var = random.choice(variables)
            prog.set_phase("line intersection")
            start, end = random_line_in_bbox(bounds)
            cells = query_fn(start, end)
            if len(cells) > 0:
                prog.set_phase(f"scalar query {var} ({len(cells)} cells)")
                scalar_fn(cells, var)
            txn += 1
            prog.transaction()
    print(f"{label} line intersection: {txn} txns in {duration}s")


def _run_plane_queries(label, query_fn, scalar_fn, bounds, duration, variables, *, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W1/plane", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            var = random.choice(variables)
            prog.set_phase("plane intersection")
            origin, normal = random_plane_in_bbox(bounds)
            cells = query_fn(origin, normal)
            if len(cells) > 0:
                prog.set_phase(f"scalar query {var} ({len(cells)} cells)")
                scalar_fn(cells, var)
            txn += 1
            prog.transaction()
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
            pg = make_pg(ship, step, zone=cfg.fluid_zone(ship))
            geom = make_geom_client(cfg, ship, step, pg, cfg.fluid_zone(ship))
            bounds = mesh_bounds(cfg, ship, step, pg, geom, cfg.fluid_zone(ship))
            if bounds is None:
                print("PG: no bounds, skip")
            else:
                shared_bounds = shared_bounds or bounds

                def pg_scalar(cells, var):
                    return pg.point_query(cells, var)

                _run_point_queries("PG", geom.point_intersection, pg_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None if is_h5_client(pg) else 16, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_line_queries("PG", geom.line_intersection, pg_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_plane_queries("PG", geom.plane_intersection, pg_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            pg.close()

        if "iotdb" in backends:
            iotdb = make_iotdb(ship, step, cfg.fluid_zone(ship))
            geom = make_geom_client(cfg, ship, step, iotdb, cfg.fluid_zone(ship))
            bounds = mesh_bounds(cfg, ship, step, iotdb, geom, cfg.fluid_zone(ship)) or shared_bounds
            if bounds is None:
                print("IoTDB: no bounds, skip")
            else:

                def iotdb_scalar(cells, var):
                    return iotdb.point_query(cells, var)

                _run_point_queries("IoTDB", geom.point_intersection, iotdb_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None if is_h5_client(iotdb) else 16, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_line_queries("IoTDB", geom.line_intersection, iotdb_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_plane_queries("IoTDB", geom.plane_intersection, iotdb_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            iotdb.close()

        if "tiledb" in backends:
            tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship))
            geom = make_geom_client(cfg, ship, step, tiledb, cfg.fluid_zone(ship))
            bounds = mesh_bounds(cfg, ship, step, tiledb, geom, cfg.fluid_zone(ship)) or shared_bounds
            if bounds is None:
                print("TileDB: no bounds, skip")
            else:

                def tdb_scalar(cells, var):
                    return tiledb.point_query(cells, var)

                _run_point_queries("TileDB", geom.point_intersection, tdb_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None if is_h5_client(tiledb) else 16, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_line_queries("TileDB", geom.line_intersection, tdb_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_plane_queries("TileDB", geom.plane_intersection, tdb_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            tiledb.close()

        if "vtk" in backends:
            vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.fluid_zone(ship))
            bounds = mesh_bounds(cfg, ship, step, vtk, vtk, cfg.fluid_zone(ship)) or shared_bounds
            if bounds is None:
                print("VTK: no bounds, skip")
            else:

                def vtk_scalar(cells, var):
                    return vtk.point_query(cells, var)

                _run_point_queries("VTK", vtk.point_intersection, vtk_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_line_queries("VTK", vtk.line_intersection, vtk_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
                _run_plane_queries("VTK", vtk.plane_intersection, vtk_scalar, bounds, cfg.duration_sec, cfg.valid_variables(ship), progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
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
