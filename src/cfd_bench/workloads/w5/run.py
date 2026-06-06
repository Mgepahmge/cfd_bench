"""W5: Single-timestep streamline integration (delta_t=1.0)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.metrics import cal_next_point
from cfd_bench.workloads.common.random_geom import random_start_point


def _streamline(label, scalar_fn, intersect_fn, bounds, duration, delta_t=1.0):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        cid, coord = random_start_point(intersect_fn, bounds)
        cur_cid, cur_coord = cid, coord
        while True:
            u, v, w = scalar_fn(cur_cid)
            vel = np.array([u, v, w], dtype=np.float64)
            nxt = cal_next_point(cur_coord, vel, delta_t)
            nxt_cells = intersect_fn(np.array([nxt], dtype=np.float64))
            if len(nxt_cells) == 0:
                break
            cur_cid = int(nxt_cells[0])
            cur_coord = nxt
        txn += 1
    print(f"{label} W5: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    shared_bounds = None

    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, pg, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, pg, geom, cfg.zone_fluid)
        if bounds:

            def pg_scalar(cid):
                return (
                    float(pg.point_query([cid], "U")[0]),
                    float(pg.point_query([cid], "V")[0]),
                    float(pg.point_query([cid], "W")[0]),
                )

            _streamline("PG", pg_scalar, geom.point_intersection, bounds, cfg.duration_sec)
            shared_bounds = bounds
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, iotdb, geom, cfg.zone_fluid) or shared_bounds
        if bounds:

            def iot_scalar(cid):
                return (
                    float(iotdb.point_query([cid], "U")[0]),
                    float(iotdb.point_query([cid], "V")[0]),
                    float(iotdb.point_query([cid], "W")[0]),
                )

            _streamline("IoTDB", iot_scalar, geom.point_intersection, bounds, cfg.duration_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, tiledb, geom, cfg.zone_fluid) or shared_bounds
        if bounds:

            def tdb_scalar(cid):
                return (
                    float(tiledb.point_query([cid], "U")[0]),
                    float(tiledb.point_query([cid], "V")[0]),
                    float(tiledb.point_query([cid], "W")[0]),
                )

            _streamline("TileDB", tdb_scalar, geom.point_intersection, bounds, cfg.duration_sec)
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, vtk, vtk, cfg.zone_fluid) or shared_bounds
        if bounds:

            def vtk_scalar(cid):
                return (
                    float(vtk.point_query([cid], "U")[0]),
                    float(vtk.point_query([cid], "V")[0]),
                    float(vtk.point_query([cid], "W")[0]),
                )

            _streamline("VTK", vtk_scalar, vtk.point_intersection, bounds, cfg.duration_sec)
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W5: streamline integration")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W5 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
