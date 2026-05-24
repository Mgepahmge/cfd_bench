"""W4: Multi-timestep particle advection (delta_t=0.01)."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk, mesh_bounds_from_client
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.metrics import cal_next_point
from cfd_bench.workloads.common.random_geom import random_start_point


def _advect(label, scalar_fn, intersect_fn, bounds, steps, duration, delta_t=0.01):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        cid, coord = random_start_point(intersect_fn, bounds)
        traj = [coord]
        cur_cid, cur_coord = cid, coord
        for step in steps:
            u, v, w = scalar_fn(cur_cid, step)
            vel = np.array([u, v, w], dtype=np.float64)
            nxt = cal_next_point(cur_coord, vel, delta_t)
            nxt_cells = intersect_fn(np.array([nxt], dtype=np.float64))
            if len(nxt_cells) == 0:
                break
            cur_cid = int(nxt_cells[0])
            cur_coord = nxt
            traj.append(nxt)
        txn += 1
    print(f"{label} W4: {txn} txns in {duration}s")


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    steps = cfg.valid_steps(ship)
    geom = make_vtk(cfg.vtk_dir, ship, 200)
    bounds = mesh_bounds_from_client(geom)
    if bounds is None:
        return

    if "postgresql" in backends:
        pg = make_pg(ship, "fluid")

        def pg_scalar(cid, step):
            pg.set_step(step)
            u, v, w = pg.point_query([cid], "U")[0], pg.point_query([cid], "V")[0], pg.point_query([cid], "W")[0]
            return u, v, w

        _advect("PG", pg_scalar, geom.point_intersection, bounds, steps, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, steps[0])

        def iot_scalar(cid, step):
            iotdb.ctx.step = step
            return (
                float(iotdb.point_query([cid], "U")[0]),
                float(iotdb.point_query([cid], "V")[0]),
                float(iotdb.point_query([cid], "W")[0]),
            )

        _advect("IoTDB", iot_scalar, iotdb.point_intersection, bounds, steps, cfg.duration_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, steps[0], cfg.tiledb_root)

        def tdb_scalar(cid, step):
            tiledb.ctx.step = step
            return (
                float(tiledb.point_query([cid], "U")[0]),
                float(tiledb.point_query([cid], "V")[0]),
                float(tiledb.point_query([cid], "W")[0]),
            )

        _advect("TileDB", tdb_scalar, tiledb.point_intersection, bounds, steps, cfg.duration_sec)
        tiledb.close()

    if "vtk" in backends:

        def vtk_scalar(cid, step):
            vtk = make_vtk(cfg.vtk_dir, ship, step)
            return (
                float(vtk.point_query([cid], "U")[0]),
                float(vtk.point_query([cid], "V")[0]),
                float(vtk.point_query([cid], "W")[0]),
            )

        _advect("VTK", vtk_scalar, geom.point_intersection, bounds, steps, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W4: multi-timestep advection")
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
    for ship in cfg.ships:
        print(f"\n=== W4 ship={ship} ===")
        run_ship(cfg, ship, set(args.backend))


if __name__ == "__main__":
    main()
