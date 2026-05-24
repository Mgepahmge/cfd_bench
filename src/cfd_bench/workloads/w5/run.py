"""W5: Single-timestep streamline integration (delta_t=1.0)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk, mesh_bounds_from_client
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.metrics import cal_next_point
from cfd_bench.workloads.common.random_geom import random_start_point


def _streamline(label, scalar_fn, intersect_fn, bounds, duration, delta_t=1.0):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        cid, coord = random_start_point(intersect_fn, bounds)
        line = [coord]
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
            line.append(nxt)
        txn += 1
    print(f"{label} W5: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    geom = make_vtk(cfg.vtk_dir, ship, 200)
    bounds = mesh_bounds_from_client(geom)
    if bounds is None:
        return

    if "postgresql" in backends:
        pg = make_pg(ship, "fluid")
        pg.set_step(step)

        def pg_scalar(cid):
            return (
                float(pg.point_query([cid], "U")[0]),
                float(pg.point_query([cid], "V")[0]),
                float(pg.point_query([cid], "W")[0]),
            )

        _streamline("PG", pg_scalar, geom.point_intersection, bounds, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step)

        def iot_scalar(cid):
            return (
                float(iotdb.point_query([cid], "U")[0]),
                float(iotdb.point_query([cid], "V")[0]),
                float(iotdb.point_query([cid], "W")[0]),
            )

        _streamline("IoTDB", iot_scalar, iotdb.point_intersection, bounds, cfg.duration_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root)

        def tdb_scalar(cid):
            return (
                float(tiledb.point_query([cid], "U")[0]),
                float(tiledb.point_query([cid], "V")[0]),
                float(tiledb.point_query([cid], "W")[0]),
            )

        _streamline("TileDB", tdb_scalar, tiledb.point_intersection, bounds, cfg.duration_sec)
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step)

        def vtk_scalar(cid):
            return (
                float(vtk.point_query([cid], "U")[0]),
                float(vtk.point_query([cid], "V")[0]),
                float(vtk.point_query([cid], "W")[0]),
            )

        _streamline("VTK", vtk_scalar, vtk.point_intersection, bounds, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W5: streamline integration")
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
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W5 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
