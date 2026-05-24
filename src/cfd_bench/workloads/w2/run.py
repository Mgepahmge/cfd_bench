"""W2: Coordinate range query + multi-timestep point query + aggregation."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk, mesh_bounds_from_client
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.metrics import aggregation_w2
from cfd_bench.workloads.common.random_geom import random_coord_range


def _bench(label, coord_fn, scalar_fn, bounds, steps, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(VARIABLES)
        while True:
            lo, hi = random_coord_range(bounds)
            cells = coord_fn(lo, hi)
            if len(cells) > 0:
                break
        result = []
        for step in steps:
            result.extend(scalar_fn(cells, var, step))
        aggregation_w2(np.array(result, dtype=np.float64))
        txn += 1
    print(f"{label} W2: {txn} txns in {duration}s")


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    steps = cfg.valid_steps(ship)
    geom = make_vtk(cfg.vtk_dir, ship, 200)
    bounds = mesh_bounds_from_client(geom)
    if bounds is None:
        return

    if "postgresql" in backends:
        pg = make_pg(ship, "fluid")

        def pg_scalar(cells, var, step):
            pg.set_step(step)
            return pg.point_query(cells, var).tolist()

        _bench("PG", pg.range_query_coord, pg_scalar, bounds, steps, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, steps[0])
        db_bounds = mesh_bounds_from_client(iotdb) or bounds

        def iot_scalar(cells, var, step):
            iotdb.ctx.step = step
            return iotdb.point_query(cells, var).tolist()

        _bench("IoTDB", lambda lo, hi: iotdb.range_query_coord(lo, hi), iot_scalar, db_bounds, steps, cfg.duration_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, steps[0], cfg.tiledb_root)
        db_bounds = mesh_bounds_from_client(tiledb) or bounds

        def tdb_scalar(cells, var, step):
            tiledb.ctx.step = step
            return tiledb.point_query(cells, var).tolist()

        _bench("TileDB", lambda lo, hi: tiledb.range_query_coord(lo, hi), tdb_scalar, db_bounds, steps, cfg.duration_sec)
        tiledb.close()

    if "vtk" in backends:

        def vtk_scalar(cells, var, step):
            vtk = make_vtk(cfg.vtk_dir, ship, step)
            return vtk.point_query(cells, var).tolist()

        _bench("VTK", lambda lo, hi: geom.range_query_coord(lo, hi), vtk_scalar, bounds, steps, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W2: coord range + multi-step query")
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
        print(f"\n=== W2 ship={ship} ===")
        run_ship(cfg, ship, set(args.backend))


if __name__ == "__main__":
    main()
