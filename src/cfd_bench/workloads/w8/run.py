"""W8: Variable range query (vortex / threshold cell selection)."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig


def random_var_range_vtk(vtk_mesh, attribute_name: str):
    arr = vtk_mesh.GetCellData().GetArray(attribute_name)
    if arr is None:
        raise ValueError(f"attribute {attribute_name} not found")
    vmin, vmax = arr.GetRange()
    lo, hi = sorted([random.uniform(vmin, vmax), random.uniform(vmin, vmax)])
    return lo, hi


def _bench(label, range_fn, geom, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(VARIABLES)
        if geom is not None:
            lo, hi = random_var_range_vtk(geom.vtk_mesh, var)
        else:
            lo, hi = 0.0, 1.0
        range_fn(lo, hi, var)
        txn += 1
    print(f"{label} W8: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    geom = make_vtk(cfg.vtk_dir, ship, step)

    if "postgresql" in backends:
        pg = make_pg(ship, "fluid")
        pg.set_step(step)
        _bench("PG", lambda lo, hi, v: pg.range_query_var(lo, hi, v), geom, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step)
        _bench("IoTDB", lambda lo, hi, v: iotdb.range_query_var(lo, hi, v), geom, cfg.duration_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root)
        _bench("TileDB", lambda lo, hi, v: tiledb.range_query_var(lo, hi, v), geom, cfg.duration_sec)
        tiledb.close()

    if "vtk" in backends:
        _bench("VTK", lambda lo, hi, v: geom.range_query_var(lo, hi, v), geom, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W8: variable range query")
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
            print(f"\n=== W8 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
