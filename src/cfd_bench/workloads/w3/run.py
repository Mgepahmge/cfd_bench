"""W3: Variable-range submesh + isosurface extraction."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig


def read_max_diffs(path: str) -> dict:
    out = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                out[row[0]] = float(row[1])
    return out


def _bench(label, scalar_fn, range_fn, extract_fn, iso_fn, n_cells, max_diffs, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(VARIABLES)
        delta = max_diffs[var]
        cid = random.randint(0, max(0, n_cells - 1))
        iso_val = float(scalar_fn([cid], var)[0])
        cells = range_fn(iso_val - delta, iso_val + delta, var)
        sub = extract_fn(cells)
        iso_fn(sub, var, iso_val)
        txn += 1
    print(f"{label} W3: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    delta_file = None
    for f in os.listdir(cfg.max_range_dir):
        if ship in f and f.endswith(f"_{step}_max_diffs.csv"):
            delta_file = os.path.join(cfg.max_range_dir, f)
            break
    if delta_file is None:
        print(f"no max_diffs for {ship} step {step}, skip")
        return
    max_diffs = read_max_diffs(delta_file)

    geom = make_vtk(cfg.vtk_dir, ship, step)
    n_cells = geom.vtk_mesh.GetNumberOfCells()

    if "postgresql" in backends:
        pg = make_pg(ship, "fluid")
        pg.set_step(step)
        _bench(
            "PG",
            lambda c, v: pg.point_query(c, v),
            lambda lo, hi, v: pg.range_query_var(lo, hi, v),
            lambda cells: geom.extract_submesh(cells),
            lambda mesh, v, val: geom.isosurface_extraction(v, val) if mesh else None,
            n_cells,
            max_diffs,
            cfg.duration_sec,
        )
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step)
        _bench(
            "IoTDB",
            lambda c, v: iotdb.point_query(c, v),
            lambda lo, hi, v: iotdb.range_query_var(lo, hi, v),
            lambda cells: iotdb.extract_submesh(cells),
            lambda mesh, v, val: iotdb.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
        )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root)
        _bench(
            "TileDB",
            lambda c, v: tiledb.point_query(c, v),
            lambda lo, hi, v: tiledb.range_query_var(lo, hi, v),
            lambda cells: tiledb.extract_submesh(cells),
            lambda mesh, v, val: tiledb.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
        )
        tiledb.close()

    if "vtk" in backends:
        _bench(
            "VTK",
            lambda c, v: geom.point_query(c, v),
            lambda lo, hi, v: geom.range_query_var(lo, hi, v),
            lambda cells: geom.extract_submesh(cells),
            lambda mesh, v, val: geom.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
        )


def main(ships=None):
    ap = argparse.ArgumentParser(description="W3: isosurface from variable range")
    ap.add_argument("--ships", nargs="+", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--backend", nargs="+", default=["postgresql", "iotdb", "tiledb", "vtk"])
    ap.add_argument("--vtk-dir", default="../vtk_dir")
    ap.add_argument("--tiledb-root", default="../TileDB_Instances")
    ap.add_argument("--max-range-dir", default="../Max_Range")
    args = ap.parse_args()

    cfg = WorkloadConfig(
        ships=args.ships or ships or WorkloadConfig().ships,
        duration_sec=args.duration,
        vtk_dir=args.vtk_dir,
        tiledb_root=args.tiledb_root,
        max_range_dir=args.max_range_dir,
    )
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W3 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
