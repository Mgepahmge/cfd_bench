"""W3: Variable-range submesh + isosurface extraction."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import cell_count, make_geom_client


def read_max_diffs(path: str) -> dict:
    out = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                out[row[0]] = float(row[1])
    return out


def _bench(label, scalar_fn, range_fn, extract_fn, iso_fn, n_cells, max_diffs, duration, variables):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        var = random.choice(variables)
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

    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, pg, cfg.zone_fluid)
        n_cells = cell_count(geom)
        _bench(
            "PG",
            lambda c, v: pg.point_query(c, v),
            lambda lo, hi, v: pg.range_query_var(lo, hi, v),
            lambda cells: geom.extract_submesh(cells),
            lambda mesh, v, val: geom.isosurface_extraction(v, val) if mesh else None,
            n_cells,
            max_diffs,
            cfg.duration_sec,
            cfg.variables,
        )
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.zone_fluid)
        n_cells = cell_count(geom)
        _bench(
            "IoTDB",
            lambda c, v: iotdb.point_query(c, v),
            lambda lo, hi, v: iotdb.range_query_var(lo, hi, v),
            lambda cells: geom.extract_submesh(cells),
            lambda mesh, v, val: geom.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
            cfg.variables,
        )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.zone_fluid)
        n_cells = cell_count(geom)
        _bench(
            "TileDB",
            lambda c, v: tiledb.point_query(c, v),
            lambda lo, hi, v: tiledb.range_query_var(lo, hi, v),
            lambda cells: geom.extract_submesh(cells),
            lambda mesh, v, val: geom.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
            cfg.variables,
        )
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.zone_fluid)
        n_cells = cell_count(vtk)
        _bench(
            "VTK",
            lambda c, v: vtk.point_query(c, v),
            lambda lo, hi, v: vtk.range_query_var(lo, hi, v),
            lambda cells: vtk.extract_submesh(cells),
            lambda mesh, v, val: vtk.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
            cfg.variables,
        )
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W3: isosurface from variable range")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W3 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
