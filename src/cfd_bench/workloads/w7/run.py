"""W7: ROI Q-criterion computation."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk, mesh_bounds_from_client
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.random_geom import random_coord_range


def _bench_db(label, coord_fn, qc_fn, bounds, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        while True:
            lo, hi = random_coord_range(bounds)
            cells = coord_fn(lo, hi)
            if len(cells) > 0:
                break
        qc_fn(lo, hi)
        txn += 1
    print(f"{label} W7: {txn} txns in {duration}s")


def _bench_vtk(label, geom, bounds, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        while True:
            lo, hi = random_coord_range(bounds)
            cells = geom.range_query_coord(lo, hi)
            if len(cells) > 0:
                break
        sub = geom.extract_submesh(cells)
        if sub is not None:
            geom.vtk_mesh = sub
            try:
                from vtk import vtkDataObject, vtkGradientFilter

                vel = sub.GetPointData().GetArray("Velocity")
                if vel:
                    gf = vtkGradientFilter()
                    gf.SetInputData(sub)
                    gf.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
                    gf.SetComputeQCriterion(True)
                    gf.Update()
            except Exception:
                pass
        txn += 1
    print(f"{label} W7: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    geom = make_vtk(cfg.vtk_dir, ship, step)
    bounds = mesh_bounds_from_client(geom)
    if bounds is None:
        return

    if "postgresql" in backends:
        pg = make_pg(ship, "fluid")
        pg.set_step(step)
        _bench_db(
            "PG",
            pg.range_query_coord,
            lambda lo, hi: geom.extract_submesh(pg.range_query_coord(lo, hi)),
            bounds,
            cfg.duration_sec,
        )
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step)
        db_bounds = mesh_bounds_from_client(iotdb) or bounds
        _bench_db(
            "IoTDB",
            iotdb.range_query_coord,
            lambda lo, hi: iotdb.compute_qcriterion_roi(lo, hi),
            db_bounds,
            cfg.duration_sec,
        )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root)
        db_bounds = mesh_bounds_from_client(tiledb) or bounds
        _bench_db(
            "TileDB",
            tiledb.range_query_coord,
            lambda lo, hi: tiledb.compute_qcriterion_roi(lo, hi),
            db_bounds,
            cfg.duration_sec,
        )
        tiledb.close()

    if "vtk" in backends:
        _bench_vtk("VTK", geom, bounds, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W7: Q-criterion ROI")
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
            print(f"\n=== W7 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
