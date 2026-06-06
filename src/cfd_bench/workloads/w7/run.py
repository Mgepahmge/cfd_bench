"""W7: ROI Q-criterion computation."""

from __future__ import annotations

import argparse
import random
import time

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
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
    shared_bounds = None

    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, pg, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, pg, geom, cfg.zone_fluid)
        if bounds:
            _bench_db(
                "PG",
                geom.range_query_coord,
                lambda lo, hi: pg.compute_qcriterion_roi(lo, hi),
                bounds,
                cfg.duration_sec,
            )
            shared_bounds = bounds
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, iotdb, geom, cfg.zone_fluid) or shared_bounds
        if bounds:
            _bench_db(
                "IoTDB",
                geom.range_query_coord,
                lambda lo, hi: iotdb.compute_qcriterion_roi(lo, hi),
                bounds,
                cfg.duration_sec,
            )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.zone_fluid)
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, tiledb, geom, cfg.zone_fluid) or shared_bounds
        if bounds:
            _bench_db(
                "TileDB",
                geom.range_query_coord,
                lambda lo, hi: tiledb.compute_qcriterion_roi(lo, hi),
                bounds,
                cfg.duration_sec,
            )
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.zone_fluid)
        bounds = mesh_bounds(cfg, ship, step, vtk, vtk, cfg.zone_fluid) or shared_bounds
        if bounds:
            _bench_vtk("VTK", vtk, bounds, cfg.duration_sec)
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W7: Q-criterion ROI")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W7 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
