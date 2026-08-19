"""W7: ROI Q-criterion computation."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import is_h5_client, make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.random_geom import random_coord_range
from cfd_bench.core.observability import benchmark_progress


def _bench_db(label, coord_fn, qc_fn, bounds, duration, *, max_hit_attempts=None, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W7", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            attempts = 0
            cells = np.zeros((0,), dtype=np.int32)
            while True:
                prog.set_phase("coordinate range query")
                lo, hi = random_coord_range(bounds)
                cells = coord_fn(lo, hi)
                attempts += 1
                if len(cells) > 0 or (max_hit_attempts is not None and attempts >= int(max_hit_attempts)):
                    break
            if len(cells) > 0:
                prog.set_phase(f"Q-criterion ({len(cells)} ROI cells)")
                qc_fn(lo, hi)
            txn += 1
            prog.transaction()
    print(f"{label} W7: {txn} txns in {duration}s")


def _bench_vtk(label, geom, bounds, duration, *, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W7", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            while True:
                prog.set_phase("coordinate range query")
                lo, hi = random_coord_range(bounds)
                cells = geom.range_query_coord(lo, hi)
                if len(cells) > 0:
                    break
            prog.set_phase(f"extract submesh ({len(cells)} cells)")
            sub = geom.extract_submesh(cells)
            if sub is not None:
                try:
                    from vtk import vtkDataObject, vtkGradientFilter

                    prog.set_phase("VTK gradient/Q filter")
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
            prog.transaction()
    print(f"{label} W7: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    shared_bounds = None

    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, pg, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, pg, geom, cfg.fluid_zone(ship))
        if bounds:
            _bench_db(
                "PG",
                geom.range_query_coord,
                lambda lo, hi: pg.compute_qcriterion_roi(lo, hi),
                bounds,
                cfg.duration_sec,
                max_hit_attempts=None if is_h5_client(pg) else 16,
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
            shared_bounds = bounds
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, iotdb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:
            _bench_db(
                "IoTDB",
                geom.range_query_coord,
                lambda lo, hi: iotdb.compute_qcriterion_roi(lo, hi),
                bounds,
                cfg.duration_sec,
                max_hit_attempts=None if is_h5_client(iotdb) else 16,
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, tiledb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:
            _bench_db(
                "TileDB",
                geom.range_query_coord,
                lambda lo, hi: tiledb.compute_qcriterion_roi(lo, hi),
                bounds,
                cfg.duration_sec,
                max_hit_attempts=None if is_h5_client(tiledb) else 16,
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, vtk, vtk, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:
            _bench_db(
                "VTK", vtk.range_query_coord,
                lambda lo, hi: vtk.compute_qcriterion_roi(lo, hi),
                bounds, cfg.duration_sec,
                max_hit_attempts=64,
                progress=cfg.progress, progress_interval=cfg.progress_interval_sec,
            )
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
