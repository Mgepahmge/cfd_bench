"""W2: Coordinate range query + multi-timestep point query + aggregation."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import is_h5_client, make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.metrics import aggregation_w2
from cfd_bench.core.observability import benchmark_progress
from cfd_bench.workloads.common.random_geom import random_coord_range


def _bench(label, coord_fn, scalar_fn, bounds, steps, duration, variables, *, max_hit_attempts=None, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W2", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            var = random.choice(variables)
            attempts = 0
            cells = np.zeros((0,), dtype=np.int32)
            while True:
                prog.set_phase("coordinate range query")
                lo, hi = random_coord_range(bounds)
                cells = coord_fn(lo, hi)
                attempts += 1
                if len(cells) > 0 or (max_hit_attempts is not None and attempts >= int(max_hit_attempts)):
                    break
            if len(cells) == 0:
                txn += 1
                prog.transaction()
                continue
            chunks = []
            for step in steps:
                prog.set_phase(f"scalar query {var} step={step} ({len(cells)} cells)")
                chunks.append(np.asarray(scalar_fn(cells, var, step), dtype=np.float64).reshape(-1))
            result = np.concatenate(chunks) if len(chunks) > 1 else (chunks[0] if chunks else np.zeros((0,), dtype=np.float64))
            prog.set_phase("aggregation")
            aggregation_w2(result)
            txn += 1
            prog.transaction()
    print(f"{label} W2: {txn} txns in {duration}s")


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    steps = cfg.valid_steps(ship)
    shared_bounds = None
    ref_step = steps[0]

    if "postgresql" in backends:
        pg = make_pg(ship, ref_step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, ref_step, pg, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, pg, geom, cfg.fluid_zone(ship))
        if bounds:

            def pg_scalar(cells, var, step):
                pg.ctx.step = int(step)
                pg._sync_timestep(int(step))
                return pg.point_query(cells, var)

            _bench("PG", pg.range_query_coord, pg_scalar, bounds, steps, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None if is_h5_client(pg) else 16, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            shared_bounds = bounds
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, ref_step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, ref_step, iotdb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, iotdb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def iot_scalar(cells, var, step):
                iotdb.ctx.step = step
                return iotdb.point_query(cells, var)

            _bench("IoTDB", geom.range_query_coord, iot_scalar, bounds, steps, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None if is_h5_client(iotdb) else 16, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, ref_step, cfg.tiledb_root, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, ref_step, tiledb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, tiledb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def tdb_scalar(cells, var, step):
                tiledb.ctx.step = step
                return tiledb.point_query(cells, var)

            _bench("TileDB", geom.range_query_coord, tdb_scalar, bounds, steps, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None if is_h5_client(tiledb) else 16, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, ref_step, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, vtk, vtk, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def vtk_scalar(cells, var, step):
                v = make_vtk(cfg.vtk_dir, ship, step, cfg.fluid_zone(ship))
                try:
                    return v.point_query(cells, var)
                finally:
                    v.close()

            _bench("VTK", vtk.range_query_coord, vtk_scalar, bounds, steps, cfg.duration_sec, cfg.valid_variables(ship), max_hit_attempts=None, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W2: coord range + multi-step query")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        print(f"\n=== W2 ship={ship} ===")
        run_ship(cfg, ship, set(args.backend))


if __name__ == "__main__":
    main()
