"""W4: Multi-timestep particle advection (delta_t=0.01)."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cfd_bench.workloads.common.backends import is_h5_client, make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.metrics import cal_next_point
from cfd_bench.core.observability import benchmark_progress
from cfd_bench.core.results import emit_benchmark_result
from cfd_bench.workloads.common.random_geom import random_start_point


def _advect(label, scalar_fn, intersect_fn, bounds, steps, duration, delta_t=0.01, *, max_start_attempts=None, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W4", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            prog.set_phase("find start point")
            try:
                cid, coord = random_start_point(intersect_fn, bounds, max_attempts=max_start_attempts)
            except LookupError:
                continue
            cur_cid, cur_coord = cid, coord
            for step in steps:
                prog.set_phase(f"velocity query step={step}")
                u, v, w = scalar_fn(cur_cid, step)
                vel = np.array([u, v, w], dtype=np.float64)
                nxt = cal_next_point(cur_coord, vel, delta_t)
                prog.set_phase(f"point intersection step={step}")
                nxt_cells = intersect_fn(np.array([nxt], dtype=np.float64))
                if len(nxt_cells) == 0:
                    break
                cur_cid = int(nxt_cells[0])
                cur_coord = nxt
            txn += 1
            prog.transaction()
    emit_benchmark_result(
        f"{label} W4: {txn} txns in {duration}s",
        backend=label, operation="particle_advection_multi_step",
        transactions=txn, duration_sec=duration,
        details=f"steps={','.join(str(s) for s in steps)}",
    )


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    steps = cfg.valid_steps(ship)
    ref_step = steps[0]
    shared_bounds = None

    if "postgresql" in backends:
        pg = make_pg(ship, ref_step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, ref_step, pg, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, pg, geom, cfg.fluid_zone(ship))
        if bounds:

            def pg_scalar(cid, step):
                pg.ctx.step = int(step)
                pg._sync_timestep(int(step))
                vel = pg.velocity_query([cid], step=step)[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _advect("PG", pg_scalar, geom.point_intersection, bounds, steps, cfg.duration_sec, max_start_attempts=None if is_h5_client(pg) else 64, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            shared_bounds = bounds
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, ref_step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, ref_step, iotdb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, iotdb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def iot_scalar(cid, step):
                iotdb.ctx.step = step
                vel = iotdb.velocity_query([cid], step=step)[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _advect("IoTDB", iot_scalar, geom.point_intersection, bounds, steps, cfg.duration_sec, max_start_attempts=None if is_h5_client(iotdb) else 64, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, ref_step, cfg.tiledb_root, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, ref_step, tiledb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, tiledb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def tdb_scalar(cid, step):
                tiledb.ctx.step = step
                vel = tiledb.velocity_query([cid], step=step)[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _advect("TileDB", tdb_scalar, geom.point_intersection, bounds, steps, cfg.duration_sec, max_start_attempts=None if is_h5_client(tiledb) else 64, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, ref_step, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, ref_step, vtk, vtk, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def vtk_scalar(cid, step):
                vel = vtk.velocity_query([cid], step=step)[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _advect(
                "VTK", vtk_scalar, vtk.point_intersection, bounds, steps, cfg.duration_sec,
                max_start_attempts=128,
                progress=cfg.progress, progress_interval=cfg.progress_interval_sec,
            )
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W4: multi-timestep advection")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        print(f"\n=== W4 ship={ship} ===")
        run_ship(cfg, ship, set(args.backend))


if __name__ == "__main__":
    main()
