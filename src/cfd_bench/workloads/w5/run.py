"""W5: Single-timestep streamline integration (delta_t=1.0)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, mesh_bounds
from cfd_bench.workloads.common.metrics import cal_next_point
from cfd_bench.core.observability import benchmark_progress
from cfd_bench.workloads.common.random_geom import random_start_point


# A transaction must never be able to outlive the benchmark indefinitely,
# even if a backend accidentally reports a hit forever.
_MAX_STREAMLINE_STEPS = 10_000
_MIN_SPEED = 1e-15


def _integrate_one_streamline(
    scalar_fn,
    intersect_fn,
    cid: int,
    coord: np.ndarray,
    *,
    deadline: float,
    delta_t: float,
    max_steps: int = _MAX_STREAMLINE_STEPS,
    reporter=None,
) -> bool:
    """Integrate one streamline.

    Returns ``True`` when the transaction reached a normal/safety terminal
    condition and ``False`` when the global benchmark deadline expired.
    """
    cur_cid = int(cid)
    cur_coord = np.asarray(coord, dtype=np.float64)

    for _ in range(max_steps):
        if time.monotonic() >= deadline:
            return False

        if reporter is not None:
            reporter.set_phase(f"velocity query cell={cur_cid}")
        u, v, w = scalar_fn(cur_cid)
        vel = np.array([u, v, w], dtype=np.float64)
        if not np.all(np.isfinite(vel)) or float(np.linalg.norm(vel)) <= _MIN_SPEED:
            return True

        nxt = np.asarray(cal_next_point(cur_coord, vel, delta_t), dtype=np.float64)
        if not np.all(np.isfinite(nxt)) or np.array_equal(nxt, cur_coord):
            return True

        if reporter is not None:
            reporter.set_phase("point intersection")
        nxt_cells = intersect_fn(np.array([nxt], dtype=np.float64))
        if len(nxt_cells) == 0:
            return True

        cur_cid = int(nxt_cells[0])
        cur_coord = nxt

    # Safety cap: treat a very long streamline as one completed transaction
    # rather than allowing a single transaction to run forever.
    return True


def _streamline(label, scalar_fn, intersect_fn, bounds, duration, delta_t=1.0, *, progress=False, progress_interval=5.0):
    txn = 0
    start = time.monotonic()
    deadline = start + float(duration)

    with benchmark_progress(f"{label} W5", duration, enabled=progress, interval=progress_interval) as prog:
        while time.monotonic() < deadline:
            try:
                prog.set_phase("find start point")
                cid, coord = random_start_point(intersect_fn, bounds, deadline=deadline)
            except TimeoutError:
                break

            completed = _integrate_one_streamline(
                scalar_fn,
                intersect_fn,
                cid,
                coord,
                deadline=deadline,
                delta_t=delta_t,
                reporter=prog,
            )
            if not completed:
                break
            txn += 1
            prog.transaction()

    print(f"{label} W5: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    shared_bounds = None

    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, pg, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, pg, geom, cfg.fluid_zone(ship))
        if bounds:

            def pg_scalar(cid):
                vel = pg.velocity_query([cid])[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _streamline("PG", pg_scalar, geom.point_intersection, bounds, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            shared_bounds = bounds
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, iotdb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def iot_scalar(cid):
                vel = iotdb.velocity_query([cid])[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _streamline("IoTDB", iot_scalar, geom.point_intersection, bounds, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, tiledb, geom, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def tdb_scalar(cid):
                vel = tiledb.velocity_query([cid])[0]
                return float(vel[0]), float(vel[1]), float(vel[2])

            _streamline("TileDB", tdb_scalar, geom.point_intersection, bounds, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.fluid_zone(ship))
        bounds = mesh_bounds(cfg, ship, step, vtk, vtk, cfg.fluid_zone(ship)) or shared_bounds
        if bounds:

            def vtk_scalar(cid):
                return (
                    float(vtk.point_query([cid], "U")[0]),
                    float(vtk.point_query([cid], "V")[0]),
                    float(vtk.point_query([cid], "W")[0]),
                )

            _streamline("VTK", vtk_scalar, vtk.point_intersection, bounds, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W5: streamline integration")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W5 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
