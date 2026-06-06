"""W6: Hull surface pressure integration (normals + scalar query)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import cell_count, make_geom_client
from cfd_bench.workloads.common.metrics import calculate_force


def _bench(label, norm_fn, pressure_fn, n_cells, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        normals = norm_fn()
        cells = np.arange(n_cells, dtype=np.int32)
        pressures = pressure_fn(cells)
        calculate_force(normals, pressures)
        txn += 1
    print(f"{label} W6: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    if "postgresql" in backends:
        pg = make_pg(ship, step, zone=cfg.zone_hull)
        geom = make_geom_client(cfg, ship, step, pg, cfg.zone_hull)
        n_cells = cell_count(geom)
        _bench("PG", lambda: geom.surface_norm(), lambda c: pg.point_query(c, "P"), n_cells, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, zone=cfg.zone_hull)
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.zone_hull)
        n_cells = cell_count(geom)
        sub = geom.extract_submesh(list(range(n_cells)))
        _bench(
            "IoTDB",
            lambda: geom.surface_norm(sub),
            lambda c: iotdb.point_query(c, "P"),
            n_cells,
            cfg.duration_sec,
        )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, zone=cfg.zone_hull)
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.zone_hull)
        n_cells = cell_count(geom)
        sub = geom.extract_submesh(list(range(n_cells)))
        _bench(
            "TileDB",
            lambda: geom.surface_norm(sub),
            lambda c: tiledb.point_query(c, "P"),
            n_cells,
            cfg.duration_sec,
        )
        tiledb.close()

    if "vtk" in backends:
        hull = make_vtk(cfg.vtk_hull_dir, ship, step, zone=cfg.zone_hull)
        n_cells = cell_count(hull)
        _bench("VTK", lambda: hull.surface_norm(), lambda c: hull.point_query(c, "P"), n_cells, cfg.duration_sec)
        hull.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W6: hull force integration")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W6 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
