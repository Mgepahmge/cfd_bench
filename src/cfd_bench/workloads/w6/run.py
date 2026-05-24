"""W6: Hull surface pressure integration (normals + scalar query)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.config import WorkloadConfig
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
    hull = make_vtk(cfg.vtk_hull_dir, ship, step, zone=cfg.zone_hull)
    n_cells = hull.vtk_mesh.GetNumberOfCells()
    cells = np.arange(n_cells, dtype=np.int32)

    if "postgresql" in backends:
        pg = make_pg(ship, "hull")
        pg.set_step(step)
        _bench("PG", lambda: hull.surface_norm(), lambda c: pg.point_query(c, "P"), n_cells, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship + "hull", step, zone=cfg.zone_hull)
        sub = iotdb.extract_submesh(list(range(n_cells)))

        def iot_norm():
            try:
                return iotdb.surface_norm(sub)
            except TypeError:
                return hull.surface_norm()

        _bench("IoTDB", iot_norm, lambda c: iotdb.point_query(c, "P"), n_cells, cfg.duration_sec)
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, zone=cfg.zone_hull)
        sub = tiledb.extract_submesh(list(range(min(n_cells, len(tiledb.runtime.ensure_cells(ship, cfg.zone_hull).cells)))))

        _bench(
            "TileDB",
            lambda: tiledb.surface_norm(sub),
            lambda c: tiledb.point_query(c, "P"),
            n_cells,
            cfg.duration_sec,
        )
        tiledb.close()

    if "vtk" in backends:
        _bench("VTK", lambda: hull.surface_norm(), lambda c: hull.point_query(c, "P"), n_cells, cfg.duration_sec)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W6: hull force integration")
    ap.add_argument("--ships", nargs="+", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--backend", nargs="+", default=["postgresql", "iotdb", "tiledb", "vtk"])
    ap.add_argument("--vtk-dir", default="../vtk_dir")
    ap.add_argument("--vtk-hull-dir", default="../vtk_hull_dir")
    ap.add_argument("--tiledb-root", default="../TileDB_Instances")
    args = ap.parse_args()

    cfg = WorkloadConfig(
        ships=args.ships or ships or WorkloadConfig().ships,
        duration_sec=args.duration,
        vtk_dir=args.vtk_dir,
        vtk_hull_dir=args.vtk_hull_dir,
        tiledb_root=args.tiledb_root,
    )
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W6 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
