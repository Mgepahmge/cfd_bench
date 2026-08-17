"""W6: Hull surface pressure integration (normals + scalar query)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import cell_count, make_geom_client, uses_vtk_geom
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


def _bench_iotdb_native(client, scalar_name: str, duration: float):
    """IoTDB-native W6 with explicit surface-cell ids.

    Legacy CFD boundary faces are stored as face rows whose owning cell ids are
    measurements, so assuming ``0..N-1`` can query the wrong scalar rows.  H5
    meshes may not have a boundary-face device at all.  The client normalizes
    both cases into aligned ``(cell_ids, normals)`` here.
    """
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        cells, normals = client.surface_cells_and_normals()
        if len(cells) == 0 or len(normals) == 0:
            break
        values = client.point_query(cells, scalar_name)
        n = min(len(normals), len(values))
        if n == 0:
            break
        calculate_force(normals[:n], values[:n])
        txn += 1
    print(f"IoTDB W6: {txn} txns in {duration}s (zone={client.ctx.zone}, scalar={scalar_name})")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    if "postgresql" in backends:
        pg = make_pg(ship, step, zone=cfg.zone_hull)
        geom = make_geom_client(cfg, ship, step, pg, cfg.zone_hull)
        n_cells = cell_count(geom)
        _bench("PG", lambda: geom.surface_norm(), lambda c: pg.point_query(c, "P"), n_cells, cfg.duration_sec)
        pg.close()

    if "iotdb" in backends:
        if uses_vtk_geom(cfg):
            # Preserve the historical VTK-geometry mode unchanged.
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
        else:
            # Prefer the physical hull zone for CFD.  Structural H5 datasets
            # commonly have only the main mesh zone, so fall back to that
            # without requiring a special ingest layout.
            zones = []
            for zone in (cfg.zone_hull, cfg.fluid_zone(ship)):
                if zone and zone not in zones:
                    zones.append(zone)
            selected = None
            last_error = None
            candidates = ["P"] + list(cfg.valid_variables(ship))
            for zone in zones:
                client = make_iotdb(ship, step, zone=zone)
                try:
                    cells, normals = client.surface_cells_and_normals()
                    if len(cells) == 0 or len(normals) == 0:
                        raise RuntimeError(f"no usable mesh cells in zone={zone}")
                    scalar_name = client.resolve_w6_scalar(candidates)
                    selected = (client, scalar_name)
                    break
                except Exception as exc:
                    last_error = exc
                    client.close()
            if selected is None:
                raise RuntimeError(
                    f"IoTDB W6 could not find a usable zone/scalar for dataset={ship} step={step}: {last_error}"
                )
            iotdb, scalar_name = selected
            try:
                _bench_iotdb_native(iotdb, scalar_name, cfg.duration_sec)
            finally:
                iotdb.close()

    if "tiledb" in backends:
        # create temporary client
        tiledb = make_tiledb(
            ship,
            step,
            cfg.tiledb_root,
            zone=cfg.fluid_zone(ship),
        )

        # automatically detect hull zone
        hull_zone = tiledb.resolve_hull_zone(
            ship
        )

        # reconnect using hull zone
        tiledb.close()

        tiledb = make_tiledb(
            ship,
            step,
            cfg.tiledb_root,
            zone=hull_zone,
        )

        geom = make_geom_client(
            cfg,
            ship,
            step,
            tiledb,
            hull_zone,
        )

        n_cells = cell_count(
            geom
        )

        sub = geom.extract_submesh(
            list(range(n_cells))
        )

        _bench(
            "TileDB",
            lambda: geom.surface_norm(sub),
            lambda c: tiledb.point_query(
                c,
                "P"
            ),
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
