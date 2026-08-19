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
from cfd_bench.core.observability import benchmark_progress, stage


def _bench(label, norm_fn, pressure_fn, n_cells, duration, *, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    cells = np.arange(n_cells, dtype=np.int32)
    with benchmark_progress(f"{label} W6", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            prog.set_phase("surface normals")
            normals = norm_fn()
            prog.set_phase(f"scalar query ({n_cells} cells)")
            pressures = pressure_fn(cells)
            prog.set_phase("force reduction")
            calculate_force(normals, pressures)
            txn += 1
            prog.transaction()
    print(f"{label} W6: {txn} txns in {duration}s")


def _bench_pg_native(client, scalar_name: str, duration: float, *, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress("PG W6", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            prog.set_phase("surface cells + normals")
            cells, normals = client.surface_cells_and_normals()
            if len(cells) == 0 or len(normals) == 0:
                break
            prog.set_phase(f"scalar query {scalar_name} ({len(cells)} cells)")
            values = client.point_query(cells, scalar_name)
            n = min(len(normals), len(values))
            if n == 0:
                break
            prog.set_phase("force reduction")
            calculate_force(normals[:n], values[:n])
            txn += 1
            prog.transaction()
    print(f"PG W6: {txn} txns in {duration}s (zone={client.ctx.zone}, scalar={scalar_name})")


def _bench_iotdb_native(client, scalar_name: str, duration: float, *, progress=False, progress_interval=5.0):
    """IoTDB-native W6 with explicit surface-cell ids.

    Legacy CFD boundary faces are stored as face rows whose owning cell ids are
    measurements, so assuming ``0..N-1`` can query the wrong scalar rows.  H5
    meshes may not have a boundary-face device at all.  The client normalizes
    both cases into aligned ``(cell_ids, normals)`` here.
    """
    txn = 0
    t0 = time.time()
    with benchmark_progress("IoTDB W6", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            prog.set_phase("surface cells + normals")
            cells, normals = client.surface_cells_and_normals()
            if len(cells) == 0 or len(normals) == 0:
                break
            prog.set_phase(f"scalar query {scalar_name} ({len(cells)} cells)")
            values = client.point_query(cells, scalar_name)
            n = min(len(normals), len(values))
            if n == 0:
                break
            prog.set_phase("force reduction")
            calculate_force(normals[:n], values[:n])
            txn += 1
            prog.transaction()
    print(f"IoTDB W6: {txn} txns in {duration}s (zone={client.ctx.zone}, scalar={scalar_name})")


def _bench_tiledb_native(client, scalar_name: str, duration: float, *, progress=False, progress_interval=5.0):
    """TileDB-native W6 using explicit surface cell ids."""
    txn = 0
    t0 = time.time()
    with benchmark_progress("TileDB W6", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            prog.set_phase("surface cells + normals")
            cells, normals = client.surface_cells_and_normals()
            if len(cells) == 0 or len(normals) == 0:
                break
            prog.set_phase(f"scalar query {scalar_name} ({len(cells)} cells)")
            values = client.point_query(cells, scalar_name)
            n = min(len(normals), len(values))
            if n == 0:
                break
            prog.set_phase("force reduction")
            calculate_force(normals[:n], values[:n])
            txn += 1
            prog.transaction()
    print(f"TileDB W6: {txn} txns in {duration}s (zone={client.ctx.zone}, scalar={scalar_name})")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    if "postgresql" in backends:
        # Keep the already-validated H5/structural path byte-for-byte in
        # behavior; only legacy CFD gets the generic zone/scalar fallback.
        probe = make_pg(ship, step, zone=cfg.fluid_zone(ship))
        h5_dataset = probe.is_h5_dataset()
        if h5_dataset:
            probe.close()
            pg = make_pg(ship, step, zone=cfg.zone_hull)
            geom = make_geom_client(cfg, ship, step, pg, cfg.zone_hull)
            n_cells = cell_count(geom)
            _bench("PG", lambda: geom.surface_norm(), lambda c: pg.point_query(c, "P"), n_cells, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            pg.close()
        else:
            try:
                zones = probe.w6_zone_candidates(
                    ship, preferred_zone=cfg.fluid_zone(ship), hull_hint=cfg.zone_hull
                )
            finally:
                probe.close()
            selected = None
            last_error = None
            candidates = ["P"] + list(cfg.valid_variables(ship))
            for zone in zones:
                stage("PostgreSQL W6", f"probe zone={zone}")
                client = make_pg(ship, step, zone=zone)
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
                    f"PostgreSQL W6 could not find a usable zone/scalar for dataset={ship} "
                    f"step={step}: {last_error}"
                )
            pg, scalar_name = selected
            try:
                _bench_pg_native(pg, scalar_name, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            finally:
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
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
            iotdb.close()
        else:
            # Frozen H5 behavior keeps the historical two-zone probe. Canonical
            # CFD metadata may expose differently named hull/wall zones, so
            # legacy CFD uses the full discovered zone list.
            probe = make_iotdb(ship, step, zone=cfg.fluid_zone(ship))
            try:
                if probe.is_h5_dataset():
                    zones = []
                    for zone in (cfg.zone_hull, cfg.fluid_zone(ship)):
                        if zone and zone not in zones:
                            zones.append(zone)
                else:
                    zones = probe.w6_zone_candidates(
                        ship, preferred_zone=cfg.fluid_zone(ship), hull_hint=cfg.zone_hull
                    )
            finally:
                probe.close()
            selected = None
            last_error = None
            candidates = ["P"] + list(cfg.valid_variables(ship))
            for zone in zones:
                stage("IoTDB W6", f"probe zone={zone}")
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
                _bench_iotdb_native(iotdb, scalar_name, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
            finally:
                iotdb.close()

    if "tiledb" in backends:
        # Use the same generic W6 strategy for CFD and H5 data: prefer a real
        # hull/wall zone and pressure, but fall back to any usable mesh zone
        # and cell scalar rather than failing when only 0_Fluid exists.
        probe = make_tiledb(
            ship, step, cfg.tiledb_root, zone=cfg.fluid_zone(ship)
        )
        try:
            zones = probe.w6_zone_candidates(
                ship, preferred_zone=cfg.fluid_zone(ship), hull_hint=cfg.zone_hull
            )
        finally:
            probe.close()

        selected = None
        last_error = None
        candidates = ["P"] + list(cfg.valid_variables(ship))
        for zone in zones:
            stage("TileDB W6", f"probe zone={zone}")
            client = make_tiledb(ship, step, cfg.tiledb_root, zone=zone)
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
                f"TileDB W6 could not find a usable zone/scalar for dataset={ship} "
                f"step={step}: {last_error}"
            )

        tiledb, scalar_name = selected
        try:
            if uses_vtk_geom(cfg):
                # Preserve the historical optional VTK-geometry mode while
                # using the same robust TileDB zone/scalar selection.
                geom = make_geom_client(cfg, ship, step, tiledb, tiledb.ctx.zone)
                try:
                    n_cells = cell_count(geom)
                    sub = geom.extract_submesh(list(range(n_cells)))
                    _bench(
                        "TileDB",
                        lambda: geom.surface_norm(sub),
                        lambda c: tiledb.point_query(c, scalar_name),
                        n_cells,
                        cfg.duration_sec,
                        progress=cfg.progress,
                        progress_interval=cfg.progress_interval_sec,
                    )
                finally:
                    if geom is not tiledb:
                        geom.close()
            else:
                _bench_tiledb_native(tiledb, scalar_name, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
        finally:
            tiledb.close()

    if "vtk" in backends:
        hull = make_vtk(cfg.vtk_hull_dir, ship, step, zone=cfg.zone_hull)
        n_cells = cell_count(hull)
        _bench("VTK", lambda: hull.surface_norm(), lambda c: hull.point_query(c, "P"), n_cells, cfg.duration_sec, progress=cfg.progress, progress_interval=cfg.progress_interval_sec)
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
