"""W3: Variable-range submesh + isosurface extraction."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import cell_count, make_geom_client
from cfd_bench.core.observability import benchmark_progress


def read_max_diffs(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                out[str(row[0]).upper()] = float(row[1])
    return out


def _find_max_diff_file(directory: str, ship: str, step: int):
    """Backward-compatible sidecar lookup for non-PostgreSQL backends."""
    if not directory or not os.path.isdir(directory):
        return None
    suffix = f"_{step}_max_diffs.csv"
    exact = os.path.join(directory, f"{ship}{suffix}")
    if os.path.isfile(exact):
        return exact
    for name in os.listdir(directory):
        if ship in name and name.endswith(suffix):
            return os.path.join(directory, name)
    return None


def _bench(label, scalar_fn, range_fn, extract_fn, iso_fn, n_cells, max_diffs, duration, variables, *, progress=False, progress_interval=5.0):
    usable = [str(v).upper() for v in variables if str(v).upper() in max_diffs]
    if not usable:
        print(f"{label} W3: skip (no variables with max-diff metadata)")
        return
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W3", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            var = random.choice(usable)
            delta = max_diffs[var]
            cid = random.randint(0, max(0, n_cells - 1))
            prog.set_phase(f"seed scalar query {var}")
            iso_val = float(scalar_fn([cid], var)[0])
            prog.set_phase(f"variable range query {var}")
            cells = range_fn(iso_val - delta, iso_val + delta, var)
            prog.set_phase(f"extract submesh ({len(cells)} cells)")
            sub = extract_fn(cells)
            prog.set_phase("isosurface extraction")
            iso_fn(sub, var, iso_val)
            txn += 1
            prog.transaction()
    print(f"{label} W3: {txn} txns in {duration}s")


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    variables = cfg.valid_variables(ship)

    # PostgreSQL is self-contained: max-diff metadata is materialized during
    # ingest and can also be recomputed from DB rows for older databases.
    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.fluid_zone(ship))
        try:
            max_diffs = pg.get_max_diffs(step)
            geom = make_geom_client(cfg, ship, step, pg, cfg.fluid_zone(ship))
            n_cells = cell_count(geom)
            _bench(
                "PG",
                lambda c, v: pg.point_query(c, v),
                lambda lo, hi, v: pg.range_query_var(lo, hi, v),
                lambda cells: geom.extract_submesh(cells),
                lambda mesh, v, val: geom.isosurface_from_submesh(mesh, v, val) if mesh and hasattr(geom, "isosurface_from_submesh") else (geom.isosurface_extraction(v, val) if mesh else None),
                n_cells,
                max_diffs,
                cfg.duration_sec,
                variables,
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
        finally:
            pg.close()

    # H5 IoTDB ingest materializes max-diff metadata in IoTDB, matching the
    # self-contained PostgreSQL path. Legacy CFD IoTDB can still use sidecars.
    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.fluid_zone(ship))
        try:
            max_diffs_iot = iotdb.get_max_diffs(step)
            if not max_diffs_iot:
                # Preserve legacy CFD IoTDB behavior: old ingests have no
                # materialized max_diff device and continue to use sidecars.
                delta_file = _find_max_diff_file(cfg.max_range_dir, ship, step)
                if delta_file is not None:
                    max_diffs_iot = read_max_diffs(delta_file)
            geom = make_geom_client(cfg, ship, step, iotdb, cfg.fluid_zone(ship))
            n_cells = cell_count(geom)
            _bench(
                "IoTDB",
                lambda c, v: iotdb.point_query(c, v),
                lambda lo, hi, v: iotdb.range_query_var(lo, hi, v),
                lambda cells: geom.extract_submesh(cells),
                lambda mesh, v, val: geom.isosurface_from_submesh(mesh, v, val) if hasattr(geom, "isosurface_from_submesh") else geom.isosurface_extraction(v, val),
                n_cells,
                max_diffs_iot,
                cfg.duration_sec,
                variables,
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
        finally:
            iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship))
        try:
            max_diffs_tiledb = tiledb.get_max_diffs(step)
            if not max_diffs_tiledb:
                # Preserve legacy CFD TileDB behavior: old ingests still use
                # the historical Max_Range sidecar.
                delta_file = _find_max_diff_file(cfg.max_range_dir, ship, step)
                if delta_file is not None:
                    max_diffs_tiledb = read_max_diffs(delta_file)
            geom = make_geom_client(cfg, ship, step, tiledb, cfg.fluid_zone(ship))
            n_cells = cell_count(geom)
            _bench(
                "TileDB",
                lambda c, v: tiledb.point_query(c, v),
                lambda lo, hi, v: tiledb.range_query_var(lo, hi, v),
                lambda cells: geom.extract_submesh(cells),
                lambda mesh, v, val: geom.isosurface_from_submesh(mesh, v, val) if hasattr(geom, "isosurface_from_submesh") else geom.isosurface_extraction(v, val),
                n_cells,
                max_diffs_tiledb,
                cfg.duration_sec,
                variables,
                progress=cfg.progress,
                progress_interval=cfg.progress_interval_sec,
            )
        finally:
            tiledb.close()

    max_diffs = None
    if "vtk" in backends:
        delta_file = _find_max_diff_file(cfg.max_range_dir, ship, step)
        if delta_file is None:
            print(f"W3: no sidecar max_diffs for {ship} step {step}; skip vtk")
        else:
            max_diffs = read_max_diffs(delta_file)

    if "vtk" in backends and max_diffs is not None:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.fluid_zone(ship))
        n_cells = cell_count(vtk)
        _bench(
            "VTK",
            lambda c, v: vtk.point_query(c, v),
            lambda lo, hi, v: vtk.range_query_var(lo, hi, v),
            lambda cells: vtk.extract_submesh(cells),
            lambda mesh, v, val: vtk.isosurface_extraction(v, val),
            n_cells,
            max_diffs,
            cfg.duration_sec,
            variables,
            progress=cfg.progress,
            progress_interval=cfg.progress_interval_sec,
        )
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W3: isosurface from variable range")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W3 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
