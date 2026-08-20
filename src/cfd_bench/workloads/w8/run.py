"""W8: Variable range query (vortex / threshold cell selection)."""

from __future__ import annotations

import argparse
import random
import time

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb, make_vtk
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import VARIABLES, WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import make_geom_client, random_var_range_db, uses_vtk_geom
from cfd_bench.core.observability import benchmark_progress
from cfd_bench.core.results import emit_benchmark_result


def random_var_range_vtk(vtk_mesh, attribute_name: str):
    arr = vtk_mesh.GetCellData().GetArray(attribute_name)
    if arr is None:
        raise ValueError(f"attribute {attribute_name} not found")
    vmin, vmax = arr.GetRange()
    lo, hi = sorted([random.uniform(vmin, vmax), random.uniform(vmin, vmax)])
    return lo, hi


def _bench(label, range_fn, client, geom_client, duration, step, variables, *, progress=False, progress_interval=5.0):
    txn = 0
    t0 = time.time()
    with benchmark_progress(f"{label} W8", duration, enabled=progress, interval=progress_interval) as prog:
        while time.time() - t0 < duration:
            var = random.choice(variables)
            prog.set_phase(f"sample variable range {var}")
            if geom_client is not None and hasattr(geom_client, "vtk_mesh") and geom_client.vtk_mesh is not None:
                lo, hi = random_var_range_vtk(geom_client.vtk_mesh, var)
            else:
                lo, hi = random_var_range_db(client, var, step=step)
            prog.set_phase(f"variable range query {var}")
            range_fn(lo, hi, var)
            txn += 1
            prog.transaction()
    emit_benchmark_result(
        f"{label} W8: {txn} txns in {duration}s",
        backend=label, operation="variable_range", transactions=txn,
        duration_sec=duration,
    )


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, pg, cfg.fluid_zone(ship))
        _bench(
            "PG",
            lambda lo, hi, v: pg.range_query_var(lo, hi, v),
            pg,
            geom if uses_vtk_geom(cfg) else None,
            cfg.duration_sec,
            step,
            cfg.valid_variables(ship),
            progress=cfg.progress,
            progress_interval=cfg.progress_interval_sec,
        )
        pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, iotdb, cfg.fluid_zone(ship))
        _bench(
            "IoTDB",
            lambda lo, hi, v: iotdb.range_query_var(lo, hi, v),
            iotdb,
            geom if uses_vtk_geom(cfg) else None,
            cfg.duration_sec,
            step,
            cfg.valid_variables(ship),
            progress=cfg.progress,
            progress_interval=cfg.progress_interval_sec,
        )
        iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship))
        geom = make_geom_client(cfg, ship, step, tiledb, cfg.fluid_zone(ship))
        _bench(
            "TileDB",
            lambda lo, hi, v: tiledb.range_query_var(lo, hi, v),
            tiledb,
            geom if uses_vtk_geom(cfg) else None,
            cfg.duration_sec,
            step,
            cfg.valid_variables(ship),
            progress=cfg.progress,
            progress_interval=cfg.progress_interval_sec,
        )
        tiledb.close()

    if "vtk" in backends:
        vtk = make_vtk(cfg.vtk_dir, ship, step, cfg.fluid_zone(ship))
        _bench(
            "VTK",
            lambda lo, hi, v: vtk.range_query_var(lo, hi, v),
            vtk,
            vtk,
            cfg.duration_sec,
            step,
            cfg.valid_variables(ship),
            progress=cfg.progress,
            progress_interval=cfg.progress_interval_sec,
        )
        vtk.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W8: variable range query")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W8 ship={ship} step={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
