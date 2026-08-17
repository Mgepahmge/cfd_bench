"""W9: H5 element IDs by coordinate range."""

from __future__ import annotations

import argparse
import time

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import mesh_bounds
from cfd_bench.workloads.common.random_geom import random_coord_range


def _require_h5(client, dataset: str) -> None:
    if not client.is_h5_dataset():
        raise RuntimeError(f"W9 currently supports H5-ingested datasets only: {dataset}")


def _bench(label, range_fn, bounds, duration):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        lo, hi = random_coord_range(bounds)
        range_fn(lo, hi)
        txn += 1
    print(f"{label} W9: {txn} txns in {duration}s")


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    unsupported = set(backends) - {"postgresql", "iotdb", "tiledb"}
    if unsupported:
        raise RuntimeError(f"W9 H5 support is not implemented for: {sorted(unsupported)}")
    steps = cfg.valid_steps(ship)
    if not steps:
        raise RuntimeError(f"W9 dataset has no discovered frames: {ship}")
    step = steps[0]

    if "postgresql" in backends:
        pg = make_pg(ship, step, cfg.fluid_zone(ship))
        try:
            _require_h5(pg, ship)
            bounds = mesh_bounds(cfg, ship, step, pg, pg, cfg.fluid_zone(ship))
            if bounds is None:
                raise RuntimeError(f"W9 cannot determine mesh bounds for {ship}")
            _bench("PG", pg.h5_element_ids_in_coordinate_range, bounds, cfg.duration_sec)
        finally:
            pg.close()

    if "iotdb" in backends:
        iotdb = make_iotdb(ship, step, cfg.fluid_zone(ship))
        try:
            _require_h5(iotdb, ship)
            bounds = mesh_bounds(cfg, ship, step, iotdb, iotdb, cfg.fluid_zone(ship))
            if bounds is None:
                raise RuntimeError(f"W9 cannot determine IoTDB mesh bounds for {ship}")
            _bench("IoTDB", iotdb.h5_element_ids_in_coordinate_range, bounds, cfg.duration_sec)
        finally:
            iotdb.close()

    if "tiledb" in backends:
        tiledb = make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship))
        try:
            _require_h5(tiledb, ship)
            bounds = mesh_bounds(cfg, ship, step, tiledb, tiledb, cfg.fluid_zone(ship))
            if bounds is None:
                raise RuntimeError(f"W9 cannot determine TileDB mesh bounds for {ship}")
            _bench("TileDB", tiledb.h5_element_ids_in_coordinate_range, bounds, cfg.duration_sec)
        finally:
            tiledb.close()


def main(ships=None):
    ap = argparse.ArgumentParser(description="W9: coordinate range -> H5 element IDs")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        print(f"\n=== W9 dataset={ship} ===")
        run_ship(cfg, ship, set(args.backend))


if __name__ == "__main__":
    main()
