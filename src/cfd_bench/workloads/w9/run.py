"""W9: H5 element IDs by coordinate range (PostgreSQL only for now)."""

from __future__ import annotations

import argparse
import time

from cfd_bench.workloads.common.backends import make_pg
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.common.geom_resolver import mesh_bounds
from cfd_bench.workloads.common.random_geom import random_coord_range


def _require_h5(pg, dataset: str) -> None:
    if not pg.is_h5_dataset():
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
    if "postgresql" not in backends:
        raise RuntimeError("W9 currently supports PostgreSQL only")
    steps = cfg.valid_steps(ship)
    if not steps:
        raise RuntimeError(f"W9 dataset has no discovered frames: {ship}")
    step = steps[0]
    pg = make_pg(ship, step, cfg.fluid_zone(ship))
    try:
        _require_h5(pg, ship)
        bounds = mesh_bounds(cfg, ship, step, pg, pg, cfg.fluid_zone(ship))
        if bounds is None:
            raise RuntimeError(f"W9 cannot determine mesh bounds for {ship}")
        _bench("PG", pg.h5_element_ids_in_coordinate_range, bounds, cfg.duration_sec)
    finally:
        pg.close()


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
