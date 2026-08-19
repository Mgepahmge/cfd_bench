"""W10: per-frame statistics over available physical quantities."""

from __future__ import annotations

import argparse
import time

from cfd_bench.workloads.common.backends import make_iotdb, make_pg, make_tiledb
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig


def _bench(label, stats_fn, duration, step):
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        stats_fn(step=step)
        txn += 1
    print(f"{label} W10: {txn} txns in {duration}s")


def _run_client(label, client, ship, step, duration):
    try:
        # Frozen structural path keeps genuine H5 nodal-vs-cell semantics.
        stats_fn = client.frame_statistics if client.is_h5_dataset() else client.cfd_frame_statistics
        _bench(label, stats_fn, duration, step)
    finally:
        client.close()


def run_ship_step(cfg: WorkloadConfig, ship: str, step: int, backends: set):
    unsupported = set(backends) - {"postgresql", "iotdb", "tiledb"}
    if unsupported:
        raise RuntimeError(f"W10 H5 support is not implemented for: {sorted(unsupported)}")
    if "postgresql" in backends:
        _run_client(
            "PG", make_pg(ship, step, cfg.fluid_zone(ship)),
            ship, step, cfg.duration_sec,
        )
    if "iotdb" in backends:
        _run_client(
            "IoTDB", make_iotdb(ship, step, cfg.fluid_zone(ship)),
            ship, step, cfg.duration_sec,
        )
    if "tiledb" in backends:
        _run_client(
            "TileDB", make_tiledb(ship, step, cfg.tiledb_root, cfg.fluid_zone(ship)),
            ship, step, cfg.duration_sec,
        )


def main(ships=None):
    ap = argparse.ArgumentParser(description="W10: per-frame min/max/mean/stddev statistics")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        for step in cfg.valid_steps(ship):
            if cfg.skip_step(ship, step):
                continue
            print(f"\n=== W10 dataset={ship} frame={step} ===")
            run_ship_step(cfg, ship, step, set(args.backend))


if __name__ == "__main__":
    main()
