"""W11: H5 point extrema across all frames."""

from __future__ import annotations

import argparse
import random
import time

from cfd_bench.workloads.common.backends import make_iotdb, make_pg
from cfd_bench.workloads.common.cli import add_common_workload_args, workload_config_from_args
from cfd_bench.workloads.common.config import WorkloadConfig


POINT_BATCH_SIZE = 32


def _select_variables(client, cfg: WorkloadConfig, dataset: str):
    available = list(client.h5_nodal_variables())
    if cfg.variables is None:
        return available
    requested = [str(v).upper() for v in cfg.variables]
    selected = [v for v in requested if v in available]
    if not selected:
        raise RuntimeError(
            f"W11 requested variables {requested} have no direct nodal H5 values; "
            f"available={available}"
        )
    return selected


def _bench(label, extrema_fn, point_ids, variables, duration, batch_size=POINT_BATCH_SIZE):
    if not point_ids:
        raise RuntimeError("W11 has no H5 point IDs to query")
    if not variables:
        raise RuntimeError("W11 has no nodal variable present in every frame")
    batch_size = min(int(batch_size), len(point_ids))
    txn = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        points = random.sample(point_ids, batch_size)
        var = random.choice(variables)
        extrema_fn(points, var)
        txn += 1
    print(
        f"{label} W11: {txn} txns in {duration}s "
        f"(batch={batch_size}, nodal_vars={variables})"
    )


def _run_client(label, client, cfg: WorkloadConfig, ship: str):
    try:
        if not client.is_h5_dataset():
            raise RuntimeError(f"W11 currently supports H5-ingested datasets only: {ship}")
        point_ids = [int(x) for x in client.h5_point_ids().tolist()]
        variables = _select_variables(client, cfg, ship)
        _bench(
            label,
            client.h5_point_frame_extrema,
            point_ids,
            variables,
            cfg.duration_sec,
        )
    finally:
        client.close()


def run_ship(cfg: WorkloadConfig, ship: str, backends: set):
    unsupported = set(backends) - {"postgresql", "iotdb"}
    if unsupported:
        raise RuntimeError(f"W11 H5 support is not implemented for: {sorted(unsupported)}")
    steps = cfg.valid_steps(ship)
    if not steps:
        raise RuntimeError(f"W11 dataset has no discovered frames: {ship}")
    if "postgresql" in backends:
        _run_client("PG", make_pg(ship, steps[0], cfg.fluid_zone(ship)), cfg, ship)
    if "iotdb" in backends:
        _run_client("IoTDB", make_iotdb(ship, steps[0], cfg.fluid_zone(ship)), cfg, ship)


def main(ships=None):
    ap = argparse.ArgumentParser(description="W11: point IDs -> min/max across all H5 frames")
    add_common_workload_args(ap)
    args = ap.parse_args()
    cfg = workload_config_from_args(args, ships=ships)
    for ship in cfg.ships:
        print(f"\n=== W11 dataset={ship} ===")
        run_ship(cfg, ship, set(args.backend))


if __name__ == "__main__":
    main()
