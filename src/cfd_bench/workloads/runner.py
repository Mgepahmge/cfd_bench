"""Orchestrate CFD-Bench workloads W1–W11."""

from __future__ import annotations

import importlib
import time
from typing import Sequence, Set

from cfd_bench.core.observability import stage
from cfd_bench.workloads.common.config import WorkloadConfig

DEFAULT_WORKLOADS = ("w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8")
H5_ONLY_WORKLOADS = ("w9", "w10", "w11")


def run_workload(workload_id: str, cfg: WorkloadConfig, backends: Set[str]) -> None:
    mod = importlib.import_module(f"cfd_bench.workloads.{workload_id}.run")
    if hasattr(mod, "run_ship"):
        for ship in cfg.ships:
            print(f"\n=== {workload_id.upper()} dataset={ship} ===")
            label = f"{workload_id.upper()} dataset={ship}"
            stage(label, "start")
            t0 = time.perf_counter()
            try:
                mod.run_ship(cfg, ship, backends)
            finally:
                stage(label, f"finished (wall={time.perf_counter() - t0:.3f}s)")
    elif hasattr(mod, "run_ship_step"):
        for ship in cfg.ships:
            for step in cfg.valid_steps(ship):
                if cfg.skip_step(ship, step):
                    continue
                print(f"\n=== {workload_id.upper()} dataset={ship} step={step} ===")
                label = f"{workload_id.upper()} dataset={ship} step={step}"
                stage(label, "start")
                t0 = time.perf_counter()
                try:
                    mod.run_ship_step(cfg, ship, step, backends)
                finally:
                    stage(label, f"finished (wall={time.perf_counter() - t0:.3f}s)")
    else:
        raise RuntimeError(f"{workload_id}: missing run_ship/run_ship_step")


def run_all(
    cfg: WorkloadConfig,
    backends: Set[str],
    workloads: Sequence[str] = DEFAULT_WORKLOADS,
) -> None:
    for workload_id in workloads:
        print(f"\n{'=' * 20} Running {workload_id} {'=' * 20}")
        run_workload(workload_id, cfg, backends)
