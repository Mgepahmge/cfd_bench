"""Shared workload configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from cfd_bench.core.paths import (
    resolve_max_range_dir,
    resolve_tiledb_root,
    resolve_vtk_dir,
    resolve_vtk_hull_dir,
)


DEFAULT_SHIPS = ["JBC_615k"]
DEFAULT_STEPS = [200, 400, 600]
VARIABLES = ["U", "V", "W", "P", "K", "E"]
SHIPS_WITHOUT_STEP_2000 = {"JBC_3843k", "Kvlcc2_3709k"}


@dataclass
class WorkloadConfig:
    ships: List[str] = field(default_factory=lambda: list(DEFAULT_SHIPS))
    duration_sec: float = 60.0
    vtk_dir: str = field(default_factory=resolve_vtk_dir)
    vtk_hull_dir: str = field(default_factory=resolve_vtk_hull_dir)
    tiledb_root: str = field(default_factory=resolve_tiledb_root)
    max_range_dir: str = field(default_factory=resolve_max_range_dir)
    steps: Optional[List[int]] = None
    variables: List[str] = field(default_factory=lambda: list(VARIABLES))
    geom_engine: str = "db"
    zone_fluid: str = "0_Fluid"
    zone_hull: str = "0_Wall_hull"

    def valid_steps(self, ship: str) -> List[int]:
        source = list(self.steps) if self.steps is not None else list(DEFAULT_STEPS)
        if ship in SHIPS_WITHOUT_STEP_2000:
            return [s for s in source if s != 2000]
        return source

    def skip_step(self, ship: str, step: int) -> bool:
        return step == 2000 and ship in SHIPS_WITHOUT_STEP_2000
