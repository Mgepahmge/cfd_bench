"""Shared workload configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    # Explicit overrides.  None means "use discovered metadata if available".
    steps: Optional[List[int]] = None
    variables: Optional[List[str]] = None
    geom_engine: str = "db"
    zone_fluid: Optional[str] = None
    zone_hull: str = "0_Wall_hull"
    discovered_steps: Dict[str, List[int]] = field(default_factory=dict)
    discovered_variables: Dict[str, List[str]] = field(default_factory=dict)
    discovered_zones: Dict[str, str] = field(default_factory=dict)

    def valid_steps(self, ship: str) -> List[int]:
        if self.steps is not None:
            source = list(self.steps)
        elif ship in self.discovered_steps:
            source = list(self.discovered_steps[ship])
        else:
            source = list(DEFAULT_STEPS)
        if ship in SHIPS_WITHOUT_STEP_2000:
            return [s for s in source if s != 2000]
        return source

    def valid_variables(self, ship: str) -> List[str]:
        if self.variables is not None:
            return [str(v).upper() for v in self.variables]
        if ship in self.discovered_variables and self.discovered_variables[ship]:
            return [str(v).upper() for v in self.discovered_variables[ship]]
        return list(VARIABLES)

    def fluid_zone(self, ship: str) -> str:
        if self.zone_fluid:
            return self.zone_fluid
        return self.discovered_zones.get(ship, "0_Fluid")

    def skip_step(self, ship: str, step: int) -> bool:
        return step == 2000 and ship in SHIPS_WITHOUT_STEP_2000
