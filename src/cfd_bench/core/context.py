from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, Tuple


@dataclass(frozen=True)
class DatasetKey:
    ship: str
    scale: str
    zone: str = "0_Fluid"
    step: int = 200

    @property
    def dataset_key(self) -> str:
        if self.scale:
            return f"{self.ship}_{self.scale}"
        return self.ship

    @property
    def pg_ship_type(self) -> str:
        return self.ship

    @property
    def pg_scale(self) -> str:
        return self.scale

    @property
    def pg_zone_type(self) -> str:
        z = self.zone.lower()
        if "hull" in z:
            return "hull"
        return "fluid"


def parse_dataset_key(text: str, zone: str = "0_Fluid", step: int = 200) -> DatasetKey:
    text = text.strip()
    if "_" in text:
        ship, scale = text.split("_", 1)
        return DatasetKey(ship=ship, scale=scale, zone=zone, step=int(step))
    return DatasetKey(ship=text, scale="", zone=zone, step=int(step))


@dataclass
class MeshContext:
    dataset_key: str
    step: int
    zone: str
    available_caps: Set[str] = field(default_factory=set)
    missing_caps: Set[str] = field(default_factory=set)
