from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from cfd_bench.infra.vtk.storage import read_manifest


@dataclass(frozen=True)
class VTKDatasetInfo:
    dataset_key: str
    zone_type: str
    timesteps: Tuple[int, ...]
    variables: Tuple[str, ...]


def discover_vtk_datasets(selected_datasets: Iterable[str], *, root: str) -> List[VTKDatasetInfo]:
    out: List[VTKDatasetInfo] = []
    for raw in selected_datasets:
        key = str(raw)
        try:
            meta = read_manifest(root, key)
            zone = str(meta.get("primary_zone") or "0_Fluid")
            zmeta = (meta.get("zones") or {}).get(zone, {})
            steps = tuple(int(x) for x in zmeta.get("steps", meta.get("steps", ())))
            variables = tuple(str(v).upper() for v in zmeta.get("variables", meta.get("variables", ())))
            out.append(VTKDatasetInfo(key, zone, steps, variables))
        except Exception:
            continue
    return out
