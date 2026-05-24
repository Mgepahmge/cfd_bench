from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RuntimeMeshData:
    cells: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    cell_nodes: Dict[int, List[int]] = field(default_factory=dict)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    face_planes: Dict[int, List[Tuple[int, float, float, float, float]]] = field(default_factory=dict)
    spatial_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spatial_step: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    spatial_dims: Tuple[int, int, int] = (1, 1, 1)
    spatial_buckets: Dict[Tuple[int, int, int], List[int]] = field(default_factory=dict)
    all_cell_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    all_bbox_min: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    all_bbox_max: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))

    @property
    def cell_bbox(self) -> Dict[int, Tuple[float, float, float, float, float, float]]:
        out: Dict[int, Tuple[float, float, float, float, float, float]] = {}
        for cid, v in self.cells.items():
            out[cid] = (v[3], v[4], v[5], v[6], v[7], v[8])
        return out

    @property
    def cell_centroid(self) -> Dict[int, Tuple[float, float, float]]:
        return {cid: (v[0], v[1], v[2]) for cid, v in self.cells.items()}
