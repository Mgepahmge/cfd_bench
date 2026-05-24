from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class LiteMesh:
    cell_ids: NDArray[np.int32]
    node_xyz: Dict[int, Tuple[float, float, float]]
    cell_nodes: Dict[int, List[int]]
    cell_bbox: Dict[int, Tuple[float, float, float, float, float, float]]


@dataclass
class LitePolyData:
    points: NDArray[np.float64]
    triangles: NDArray[np.int32]
    meta: Dict[str, str] = field(default_factory=dict)
