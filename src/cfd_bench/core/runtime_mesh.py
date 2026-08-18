from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    all_centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    all_bbox_min: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    all_bbox_max: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    _cell_bbox_cache: Optional[Dict[int, Tuple[float, float, float, float, float, float]]] = field(
        default=None, init=False, repr=False
    )
    _cell_centroid_cache: Optional[Dict[int, Tuple[float, float, float]]] = field(
        default=None, init=False, repr=False
    )

    def invalidate_cell_views(self) -> None:
        """Invalidate derived Python views after replacing ``cells`` wholesale.

        Runtime meshes are effectively immutable after loading.  Keeping these
        views cached avoids rebuilding million-entry dictionaries on every
        geometry transaction while retaining compatibility with code that
        expects the historical ``cell_bbox``/``cell_centroid`` mappings.
        """
        self._cell_bbox_cache = None
        self._cell_centroid_cache = None

    @property
    def cell_bbox(self) -> Dict[int, Tuple[float, float, float, float, float, float]]:
        if self._cell_bbox_cache is None:
            self._cell_bbox_cache = {
                int(cid): (float(v[3]), float(v[4]), float(v[5]), float(v[6]), float(v[7]), float(v[8]))
                for cid, v in self.cells.items()
            }
        return self._cell_bbox_cache

    @property
    def cell_centroid(self) -> Dict[int, Tuple[float, float, float]]:
        if self._cell_centroid_cache is None:
            self._cell_centroid_cache = {
                int(cid): (float(v[0]), float(v[1]), float(v[2]))
                for cid, v in self.cells.items()
            }
        return self._cell_centroid_cache
