from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Iterator
from typing import Dict, List, Optional, Tuple

import numpy as np


class CellArrayView(Mapping[int, Tuple[float, ...]]):
    """Read-only Mapping facade over compact NumPy cell arrays.

    IoTDB/TileDB runtimes historically retained one Python tuple per cell and
    then duplicated the same data into NumPy geometry arrays.  This facade
    preserves the mapping API used by legacy mesh code while keeping the
    resident representation compact.  Tuples are materialized only for cells
    that are actually accessed through the mapping interface.
    """

    def __init__(self, ids, centroids, bbox_min, bbox_max, cell_types=None):
        self.ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        self.centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
        self.bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(-1, 3)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(-1, 3)
        if cell_types is None:
            self.cell_types = np.zeros(self.ids.size, dtype=np.int32)
        else:
            self.cell_types = np.asarray(cell_types, dtype=np.int32).reshape(-1)
        if not (
            self.ids.size == len(self.centroids) == len(self.bbox_min)
            == len(self.bbox_max) == self.cell_types.size
        ):
            raise ValueError("cell array lengths do not match")

    def __len__(self) -> int:
        return int(self.ids.size)

    def __iter__(self) -> Iterator[int]:
        return (int(x) for x in self.ids)

    def _pos(self, cid: int) -> int:
        pos = int(np.searchsorted(self.ids, int(cid)))
        if pos < 0 or pos >= self.ids.size or int(self.ids[pos]) != int(cid):
            raise KeyError(int(cid))
        return pos

    def __getitem__(self, cid: int) -> Tuple[float, ...]:
        pos = self._pos(int(cid))
        c = self.centroids[pos]
        mn = self.bbox_min[pos]
        mx = self.bbox_max[pos]
        return (
            float(c[0]), float(c[1]), float(c[2]),
            float(mn[0]), float(mx[0]),
            float(mn[1]), float(mx[1]),
            float(mn[2]), float(mx[2]),
            int(self.cell_types[pos]),
        )

    def get(self, cid: int, default=None):
        try:
            return self[int(cid)]
        except KeyError:
            return default



@dataclass
class RuntimeMeshData:
    cells: Mapping[int, Tuple[float, ...]] = field(default_factory=dict)
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
    # Precomputed AABB views used by hot geometry paths.  Keeping these as
    # contiguous NumPy arrays avoids recomputing global bounds / center-radius
    # representations on every transaction.
    all_bbox_center: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    all_bbox_extent: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    global_bbox_min: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    global_bbox_max: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    # Compact uniform-grid index.  Older versions stored a Python dict of
    # hundreds of thousands of bucket/list objects; the CSR-like arrays below
    # are much cheaper to build and retain on multi-million-cell meshes.
    spatial_bucket_keys: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    spatial_bucket_offsets: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    spatial_bucket_cell_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
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
