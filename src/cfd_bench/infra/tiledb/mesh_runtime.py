from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from .repository import TileDBRepository


class MeshRuntime:
    def __init__(self, repo: TileDBRepository):
        self.repo = repo
        self._cache: Dict[Tuple[str, str], RuntimeMeshData] = {}

    def _get_or_init(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        key = (dataset_key, zone)
        data = self._cache.get(key)
        if data is None:
            data = RuntimeMeshData()
            self._cache[key] = data
        return data

    def ensure_cells(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self._get_or_init(dataset_key, zone)
        if not data.cells:
            data.cells = self.repo.fetch_cells(dataset_key, zone)
            self._build_spatial_index(data)
        return data

    def ensure_nodes(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self._get_or_init(dataset_key, zone)
        if not data.nodes:
            data.nodes = self.repo.fetch_nodes(dataset_key, zone)
        return data

    def ensure_cell_nodes(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self.ensure_nodes(dataset_key, zone)
        if not data.cell_nodes:
            data.cell_nodes = self.repo.fetch_cell_nodes(dataset_key, zone)
        return data

    def ensure_adjacency(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self.ensure_cells(dataset_key, zone)
        if not data.adjacency:
            data.adjacency = self.repo.fetch_cell_adjacency(dataset_key, zone)
        return data

    def ensure_face_planes(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self.ensure_cells(dataset_key, zone)
        if not data.face_planes:
            data.face_planes = self.repo.fetch_face_planes(dataset_key, zone)
        return data

    def load(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self.ensure_face_planes(dataset_key, zone)
        data = self.ensure_adjacency(dataset_key, zone)
        data = self.ensure_cell_nodes(dataset_key, zone)
        return data

    def clear(self):
        self._cache.clear()

    @staticmethod
    def _build_spatial_index(data: RuntimeMeshData):
        if not data.cells:
            return
        items = sorted(data.cells.items(), key=lambda x: x[0])
        cids = np.array([int(cid) for cid, _ in items], dtype=np.int32)
        mins = np.array([[float(v[3]), float(v[5]), float(v[7])] for _, v in items], dtype=np.float64)
        maxs = np.array([[float(v[4]), float(v[6]), float(v[8])] for _, v in items], dtype=np.float64)
        centers = 0.5 * (mins + maxs)

        gmin = np.min(mins, axis=0)
        gmax = np.max(maxs, axis=0)
        span = np.maximum(gmax - gmin, 1e-12)
        target_dim = 64
        step = span / float(target_dim)
        step = np.maximum(step, 1e-12)

        ix = np.floor((centers[:, 0] - gmin[0]) / step[0]).astype(np.int32)
        iy = np.floor((centers[:, 1] - gmin[1]) / step[1]).astype(np.int32)
        iz = np.floor((centers[:, 2] - gmin[2]) / step[2]).astype(np.int32)
        ix = np.clip(ix, 0, target_dim - 1)
        iy = np.clip(iy, 0, target_dim - 1)
        iz = np.clip(iz, 0, target_dim - 1)

        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for i in range(len(cids)):
            key = (int(ix[i]), int(iy[i]), int(iz[i]))
            buckets.setdefault(key, []).append(int(cids[i]))

        data.spatial_origin = (float(gmin[0]), float(gmin[1]), float(gmin[2]))
        data.spatial_step = (float(step[0]), float(step[1]), float(step[2]))
        data.spatial_dims = (target_dim, target_dim, target_dim)
        data.spatial_buckets = buckets
        data.all_cell_ids = cids
        data.all_bbox_min = mins
        data.all_bbox_max = maxs
