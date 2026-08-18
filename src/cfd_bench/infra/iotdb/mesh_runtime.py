from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from .repository import IoTDBRepository


# Static topology is immutable during a benchmark run and identical for every
# timestep/workload.  Keep a tiny process-local LRU so W1-W8 do not repeatedly
# download and rebuild the same multi-million-cell mesh when each workload
# creates a fresh client.
_SHARED_CACHE: "OrderedDict[Tuple[object, ...], RuntimeMeshData]" = OrderedDict()
_SHARED_CACHE_LIMIT = 2


def _repo_scope(repo: IoTDBRepository) -> Optional[Tuple[object, ...]]:
    cfg = getattr(repo, "config", None)
    if cfg is None or not all(hasattr(cfg, name) for name in ("host", "port", "root_path")):
        return None
    return ("iotdb", str(cfg.host), str(cfg.port), str(cfg.root_path))


class MeshRuntime:
    def __init__(self, repo: IoTDBRepository):
        self.repo = repo
        self._cache: Dict[Tuple[str, str], RuntimeMeshData] = {}
        self._scope = _repo_scope(repo)

    def _get_or_init(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        key = (dataset_key, zone)
        data = self._cache.get(key)
        if data is not None:
            return data
        shared_key = None if self._scope is None else self._scope + key
        if shared_key is not None and shared_key in _SHARED_CACHE:
            data = _SHARED_CACHE.pop(shared_key)
            _SHARED_CACHE[shared_key] = data
        else:
            data = RuntimeMeshData()
            if shared_key is not None:
                _SHARED_CACHE[shared_key] = data
                while len(_SHARED_CACHE) > _SHARED_CACHE_LIMIT:
                    _SHARED_CACHE.popitem(last=False)
        self._cache[key] = data
        return data

    def ensure_cells(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self._get_or_init(dataset_key, zone)
        if not data.cells:
            data.cells = self.repo.fetch_cells(dataset_key, zone)
            data.invalidate_cell_views()
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
        data = self.ensure_adjacency(dataset_key, zone)
        data = self.ensure_cell_nodes(dataset_key, zone)
        return data

    def clear(self):
        # Release this client's references only.  The small process-local LRU
        # intentionally survives client.close() so the next workload can reuse
        # the immutable static topology.
        self._cache.clear()

    @staticmethod
    def _build_spatial_index(data: RuntimeMeshData):
        if not data.cells:
            return
        items = sorted(data.cells.items(), key=lambda x: x[0])
        cids = np.fromiter((int(cid) for cid, _ in items), dtype=np.int32, count=len(items))
        rows = np.asarray([v for _, v in items], dtype=np.float64)
        centers = rows[:, :3]
        mins = rows[:, [3, 5, 7]]
        maxs = rows[:, [4, 6, 8]]

        gmin = np.min(mins, axis=0)
        gmax = np.max(maxs, axis=0)
        span = np.maximum(gmax - gmin, 1e-12)
        target_dim = 64
        step = np.maximum(span / float(target_dim), 1e-12)

        idx = np.floor((centers - gmin) / step).astype(np.int32)
        idx = np.clip(idx, 0, target_dim - 1)

        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for cid, ijk in zip(cids.tolist(), idx.tolist()):
            key = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
            buckets.setdefault(key, []).append(int(cid))

        data.spatial_origin = tuple(float(x) for x in gmin)
        data.spatial_step = tuple(float(x) for x in step)
        data.spatial_dims = (target_dim, target_dim, target_dim)
        data.spatial_buckets = buckets
        data.all_cell_ids = cids
        data.all_centroids = np.ascontiguousarray(centers, dtype=np.float64)
        data.all_bbox_min = np.ascontiguousarray(mins, dtype=np.float64)
        data.all_bbox_max = np.ascontiguousarray(maxs, dtype=np.float64)
