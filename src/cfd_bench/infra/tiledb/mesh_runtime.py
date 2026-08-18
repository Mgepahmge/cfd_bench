from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import CellArrayView, RuntimeMeshData
from cfd_bench.core.observability import timed_stage
from .repository import TileDBRepository


_SHARED_CACHE: "OrderedDict[Tuple[object, ...], RuntimeMeshData]" = OrderedDict()
_SHARED_CACHE_LIMIT = 2


def _repo_scope(repo: TileDBRepository) -> Optional[Tuple[object, ...]]:
    cfg = getattr(repo, "config", None)
    if cfg is None or not hasattr(cfg, "root_path"):
        return None
    return ("tiledb", os.path.abspath(str(cfg.root_path)))


class MeshRuntime:
    def __init__(self, repo: TileDBRepository):
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
            with timed_stage("TileDB mesh", f"load cells + build spatial index dataset={dataset_key} zone={zone}"):
                if hasattr(self.repo, "fetch_cells_arrays"):
                    ids, centers, mins, maxs, cell_types = self.repo.fetch_cells_arrays(dataset_key, zone)
                    data.cells = CellArrayView(ids, centers, mins, maxs, cell_types)
                    data.all_cell_ids = np.asarray(ids, dtype=np.int32)
                    data.all_centroids = np.asarray(centers, dtype=np.float64)
                    data.all_bbox_min = np.asarray(mins, dtype=np.float64)
                    data.all_bbox_max = np.asarray(maxs, dtype=np.float64)
                else:
                    data.cells = self.repo.fetch_cells(dataset_key, zone)
                data.invalidate_cell_views()
                self._build_spatial_index(data)
        return data

    def ensure_nodes(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self._get_or_init(dataset_key, zone)
        if not data.nodes:
            with timed_stage("TileDB mesh", f"load nodes dataset={dataset_key} zone={zone}"):
                data.nodes = self.repo.fetch_nodes(dataset_key, zone)
        return data

    def ensure_cell_nodes(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self.ensure_nodes(dataset_key, zone)
        if not data.cell_nodes:
            with timed_stage("TileDB mesh", f"load cell connectivity dataset={dataset_key} zone={zone}"):
                data.cell_nodes = self.repo.fetch_cell_nodes(dataset_key, zone)
        return data

    def ensure_adjacency(self, dataset_key: str, zone: str) -> RuntimeMeshData:
        data = self.ensure_cells(dataset_key, zone)
        if not data.adjacency:
            with timed_stage("TileDB mesh", f"load adjacency dataset={dataset_key} zone={zone}"):
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
        if (
            data.all_cell_ids.size
            and data.all_centroids.shape == (data.all_cell_ids.size, 3)
            and data.all_bbox_min.shape == (data.all_cell_ids.size, 3)
            and data.all_bbox_max.shape == (data.all_cell_ids.size, 3)
        ):
            cids = np.asarray(data.all_cell_ids, dtype=np.int32)
            centers = np.asarray(data.all_centroids, dtype=np.float64)
            mins = np.asarray(data.all_bbox_min, dtype=np.float64)
            maxs = np.asarray(data.all_bbox_max, dtype=np.float64)
        else:
            items = sorted(data.cells.items(), key=lambda x: x[0])
            cids = np.fromiter((int(cid) for cid, _ in items), dtype=np.int32, count=len(items))
            rows = np.asarray([v for _, v in items], dtype=np.float64)
            centers = np.ascontiguousarray(rows[:, :3], dtype=np.float64)
            mins = np.ascontiguousarray(rows[:, [3, 5, 7]], dtype=np.float64)
            maxs = np.ascontiguousarray(rows[:, [4, 6, 8]], dtype=np.float64)

        gmin = np.min(mins, axis=0)
        gmax = np.max(maxs, axis=0)
        span = np.maximum(gmax - gmin, 1e-12)
        target_dim = 64
        step = np.maximum(span / float(target_dim), 1e-12)
        idx = np.floor((centers - gmin) / step).astype(np.int32)
        idx = np.clip(idx, 0, target_dim - 1)

        # Build a compact CSR-like bucket index entirely with NumPy.  Cells are
        # assigned by centroid exactly as before; only the in-memory
        # representation changes.
        linear = (
            (idx[:, 0].astype(np.int64) * target_dim + idx[:, 1].astype(np.int64))
            * target_dim
            + idx[:, 2].astype(np.int64)
        )
        order = np.argsort(linear, kind="stable")
        sorted_keys = linear[order]
        sorted_cids = cids[order]
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        offsets = np.empty(unique_keys.size + 1, dtype=np.int64)
        offsets[:-1] = starts
        offsets[-1] = sorted_cids.size

        data.spatial_origin = tuple(float(x) for x in gmin)
        data.spatial_step = tuple(float(x) for x in step)
        data.spatial_dims = (target_dim, target_dim, target_dim)
        # Keep the legacy field empty rather than materializing millions of
        # Python integers/lists. geometry_ops transparently prefers compact
        # arrays and still supports legacy/ad-hoc RuntimeMeshData instances.
        data.spatial_buckets = {}
        data.spatial_bucket_keys = np.asarray(unique_keys, dtype=np.int64)
        data.spatial_bucket_offsets = offsets
        data.spatial_bucket_cell_ids = np.asarray(sorted_cids, dtype=np.int32)
        data.all_cell_ids = cids
        data.all_centroids = centers
        data.all_bbox_min = mins
        data.all_bbox_max = maxs
        data.all_bbox_center = np.ascontiguousarray(0.5 * (mins + maxs), dtype=np.float64)
        data.all_bbox_extent = np.ascontiguousarray(0.5 * (maxs - mins), dtype=np.float64)
        data.global_bbox_min = np.asarray(gmin, dtype=np.float64)
        data.global_bbox_max = np.asarray(gmax, dtype=np.float64)
