"""In-memory mesh assembly from PostgreSQL tables."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.core.observability import timed_stage
from cfd_bench.core.types import LiteMesh, LitePolyData
from cfd_bench.mesh_ops import iotdb_extract_submesh, iotdb_isosurface_extraction


_SHARED_CACHE: "OrderedDict[tuple, RuntimeMeshData]" = OrderedDict()
_SHARED_CACHE_LIMIT = 2


class PGMeshRuntime:
    def __init__(self, conn, ship_type: str, scale: str, zone_type: str):
        self.conn = conn
        self.ship_type = ship_type
        self.scale = scale
        self.zone_type = zone_type
        dsn = str(getattr(conn, "dsn", ""))
        self._shared_key = ("postgresql", dsn, ship_type, scale, zone_type)
        self._cache: RuntimeMeshData | None = None

    def _shared(self) -> RuntimeMeshData:
        if self._cache is not None:
            return self._cache
        if self._shared_key in _SHARED_CACHE:
            data = _SHARED_CACHE.pop(self._shared_key)
            _SHARED_CACHE[self._shared_key] = data
        else:
            data = RuntimeMeshData()
            _SHARED_CACHE[self._shared_key] = data
            while len(_SHARED_CACHE) > _SHARED_CACHE_LIMIT:
                _SHARED_CACHE.popitem(last=False)
        self._cache = data
        return data

    def ensure_cells(self) -> RuntimeMeshData:
        data = self._shared()
        if data.cells:
            return data
        with timed_stage("PostgreSQL mesh", f"load centroids dataset={self.ship_type}_{self.scale} zone={self.zone_type}"):
            cur = self.conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT cell_id, x, y, z FROM cell_centroid
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    ORDER BY cell_id
                    """,
                    (self.ship_type, self.scale, self.zone_type),
                )
                for cid, x, y, z in cur.fetchall():
                    xf, yf, zf = float(x), float(y), float(z)
                    pad = 1e-6
                    data.cells[int(cid)] = (
                        xf, yf, zf,
                        xf - pad, xf + pad,
                        yf - pad, yf + pad,
                        zf - pad, zf + pad,
                        0,
                    )
                data.invalidate_cell_views()
            finally:
                cur.close()
        return data

    def ensure_cell_nodes(self) -> RuntimeMeshData:
        data = self.ensure_cells()
        if data.cell_nodes and data.nodes:
            return data
        cur = self.conn.cursor()
        try:
            if not data.cell_nodes:
                cur.execute(
                    """
                    SELECT cell_id, node_ids FROM cell_nodes
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    """,
                    (self.ship_type, self.scale, self.zone_type),
                )
                for cid, nids in cur.fetchall():
                    data.cell_nodes[int(cid)] = [int(n) for n in (nids or [])]
            if not data.nodes:
                cur.execute(
                    """
                    SELECT node_id, x, y, z FROM node_coordinates
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    """,
                    (self.ship_type, self.scale, self.zone_type),
                )
                for nid, x, y, z in cur.fetchall():
                    data.nodes[int(nid)] = (float(x), float(y), float(z))
        finally:
            cur.close()
        return data

    def extract_submesh(self, cell_indexes: Sequence[int]) -> LiteMesh:
        data = self.ensure_cell_nodes()
        return iotdb_extract_submesh(data, cell_indexes)

    def isosurface(self, scalar_map: Dict[int, float], iso_value: float, mesh: LiteMesh | None = None) -> LitePolyData:
        if mesh is not None:
            scoped = RuntimeMeshData(nodes=dict(mesh.node_xyz), cell_nodes=dict(mesh.cell_nodes))
            return iotdb_isosurface_extraction(scoped, scalar_map, float(iso_value))
        data = self.ensure_cell_nodes()
        return iotdb_isosurface_extraction(data, scalar_map, float(iso_value))
