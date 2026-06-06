"""In-memory mesh assembly from PostgreSQL tables."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.core.types import LiteMesh, LitePolyData
from cfd_bench.mesh_ops import iotdb_extract_submesh, iotdb_isosurface_extraction


class PGMeshRuntime:
    def __init__(self, conn, ship_type: str, scale: str, zone_type: str):
        self.conn = conn
        self.ship_type = ship_type
        self.scale = scale
        self.zone_type = zone_type
        self._cache: RuntimeMeshData | None = None

    def ensure_cells(self) -> RuntimeMeshData:
        if self._cache is not None and self._cache.cells:
            return self._cache
        data = RuntimeMeshData()
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
                data.cells[int(cid)] = (xf, yf, zf, xf - pad, xf + pad, yf - pad, yf + pad, zf - pad, zf + pad, 0)
                data.cell_bbox[int(cid)] = (xf - pad, xf + pad, yf - pad, yf + pad, zf - pad, zf + pad)
        finally:
            cur.close()
        self._cache = data
        return data

    def ensure_cell_nodes(self) -> RuntimeMeshData:
        data = self.ensure_cells()
        if data.cell_nodes:
            return data
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id, node_ids FROM cell_nodes
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                """,
                (self.ship_type, self.scale, self.zone_type),
            )
            for cid, nids in cur.fetchall():
                data.cell_nodes[int(cid)] = [int(n) for n in (nids or [])]
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

    def isosurface(self, scalar_map: Dict[int, float], iso_value: float) -> LitePolyData:
        data = self.ensure_cell_nodes()
        return iotdb_isosurface_extraction(data, scalar_map, float(iso_value))
