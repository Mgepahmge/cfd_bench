from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .mesh_runtime import RuntimeMeshData
from .types import LitePolyData


TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def _interp(p1: np.ndarray, p2: np.ndarray, v1: float, v2: float, iso: float) -> np.ndarray:
    dv = v2 - v1
    if abs(dv) < 1e-15:
        return (p1 + p2) * 0.5
    t = (iso - v1) / dv
    t = max(0.0, min(1.0, t))
    return p1 + t * (p2 - p1)


def _cell_tets(node_ids: List[int]) -> List[Tuple[int, int, int, int]]:
    if len(node_ids) < 4:
        return []
    if len(node_ids) == 4:
        return [(0, 1, 2, 3)]
    base = 0
    tets = []
    for i in range(1, len(node_ids) - 2):
        tets.append((base, i, i + 1, i + 2))
    return tets


def iotdb_isosurface_extraction(
    data: RuntimeMeshData,
    scalar_by_cell: Dict[int, float],
    iso_value: float,
) -> LitePolyData:
    # NOTE: fallback strategy: project cell scalar to all nodes of a cell.
    # This is an approximation for cell-centered variables.
    points: List[np.ndarray] = []
    tris: List[Tuple[int, int, int]] = []

    for cid, node_ids in data.cell_nodes.items():
        if len(node_ids) < 4:
            continue
        cell_scalar = scalar_by_cell.get(int(cid))
        if cell_scalar is None or np.isnan(cell_scalar):
            continue
        tet_defs = _cell_tets(node_ids)
        for tet in tet_defs:
            p = []
            v = []
            valid = True
            for idx in tet:
                nid = int(node_ids[idx])
                xyz = data.nodes.get(nid)
                if xyz is None:
                    valid = False
                    break
                p.append(np.array(xyz, dtype=np.float64))
                v.append(float(cell_scalar))
            if not valid:
                continue
            # all equal in fallback => no surface
            if max(v) - min(v) < 1e-15:
                continue

            cross_pts: List[np.ndarray] = []
            for a, b in TET_EDGES:
                va, vb = v[a], v[b]
                if (va - iso_value) * (vb - iso_value) < 0:
                    cross_pts.append(_interp(p[a], p[b], va, vb, iso_value))
            if len(cross_pts) < 3:
                continue
            i0 = len(points)
            for cp in cross_pts[:3]:
                points.append(cp)
            tris.append((i0, i0 + 1, i0 + 2))

    if not points:
        return LitePolyData(
            points=np.zeros((0, 3), dtype=np.float64),
            triangles=np.zeros((0, 3), dtype=np.int32),
            meta={"iso_value": str(iso_value)},
        )
    return LitePolyData(
        points=np.array(points, dtype=np.float64),
        triangles=np.array(tris, dtype=np.int32),
        meta={"iso_value": str(iso_value)},
    )
