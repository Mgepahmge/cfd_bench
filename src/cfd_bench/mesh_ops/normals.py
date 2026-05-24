from __future__ import annotations

from typing import List

import numpy as np

from cfd_bench.core.types import LiteMesh, LitePolyData


def iotdb_surface_norm_from_mesh(mesh: LiteMesh) -> np.ndarray:
    """Compute per-cell normal from first 3 nodes of each cell."""
    normals: List[np.ndarray] = []
    for cid in mesh.cell_ids:
        node_ids = mesh.cell_nodes.get(int(cid), [])
        if len(node_ids) < 3:
            normals.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))
            continue
        pts = [mesh.node_xyz.get(int(nid)) for nid in node_ids[:3]]
        if any(p is None for p in pts):
            normals.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))
            continue
        p0 = np.array(pts[0], dtype=np.float64)
        p1 = np.array(pts[1], dtype=np.float64)
        p2 = np.array(pts[2], dtype=np.float64)
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            normals.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        else:
            normals.append((n / norm).astype(np.float64))
    return np.array(normals, dtype=np.float64)


def iotdb_surface_norm(poly: LitePolyData) -> np.ndarray:
    if poly.points.size == 0 or poly.triangles.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    normals: List[np.ndarray] = []
    for tri in poly.triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0 = poly.points[i0]
        p1 = poly.points[i1]
        p2 = poly.points[i2]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            normals.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        else:
            normals.append((n / norm).astype(np.float64))
    return np.array(normals, dtype=np.float64)
