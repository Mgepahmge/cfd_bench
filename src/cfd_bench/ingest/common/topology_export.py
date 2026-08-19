"""Backend-neutral topology export for legacy Tecplot CFD data.

The exporter deliberately contains no PostgreSQL/IoTDB/TileDB assumptions.
Backends receive the same node/cell ids, centroids, bounding boxes, adjacency,
face planes and boundary faces.  This is the canonical contract for the DAT
path; H5 ingest uses its own frozen canonical model.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.ingest.decoder.Zone import Zone_3D

ProgressFn = Optional[Callable[[str, int, int], None]]


def _tick(progress: ProgressFn, phase: str, current: int, total: int) -> None:
    if progress is not None:
        progress(phase, min(max(0, int(current)), max(1, int(total))), max(1, int(total)))


def _stride(total: int, updates: int = 200) -> int:
    return max(1, int(total) // max(1, int(updates)))


def _cell_node_ids(zone: Zone_3D, cid: int) -> np.ndarray:
    return np.asarray(zone.EN.get(int(cid), ()), dtype=np.int64).reshape(-1)


def cell_bbox(
    zone: Zone_3D,
    cid: int,
    *,
    xyz: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, float]:
    """Return a cell AABB without rebuilding the full node matrix per cell."""
    ids = _cell_node_ids(zone, cid)
    if ids.size == 0:
        c = np.asarray([zone.Element_Coordinates[k][cid] for k in range(3)], dtype=np.float64)
        return (float(c[0]), float(c[0]), float(c[1]), float(c[1]), float(c[2]), float(c[2]))
    if xyz is None:
        xyz = np.column_stack(zone.Node_Coordinates[:3]).astype(np.float64, copy=False)
    pts = xyz[ids]
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


def padded(vals: Iterable[int], length: int = 16) -> List[int]:
    arr = [int(v) for v in vals][: int(length)]
    if len(arr) < int(length):
        arr.extend([-1] * (int(length) - len(arr)))
    return arr


def padded_matrix(rows: Sequence[Sequence[int]], width: int | None = None) -> np.ndarray:
    if width is None:
        width = max((len(row) for row in rows), default=0)
    width = int(max(1, width))
    out = np.full((len(rows), width), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        vals = np.asarray(list(row), dtype=np.int64).reshape(-1)
        if vals.size:
            if vals.size > width:
                raise ValueError(f"topology row {i} has {vals.size} values but width={width}")
            out[i, : vals.size] = vals.astype(np.int32, copy=False)
    return out


def _polygon_normal_area(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return a stable unit normal and polygon area for a 3-D face."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return np.zeros(3, dtype=np.float64), 0.0
    nxt = np.roll(pts, -1, axis=0)
    n = np.array(
        [
            np.sum((pts[:, 1] - nxt[:, 1]) * (pts[:, 2] + nxt[:, 2])),
            np.sum((pts[:, 2] - nxt[:, 2]) * (pts[:, 0] + nxt[:, 0])),
            np.sum((pts[:, 0] - nxt[:, 0]) * (pts[:, 1] + nxt[:, 1])),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(n))
    if norm <= 1e-15:
        return np.zeros(3, dtype=np.float64), 0.0
    return n / norm, 0.5 * norm


def face_plane_for_zone(
    zone: Zone_3D,
    *,
    include_interior: bool = True,
    progress: ProgressFn = None,
):
    xyz = np.column_stack(zone.Node_Coordinates[:3]).astype(np.float64, copy=False)
    centers = np.column_stack(zone.Element_Coordinates[:3]).astype(np.float64, copy=False)
    out = []
    boundary = []
    stride = _stride(zone.Face_count)
    _tick(progress, "compute face geometry", 0, zone.Face_count)
    for f in range(zone.Face_count):
        le, re = int(zone.LE[f]), int(zone.RE[f])

        # Canonical CFD currently stores only boundary geometry.  Skip all
        # expensive normal/area work for interior faces before indexing nodes.
        if le >= 0 and re >= 0 and not include_interior:
            if (f + 1) % stride == 0 or f + 1 == zone.Face_count:
                _tick(progress, "compute face geometry", f + 1, zone.Face_count)
            continue

        node_ids = np.asarray(zone.FN[f], dtype=np.int64)
        if node_ids.size >= 3:
            pts = xyz[node_ids]
            face_center = np.mean(pts, axis=0)
            n, area = _polygon_normal_area(pts)
            if area > 0.0:
                if le >= 0 and re >= 0:
                    if float(np.dot(n, centers[re] - centers[le])) < 0.0:
                        n = -n
                    d = -float(np.dot(n, face_center))
                    out.append((le, re, float(n[0]), float(n[1]), float(n[2]), d, area, *face_center.tolist()))
                    out.append((re, le, float(-n[0]), float(-n[1]), float(-n[2]), -d, area, *face_center.tolist()))
                else:
                    cid = le if le >= 0 else re
                    if cid >= 0:
                        if float(np.dot(n, face_center - centers[cid])) < 0.0:
                            n = -n
                        boundary.append(
                            (cid, 0.0, float(n[0]), float(n[1]), float(n[2]), area, *face_center.tolist())
                        )
        if (f + 1) % stride == 0 or f + 1 == zone.Face_count:
            _tick(progress, "compute face geometry", f + 1, zone.Face_count)
    return out, boundary


def boundary_face_nodes_for_zone(zone: Zone_3D, *, progress: ProgressFn = None) -> List[List[int]]:
    xyz = np.column_stack(zone.Node_Coordinates[:3]).astype(np.float64, copy=False)
    out: List[List[int]] = []
    stride = _stride(zone.Face_count)
    _tick(progress, "collect boundary faces", 0, zone.Face_count)
    for f in range(zone.Face_count):
        le, re = int(zone.LE[f]), int(zone.RE[f])
        if (le >= 0) == (re >= 0):
            if (f + 1) % stride == 0 or f + 1 == zone.Face_count:
                _tick(progress, "collect boundary faces", f + 1, zone.Face_count)
            continue
        node_ids = np.asarray(zone.FN[f], dtype=np.int64)
        if node_ids.size >= 3:
            _n, area = _polygon_normal_area(xyz[node_ids])
            if area > 0.0:
                out.append([int(x) for x in node_ids])
        if (f + 1) % stride == 0 or f + 1 == zone.Face_count:
            _tick(progress, "collect boundary faces", f + 1, zone.Face_count)
    return out


def export_zone_topology(zone: Zone_3D, *, progress: ProgressFn = None) -> Dict:
    """Export one decoded CFD zone into the common backend-neutral contract."""
    count = int(zone.Element_count)
    adjacency = zone.construct_element_adjacency(progress=progress)

    stride = _stride(count)
    cell_node_rows: List[List[int]] = []
    _tick(progress, "collect cell connectivity", 0, count)
    for cid in range(count):
        cell_node_rows.append([int(x) for x in _cell_node_ids(zone, cid)])
        if (cid + 1) % stride == 0 or cid + 1 == count:
            _tick(progress, "collect cell connectivity", cid + 1, count)

    adj_rows = [[int(x) for x in adjacency[cid] if int(x) >= 0] for cid in range(count)]
    max_nodes = max((len(x) for x in cell_node_rows), default=0)
    max_neighbors = max((len(x) for x in adj_rows), default=0)

    # IMPORTANT: build xyz exactly once.  v14's cell_bbox() rebuilt this
    # N-by-3 matrix once per cell, causing catastrophic O(cells*nodes) work on
    # real CFD meshes.
    xyz = np.column_stack(zone.Node_Coordinates[:3]).astype(np.float64, copy=False)
    centers = np.column_stack(zone.Element_Coordinates[:3]).astype(np.float64, copy=False)
    cells_rows = []
    _tick(progress, "compute cell bounds", 0, count)
    for cid in range(count):
        ids = np.asarray(cell_node_rows[cid], dtype=np.int64)
        if ids.size:
            pts = xyz[ids]
            lo = np.min(pts, axis=0)
            hi = np.max(pts, axis=0)
        else:
            lo = hi = centers[cid]
        cells_rows.append(
            (
                float(centers[cid, 0]), float(centers[cid, 1]), float(centers[cid, 2]),
                float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]),
                int(len(cell_node_rows[cid])),
            )
        )
        if (cid + 1) % stride == 0 or cid + 1 == count:
            _tick(progress, "compute cell bounds", cid + 1, count)

    # W1 line/plane use AABBs and W7 uses adjacency.  Only boundary geometry
    # is needed for W6, so interior faces are skipped before normal work.
    face_rows, boundary_rows = face_plane_for_zone(zone, include_interior=False, progress=progress)
    boundary_node_rows = boundary_face_nodes_for_zone(zone, progress=progress)
    if len(boundary_node_rows) != len(boundary_rows):
        raise ValueError("boundary face geometry/node payload mismatch")

    X, Y, Z = zone.Node_Coordinates[:3]
    return {
        "zone_name": str(zone.Zone_name).strip().replace(" ", "_") or "Zone_0",
        "zone_type": str(zone.Zone_type),
        "node_count": int(zone.Node_count),
        "cell_count": count,
        "face_count": int(zone.Face_count),
        "max_nodes_per_cell": int(max_nodes),
        "max_neighbors_per_cell": int(max_neighbors),
        "nodes": {
            "x": np.asarray(X, dtype=np.float64),
            "y": np.asarray(Y, dtype=np.float64),
            "z": np.asarray(Z, dtype=np.float64),
        },
        "cells": cells_rows,
        "cell_nodes": cell_node_rows,
        "adjacency": adj_rows,
        "face_planes": face_rows,
        "boundary_faces": boundary_rows,
        "boundary_face_nodes": boundary_node_rows,
        "bbox": {
            "bbox_min_x": float(np.min(X)), "bbox_max_x": float(np.max(X)),
            "bbox_min_y": float(np.min(Y)), "bbox_max_y": float(np.max(Y)),
            "bbox_min_z": float(np.min(Z)), "bbox_max_z": float(np.max(Z)),
        },
    }
