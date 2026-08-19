"""Backend-neutral topology export for legacy Tecplot CFD data.

The exporter deliberately contains no PostgreSQL/IoTDB/TileDB assumptions.
Backends receive the same node/cell ids, centroids, bounding boxes, adjacency,
face planes and boundary faces.  This is the canonical contract for the DAT
path; H5 ingest uses its own frozen canonical model.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from cfd_bench.ingest.decoder.Zone import Zone_3D


def _cell_node_ids(zone: Zone_3D, cid: int) -> np.ndarray:
    return np.asarray(zone.EN.get(int(cid), ()), dtype=np.int64).reshape(-1)


def cell_bbox(zone: Zone_3D, cid: int) -> Tuple[float, float, float, float, float, float]:
    ids = _cell_node_ids(zone, cid)
    if ids.size == 0:
        c = np.asarray([zone.Element_Coordinates[k][cid] for k in range(3)], dtype=np.float64)
        return (float(c[0]), float(c[0]), float(c[1]), float(c[1]), float(c[2]), float(c[2]))
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
    # Newell normal is robust for quads/ngons and its norm is 2*area for a
    # planar polygon.
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


def face_plane_for_zone(zone: Zone_3D, *, include_interior: bool = True):
    xyz = np.column_stack(zone.Node_Coordinates[:3]).astype(np.float64, copy=False)
    centers = np.column_stack(zone.Element_Coordinates[:3]).astype(np.float64, copy=False)
    out = []
    boundary = []
    for f in range(zone.Face_count):
        le, re = int(zone.LE[f]), int(zone.RE[f])
        node_ids = np.asarray(zone.FN[f], dtype=np.int64)
        if node_ids.size < 3:
            continue
        pts = xyz[node_ids]
        face_center = np.mean(pts, axis=0)
        n, area = _polygon_normal_area(pts)
        if area <= 0.0:
            continue

        if le >= 0 and re >= 0:
            if include_interior:
                # Orient from left cell toward right cell.
                if float(np.dot(n, centers[re] - centers[le])) < 0.0:
                    n = -n
                d = -float(np.dot(n, face_center))
                out.append((le, re, float(n[0]), float(n[1]), float(n[2]), d, area, *face_center.tolist()))
                out.append((re, le, float(-n[0]), float(-n[1]), float(-n[2]), -d, area, *face_center.tolist()))
        else:
            cid = le if le >= 0 else re
            if cid < 0:
                continue
            # Boundary normal must point away from the owning cell centroid.
            if float(np.dot(n, face_center - centers[cid])) < 0.0:
                n = -n
            d = -float(np.dot(n, face_center))
            boundary.append(
                (cid, 0.0, float(n[0]), float(n[1]), float(n[2]), area, *face_center.tolist())
            )
    return out, boundary



def boundary_face_nodes_for_zone(zone: Zone_3D) -> List[List[int]]:
    xyz = np.column_stack(zone.Node_Coordinates[:3]).astype(np.float64, copy=False)
    out: List[List[int]] = []
    for f in range(zone.Face_count):
        le, re = int(zone.LE[f]), int(zone.RE[f])
        if (le >= 0) == (re >= 0):
            continue
        node_ids = np.asarray(zone.FN[f], dtype=np.int64)
        if node_ids.size < 3:
            continue
        _n, area = _polygon_normal_area(xyz[node_ids])
        if area <= 0.0:
            continue
        out.append([int(x) for x in node_ids])
    return out


def export_zone_topology(zone: Zone_3D) -> Dict:
    """Export one decoded CFD zone into the common backend-neutral contract."""
    cell_ids = list(range(int(zone.Element_count)))
    adjacency = zone.construct_element_adjacency()
    cell_node_rows = [list(map(int, _cell_node_ids(zone, cid))) for cid in cell_ids]
    adj_rows = [[int(x) for x in adjacency[cid] if int(x) >= 0] for cid in cell_ids]
    max_nodes = max((len(x) for x in cell_node_rows), default=0)
    max_neighbors = max((len(x) for x in adj_rows), default=0)

    cells_rows = []
    for cid in cell_ids:
        bb = cell_bbox(zone, cid)
        cells_rows.append(
            (
                float(zone.Element_Coordinates[0][cid]),
                float(zone.Element_Coordinates[1][cid]),
                float(zone.Element_Coordinates[2][cid]),
                bb[0], bb[1], bb[2], bb[3], bb[4], bb[5],
                int(len(cell_node_rows[cid])),
            )
        )

    # W1 line/plane now use cell AABBs and W7 uses adjacency; retaining two
    # interior plane tuples per face only bloats million-face CFD meshes.
    # Boundary geometry is still materialised for W6.
    face_rows, boundary_rows = face_plane_for_zone(zone, include_interior=False)
    boundary_node_rows = boundary_face_nodes_for_zone(zone)
    if len(boundary_node_rows) != len(boundary_rows):
        raise ValueError("boundary face geometry/node payload mismatch")
    X, Y, Z = zone.Node_Coordinates[:3]
    return {
        "zone_name": str(zone.Zone_name).strip().replace(" ", "_") or "Zone_0",
        "zone_type": str(zone.Zone_type),
        "node_count": int(zone.Node_count),
        "cell_count": int(zone.Element_count),
        "face_count": int(zone.Face_count),
        "max_nodes_per_cell": int(max_nodes),
        "max_neighbors_per_cell": int(max_neighbors),
        "nodes": {
            "x": np.asarray(X, dtype=np.float64),
            "y": np.asarray(Y, dtype=np.float64),
            "z": np.asarray(Z, dtype=np.float64),
        },
        "cells": cells_rows,
        # Keep unpadded lists here.  Each backend may choose its native
        # representation without losing topology.
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
