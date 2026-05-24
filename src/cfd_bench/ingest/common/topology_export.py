"""Shared topology export from Zone_3D to structured dicts."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from cfd_bench.ingest.decoder.Zone import Zone_3D


def cell_bbox(zone: Zone_3D, cid: int) -> Tuple[float, float, float, float, float, float]:
    ids = list(zone.EN.get(cid, []))
    if not ids:
        c = (
            float(zone.Element_Coordinates[0][cid]),
            float(zone.Element_Coordinates[1][cid]),
            float(zone.Element_Coordinates[2][cid]),
        )
        return (c[0], c[0], c[1], c[1], c[2], c[2])
    X = zone.Node_Coordinates[0]
    Y = zone.Node_Coordinates[1]
    Z = zone.Node_Coordinates[2]
    xs = [float(X[i]) for i in ids]
    ys = [float(Y[i]) for i in ids]
    zs = [float(Z[i]) for i in ids]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def padded(vals: Iterable[int], length: int = 16) -> List[float]:
    arr = [float(int(v)) for v in vals][:length]
    if len(arr) < length:
        arr.extend([-1.0] * (length - len(arr)))
    return arr


def face_plane_for_zone(zone: Zone_3D):
    X = zone.Node_Coordinates[0]
    Y = zone.Node_Coordinates[1]
    Z = zone.Node_Coordinates[2]
    EcX = zone.Element_Coordinates[0]
    EcY = zone.Element_Coordinates[1]
    EcZ = zone.Element_Coordinates[2]
    LE = zone.LE
    RE = zone.RE
    FN = zone.FN
    out = []
    boundary = []
    for f in range(zone.Face_count):
        le, re = int(LE[f]), int(RE[f])
        node_ids = FN[f]
        if len(node_ids) < 3:
            continue
        pts = np.array([[float(X[n]), float(Y[n]), float(Z[n])] for n in node_ids], dtype=np.float64)
        face_center = pts.mean(axis=0)
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        n = np.cross(v0, v1)
        nnorm = np.linalg.norm(n)
        if nnorm < 1e-15:
            continue
        n = n / nnorm
        area = float(max(1e-12, nnorm * 0.5))

        if le >= 0 and re >= 0:
            le_centroid = np.array([EcX[le], EcY[le], EcZ[le]], dtype=np.float64)
            re_centroid = np.array([EcX[re], EcY[re], EcZ[re]], dtype=np.float64)
            if np.dot(n, re_centroid - le_centroid) < 0:
                n = -n
            d = -float(np.dot(n, face_center))
            out.append((le, re, float(n[0]), float(n[1]), float(n[2]), d, area, *face_center))
            out.append((re, le, float(-n[0]), float(-n[1]), float(-n[2]), -d, area, *face_center))
        elif le >= 0:
            d = -float(np.dot(n, face_center))
            boundary.append((le, 0.0, float(n[0]), float(n[1]), float(n[2]), area, *face_center))
        elif re >= 0:
            d = -float(np.dot(-n, face_center))
            boundary.append((re, 0.0, float(-n[0]), float(-n[1]), float(-n[2]), area, *face_center))
    return out, boundary


def export_zone_topology(zone: Zone_3D) -> Dict:
    """Export full topology dict for backend-specific writers."""
    cell_ids = list(range(zone.Element_count))
    cells_rows = []
    for cid in cell_ids:
        bb = cell_bbox(zone, cid)
        cells_rows.append(
            (
                float(zone.Element_Coordinates[0][cid]),
                float(zone.Element_Coordinates[1][cid]),
                float(zone.Element_Coordinates[2][cid]),
                bb[0],
                bb[1],
                bb[2],
                bb[3],
                bb[4],
                bb[5],
                float(len(zone.EN.get(cid, []))),
            )
        )
    adj = zone.construct_element_adjacency()
    adj_rows = [
        padded([x for x in (adj[cid] if cid < len(adj) else []) if int(x) >= 0], 16) for cid in cell_ids
    ]
    node_rows = [padded(zone.EN.get(cid, []), 16) for cid in cell_ids]
    face_rows, boundary_rows = face_plane_for_zone(zone)
    X = zone.Node_Coordinates[0]
    Y = zone.Node_Coordinates[1]
    Z = zone.Node_Coordinates[2]
    return {
        "node_count": zone.Node_count,
        "cell_count": zone.Element_count,
        "face_count": zone.Face_count,
        "nodes": {
            "x": zone.Node_Coordinates[0].astype(np.float64),
            "y": zone.Node_Coordinates[1].astype(np.float64),
            "z": zone.Node_Coordinates[2].astype(np.float64),
        },
        "cells": cells_rows,
        "cell_nodes": node_rows,
        "adjacency": adj_rows,
        "face_planes": face_rows,
        "boundary_faces": boundary_rows,
        "bbox": {
            "bbox_min_x": float(np.min(X)),
            "bbox_max_x": float(np.max(X)),
            "bbox_min_y": float(np.min(Y)),
            "bbox_max_y": float(np.max(Y)),
            "bbox_min_z": float(np.min(Z)),
            "bbox_max_z": float(np.max(Z)),
        },
    }
