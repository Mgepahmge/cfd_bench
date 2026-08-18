"""Gradient and Q-criterion utilities for W7/W8."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData


def estimate_gradient_least_squares(
    center_xyz,
    center_vel,
    nb_xyz_list,
    nb_vel_list,
    *,
    min_neighbors: int = 3,
):
    if len(nb_xyz_list) < int(min_neighbors):
        return None
    A = []
    dU = []
    c = np.array(center_xyz, dtype=np.float64)
    u = np.array(center_vel, dtype=np.float64)
    for xyz, vel in zip(nb_xyz_list, nb_vel_list):
        dx = np.array(xyz, dtype=np.float64) - c
        du = np.array(vel, dtype=np.float64) - u
        if np.linalg.norm(dx) < 1e-15:
            continue
        A.append(dx)
        dU.append(du)
    if len(A) < int(min_neighbors):
        return None
    A = np.asarray(A, dtype=np.float64)
    dU = np.asarray(dU, dtype=np.float64)
    # np.linalg.lstsq supports multiple right-hand sides.  Solving U/V/W in a
    # single factorization avoids doing the same SVD/QR work three times per
    # cell, which is a major W7 cost on large ROIs.
    try:
        coeff, *_ = np.linalg.lstsq(A, dU, rcond=None)
    except Exception:
        return None
    return np.asarray(coeff.T, dtype=np.float64)


def qcriterion_from_gradient(grad_u: np.ndarray) -> float:
    S = 0.5 * (grad_u + grad_u.T)
    O = 0.5 * (grad_u - grad_u.T)
    return 0.5 * (float(np.sum(O * O)) - float(np.sum(S * S)))



def _centroid_subset(data: RuntimeMeshData, cell_ids: Sequence[int]) -> Dict[int, Tuple[float, float, float]]:
    """Materialize centroids only for the ROI/halo ids needed by W7."""
    wanted = np.asarray(sorted(set(int(x) for x in cell_ids)), dtype=np.int64)
    if wanted.size == 0:
        return {}
    ids = np.asarray(data.all_cell_ids, dtype=np.int64).reshape(-1)
    centers = np.asarray(data.all_centroids, dtype=np.float64)
    if ids.size and centers.shape == (ids.size, 3):
        pos = np.searchsorted(ids, wanted)
        valid = (pos >= 0) & (pos < ids.size)
        clipped = np.clip(pos, 0, max(ids.size - 1, 0))
        valid &= ids[clipped] == wanted
        return {
            int(cid): tuple(float(x) for x in centers[int(idx)])
            for cid, idx, ok in zip(wanted.tolist(), pos.tolist(), valid.tolist())
            if ok
        }
    return {
        int(cid): tuple(float(x) for x in data.cells[int(cid)][:3])
        for cid in wanted.tolist()
        if int(cid) in data.cells
    }

def compute_qcriterion_roi(
    data: RuntimeMeshData,
    roi_cell_ids: Sequence[int],
    velocity_map: Dict[int, Tuple[float, float, float]],
    tau: Optional[float] = None,
    *,
    min_neighbors: int = 3,
    fallback_zero: bool = False,
) -> Tuple[List[int], List[float]]:
    """Online Q-criterion for cells in ROI using adjacency + least-squares gradient."""
    qc_rows: List[Tuple[int, float]] = []
    needed = set(int(cid) for cid in roi_cell_ids)
    for cid in list(needed):
        needed.update(int(nb) for nb in data.adjacency.get(int(cid), ()))
    centroid_map = _centroid_subset(data, needed)
    for cid in roi_cell_ids:
        cxyz = centroid_map.get(int(cid))
        cvel = velocity_map.get(int(cid))
        if cxyz is None or cvel is None:
            continue
        nb_ids = data.adjacency.get(int(cid), [])
        nb_xyz = []
        nb_vel = []
        for nb in nb_ids:
            nxyz = centroid_map.get(int(nb))
            if nb not in velocity_map or nxyz is None:
                continue
            nb_xyz.append(nxyz)
            nb_vel.append(velocity_map[nb])
        G = estimate_gradient_least_squares(
            cxyz,
            cvel,
            nb_xyz,
            nb_vel,
            min_neighbors=min_neighbors,
        )
        if G is None:
            if not fallback_zero:
                continue
            qc = 0.0
        else:
            qc = qcriterion_from_gradient(G)
        if tau is None or qc >= float(tau):
            qc_rows.append((int(cid), float(qc)))
    cell_ids = [r[0] for r in qc_rows]
    qvals = [r[1] for r in qc_rows]
    return cell_ids, qvals
