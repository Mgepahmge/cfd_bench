"""Gradient and Q-criterion utilities for W7/W8."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData


def estimate_gradient_least_squares(center_xyz, center_vel, nb_xyz_list, nb_vel_list):
    if len(nb_xyz_list) < 3:
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
    if len(A) < 3:
        return None
    A = np.array(A, dtype=np.float64)
    dU = np.array(dU, dtype=np.float64)
    G = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        gi, *_ = np.linalg.lstsq(A, dU[:, i], rcond=None)
        G[i, :] = gi
    return G


def qcriterion_from_gradient(grad_u: np.ndarray) -> float:
    S = 0.5 * (grad_u + grad_u.T)
    O = 0.5 * (grad_u - grad_u.T)
    return 0.5 * (float(np.sum(O * O)) - float(np.sum(S * S)))


def compute_qcriterion_roi(
    data: RuntimeMeshData,
    roi_cell_ids: Sequence[int],
    velocity_map: Dict[int, Tuple[float, float, float]],
    tau: Optional[float] = None,
) -> Tuple[List[int], List[float]]:
    """Online Q-criterion for cells in ROI using adjacency + least-squares gradient."""
    centroids = data.cell_centroid
    qc_rows: List[Tuple[int, float]] = []
    for cid in roi_cell_ids:
        cxyz = centroids.get(int(cid))
        cvel = velocity_map.get(int(cid))
        if cxyz is None or cvel is None:
            continue
        nb_ids = data.adjacency.get(int(cid), [])
        nb_xyz = []
        nb_vel = []
        for nb in nb_ids:
            if nb not in velocity_map or nb not in centroids:
                continue
            nb_xyz.append(centroids[nb])
            nb_vel.append(velocity_map[nb])
        G = estimate_gradient_least_squares(cxyz, cvel, nb_xyz, nb_vel)
        if G is None:
            continue
        qc = qcriterion_from_gradient(G)
        if tau is None or qc >= float(tau):
            qc_rows.append((int(cid), float(qc)))
    cell_ids = [r[0] for r in qc_rows]
    qvals = [r[1] for r in qc_rows]
    return cell_ids, qvals
