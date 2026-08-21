"""Runtime fluid linear interpolation backed by Apache IoTDB.

Legacy Tecplot CFD datasets persist coordinates at mesh nodes but U/V/W/P/K/E
at cell centres.  This feature intentionally does not change that frozen ingest
contract.  For a target point it:

1. finds the containing convex cell from persisted AABBs + exact barycentric
   containment;
2. derives values at that cell's vertices by averaging the values of all cells
   incident to each vertex (the same runtime projection used by CFD W11);
3. evaluates a piecewise-linear barycentric interpolation at the target point.

No derived value is written back to IoTDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.cfd_nodal_projection import NodeCellCSR, build_node_cell_csr
from cfd_bench.core.observability import timed_stage
from cfd_bench.infra.iotdb.mesh_runtime import MeshRuntime
from cfd_bench.infra.iotdb.repository import IoTDBRepository


@dataclass(frozen=True)
class LinearSupport:
    """Four mesh vertices and barycentric weights supporting one target point."""

    local_indices: Tuple[int, int, int, int]
    weights: np.ndarray
    reconstruction_error: float
    condition_number: float
    margin: float


@dataclass(frozen=True)
class FluidInterpolationResult:
    dataset: str
    step: int
    zone: str
    target: np.ndarray
    cell_id: int
    cell_node_ids: Tuple[int, ...]
    support_node_ids: Tuple[int, int, int, int]
    weights: np.ndarray
    reconstruction_error: float
    vertex_value_source: str
    values: Mapping[str, float]
    support_vertex_values: Mapping[str, Tuple[float, float, float, float]]

    @property
    def source_element_id(self) -> int:
        """Tecplot CFD element IDs are implicit and one-based."""
        return int(self.cell_id) + 1

    @property
    def support_source_node_ids(self) -> Tuple[int, int, int, int]:
        """Tecplot CFD node IDs are implicit and one-based."""
        return tuple(int(x) + 1 for x in self.support_node_ids)



def _cell_scale(points: np.ndarray) -> float:
    span = np.ptp(np.asarray(points, dtype=np.float64), axis=0)
    return max(float(np.max(span)), 1.0)


def _barycentric_weights(point: np.ndarray, tetra: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Return translation-invariant barycentric weights for a tetrahedron."""
    pts = np.asarray(tetra, dtype=np.float64).reshape(4, 3)
    # Use the fourth vertex as the affine origin.  This avoids the poor
    # conditioning of a 4x4 homogeneous solve when mesh coordinates carry a
    # large global offset.
    matrix = np.column_stack((pts[0] - pts[3], pts[1] - pts[3], pts[2] - pts[3]))
    rhs = np.asarray(point, dtype=np.float64).reshape(3) - pts[3]
    try:
        cond = float(np.linalg.cond(matrix))
        if not np.isfinite(cond) or cond > 1.0e14:
            return None, cond
        first = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None, np.inf
    weights = np.empty((4,), dtype=np.float64)
    weights[:3] = first
    weights[3] = 1.0 - float(np.sum(first))
    return weights, cond


def find_linear_support(
    point: Sequence[float],
    vertices: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-9,
) -> Optional[LinearSupport]:
    """Find a stable containing tetrahedron formed only from cell vertices.

    For a convex 3-D cell, Caratheodory's theorem guarantees that an interior
    point can be represented by at most four cell vertices.  Enumerating those
    four-vertex simplices makes the interpolation independent of Tecplot's
    local node ordering, which is important because the canonical CFD ingest
    stores each cell's node set sorted by dense node ID.
    """
    p = np.asarray(point, dtype=np.float64).reshape(3)
    pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        return None

    scale = _cell_scale(pts)
    coord_tol = max(float(tolerance) * scale, 1.0e-12)
    weight_tol = max(float(tolerance) * 10.0, 1.0e-10)
    best = None
    best_score = None

    for combo in combinations(range(pts.shape[0]), 4):
        tetra = pts[np.asarray(combo, dtype=np.int64)]
        weights, cond = _barycentric_weights(p, tetra)
        if weights is None:
            continue
        if np.any(weights < -weight_tol) or np.any(weights > 1.0 + weight_tol):
            continue
        reconstructed = weights @ tetra
        error = float(np.linalg.norm(reconstructed - p))
        if not np.isfinite(error) or error > coord_tol * 10.0:
            continue
        margin = float(np.min(weights))
        # Prefer a simplex that contains the point well inside; use numerical
        # conditioning and reconstruction error as deterministic tie-breakers.
        score = (margin, -np.log10(max(cond, 1.0)), -error)
        if best_score is None or score > best_score:
            best_score = score
            best = LinearSupport(
                local_indices=tuple(int(x) for x in combo),
                weights=np.asarray(weights, dtype=np.float64),
                reconstruction_error=error,
                condition_number=cond,
                margin=margin,
            )
    return best


class FluidInterpolationEngine:
    """Execute one or more fluid interpolation mappings against one IoTDB session."""

    def __init__(self, repo: IoTDBRepository):
        self.repo = repo
        self.runtime = MeshRuntime(repo)
        self._csr_cache: Dict[Tuple[str, str], NodeCellCSR] = {}

    def _require_cfd_metadata(self, dataset: str) -> Dict[str, object]:
        meta = self.repo.cfd_dataset_metadata(dataset)
        if not meta or not bool(meta.get("is_cfd")):
            raise ValueError(
                f"dataset={dataset!r} is not a CFD dataset in IoTDB; "
                "fluid interpolation currently targets Tecplot CFD data only"
            )
        return meta

    def _resolve_zone(self, meta: Mapping[str, object], zone: Optional[str]) -> str:
        if zone:
            return str(zone)
        return str(meta.get("zone") or "0_Fluid")

    @staticmethod
    def _resolve_variables(meta: Mapping[str, object], variables: Optional[Sequence[str]]) -> Tuple[str, ...]:
        available = tuple(str(v).upper() for v in meta.get("variables", ()))
        requested = tuple(str(v).upper() for v in variables) if variables else available
        if not requested:
            raise ValueError("CFD metadata contains no physical variables")
        missing = [v for v in requested if v not in available]
        if missing:
            raise ValueError(
                f"variables not available in CFD dataset: {missing}; available={list(available)}"
            )
        return requested

    @staticmethod
    def _validate_step(meta: Mapping[str, object], step: int) -> int:
        value = int(step)
        steps = tuple(int(x) for x in meta.get("timesteps", ()))
        if steps and value not in steps:
            raise ValueError(f"step={value} is not available; available={list(steps)}")
        return value

    def _candidate_cells(self, dataset: str, zone: str, target: np.ndarray) -> np.ndarray:
        data = self.runtime.ensure_cells(dataset, zone)
        ids = np.asarray(data.all_cell_ids, dtype=np.int32)
        mins = np.asarray(data.all_bbox_min, dtype=np.float64)
        maxs = np.asarray(data.all_bbox_max, dtype=np.float64)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.int32)
        scale = max(float(np.max(np.max(maxs, axis=0) - np.min(mins, axis=0))), 1.0)
        eps = max(1.0e-10 * scale, 1.0e-12)
        mask = np.all(mins - eps <= target, axis=1) & np.all(maxs + eps >= target, axis=1)
        hits = ids[mask]
        if hits.size <= 1:
            return hits
        centers = np.asarray(data.all_centroids, dtype=np.float64)[mask]
        order = np.argsort(np.linalg.norm(centers - target[None, :], axis=1), kind="stable")
        return hits[order]

    def _locate_cell(
        self, dataset: str, zone: str, target: np.ndarray
    ) -> Tuple[int, Tuple[int, ...], np.ndarray, LinearSupport]:
        with timed_stage("Fluid interpolation", "find candidate cells"):
            candidate_ids = self._candidate_cells(dataset, zone, target)
        if candidate_ids.size == 0:
            raise ValueError(
                f"target={tuple(float(x) for x in target)} is outside the mesh AABB"
            )

        with timed_stage("Fluid interpolation", "verify exact cell containment"):
            connectivity = self.repo.fetch_cell_nodes_subset(dataset, zone, candidate_ids.tolist())
            node_ids = sorted({int(n) for row in connectivity.values() for n in row})
            coordinates = self.repo.fetch_nodes_subset(dataset, zone, node_ids)

            best = None
            best_key = None
            for cid in candidate_ids.tolist():
                ids = tuple(int(x) for x in connectivity.get(int(cid), ()))
                if len(ids) < 4 or any(nid not in coordinates for nid in ids):
                    continue
                pts = np.asarray([coordinates[nid] for nid in ids], dtype=np.float64)
                support = find_linear_support(target, pts)
                if support is None:
                    continue
                # Prefer the cell/simplex with the strongest interior margin,
                # then the better-conditioned simplex, then the lower cell id.
                key = (support.margin, -support.condition_number, -int(cid))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (int(cid), ids, pts, support)
        if best is None:
            raise ValueError(
                f"target={tuple(float(x) for x in target)} lies inside one or more cell AABBs "
                "but no containing convex cell could be verified"
            )
        return best

    def _node_cell_csr(self, dataset: str, zone: str) -> NodeCellCSR:
        key = (str(dataset), str(zone))
        cached = self._csr_cache.get(key)
        if cached is not None:
            return cached
        with timed_stage("Fluid interpolation", "build node-to-cell incidence"):
            meta = self.repo.fetch_mesh_meta(dataset, zone)
            node_count = int(meta.get("node_count", 0) or 0)
            cell_nodes = self.repo.fetch_cell_nodes(dataset, zone)
            if node_count <= 0 and cell_nodes:
                node_count = 1 + max(
                    (max(nodes) for nodes in cell_nodes.values() if nodes),
                    default=-1,
                )
            csr = build_node_cell_csr(cell_nodes, node_count)
            self._csr_cache[key] = csr
            return csr

    def interpolate(
        self,
        dataset: str,
        step: int,
        target: Sequence[float],
        *,
        variables: Optional[Sequence[str]] = None,
        zone: Optional[str] = None,
    ) -> FluidInterpolationResult:
        meta = self._require_cfd_metadata(str(dataset))
        resolved_zone = self._resolve_zone(meta, zone)
        resolved_step = self._validate_step(meta, int(step))
        resolved_variables = self._resolve_variables(meta, variables)
        point = np.asarray(target, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point)):
            raise ValueError("target coordinate must contain three finite numbers")

        cell_id, cell_node_ids, cell_points, support = self._locate_cell(
            str(dataset), resolved_zone, point
        )
        csr = self._node_cell_csr(str(dataset), resolved_zone)

        # Collect the union of cells incident to every vertex of the containing
        # cell.  This is the runtime-only cell-centre -> vertex projection that
        # CFD W11 already uses; no nodal field is persisted.
        incident_per_node = []
        all_incident = []
        for nid in cell_node_ids:
            incident = np.asarray(csr.incident_cells(int(nid)), dtype=np.int32)
            incident_per_node.append(incident)
            if incident.size:
                all_incident.append(incident)
        if not all_incident:
            raise RuntimeError(f"containing cell={cell_id} has no node-to-cell incidence")
        unique_cells = np.unique(np.concatenate(all_incident)).astype(np.int32, copy=False)
        cell_pos = {int(cid): i for i, cid in enumerate(unique_cells.tolist())}

        support_idx = np.asarray(support.local_indices, dtype=np.int64)
        support_node_ids = tuple(int(cell_node_ids[i]) for i in support_idx.tolist())
        values_out: Dict[str, float] = {}
        support_values_out: Dict[str, Tuple[float, float, float, float]] = {}

        with timed_stage("Fluid interpolation", "project cell values to vertices + interpolate"):
            for var in resolved_variables:
                cell_values = self.repo.fetch_cell_scalar_values(
                    str(dataset), resolved_step, str(var), unique_cells.tolist(), zone=resolved_zone
                )
                vertex_values = np.full((len(cell_node_ids),), np.nan, dtype=np.float64)
                for i, incident in enumerate(incident_per_node):
                    if incident.size == 0:
                        continue
                    idx = np.asarray([cell_pos[int(cid)] for cid in incident if int(cid) in cell_pos], dtype=np.int64)
                    if idx.size == 0:
                        continue
                    vals = np.asarray(cell_values[idx], dtype=np.float64)
                    finite = vals[np.isfinite(vals)]
                    if finite.size:
                        vertex_values[i] = float(np.mean(finite))
                support_values = vertex_values[support_idx]
                if support_values.size != 4 or not np.all(np.isfinite(support_values)):
                    raise RuntimeError(
                        f"cannot derive finite vertex values for variable={var} in cell={cell_id}"
                    )
                interpolated = float(np.dot(support.weights, support_values))
                values_out[str(var)] = interpolated
                support_values_out[str(var)] = tuple(float(x) for x in support_values.tolist())

        return FluidInterpolationResult(
            dataset=str(dataset),
            step=resolved_step,
            zone=resolved_zone,
            target=point,
            cell_id=int(cell_id),
            cell_node_ids=tuple(int(x) for x in cell_node_ids),
            support_node_ids=support_node_ids,
            weights=np.asarray(support.weights, dtype=np.float64),
            reconstruction_error=float(support.reconstruction_error),
            vertex_value_source="mean of incident cell-centered CFD values",
            values=values_out,
            support_vertex_values=support_values_out,
        )
