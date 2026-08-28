"""High-throughput mapping of structural H5 nodes onto one CFD frame.

Both source datasets are read from Apache IoTDB.  The structural dataset
provides dense node ids, original/source node labels, and XYZ coordinates.
The CFD dataset provides static topology plus one cell-centred result frame.

The expensive CFD state is prepared once per coupling run:

* static cells/nodes/connectivity are fetched once;
* node->cell incidence is built once;
* all requested CFD variables are fetched in one aligned frame read;
* cell-centred values are projected to all CFD vertices once;
* per-cell barycentric solve data is cached and reused across structure nodes.

No coupling value is written to IoTDB or to the original structural H5 file.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.cfd_nodal_projection import NodeCellCSR, build_node_cell_csr_from_incidence
from cfd_bench.features.fluid_interpolation.engine import LinearSupport
from cfd_bench.infra.iotdb.mesh_runtime import MeshRuntime
from cfd_bench.infra.iotdb.repository import IoTDBRepository

from .alignment import AlignmentDiagnostics, estimate_similarity_alignment
from .output import (
    CouplingH5Writer,
    STATUS_INTERPOLATION_FAILED,
    STATUS_NO_CONTAINING_CELL,
    STATUS_OUTSIDE_MESH,
    STATUS_PASS,
)
from .progress import CouplingProgress


@dataclass(frozen=True)
class CouplingSummary:
    output_path: Path
    structure_dataset: str
    structure_zone: str
    cfd_dataset: str
    cfd_zone: str
    cfd_step: int
    variables: Tuple[str, ...]
    node_count: int
    success_count: int
    outside_count: int
    no_containing_cell_count: int
    failed_count: int
    alignment_enabled: bool = False
    alignment_scale: Optional[float] = None
    alignment_reference_zone: Optional[str] = None
    alignment_rmse: Optional[float] = None
    alignment_confidence: Optional[str] = None


@dataclass(frozen=True)
class _PreparedCell:
    node_ids: np.ndarray
    tetra_local_indices: np.ndarray
    tetra_points: np.ndarray
    origins: np.ndarray
    inverse_matrices: np.ndarray
    condition_numbers: np.ndarray

    def support(self, point: np.ndarray, *, tolerance: float = 1.0e-9) -> Optional[LinearSupport]:
        if self.tetra_local_indices.size == 0:
            return None
        p = np.asarray(point, dtype=np.float64).reshape(3)
        if self.tetra_local_indices.shape[0] == 1:
            first = self.inverse_matrices[0] @ (p - self.origins[0])
            weights = np.empty((4,), dtype=np.float64)
            weights[:3] = first
            weights[3] = 1.0 - float(np.sum(first))
            span = np.ptp(self.tetra_points[0], axis=0)
            scale = max(float(np.max(span)), 1.0)
            coord_tol = max(float(tolerance) * scale, 1.0e-12)
            weight_tol = max(float(tolerance) * 10.0, 1.0e-10)
            if np.any(weights < -weight_tol) or np.any(weights > 1.0 + weight_tol):
                return None
            reconstructed = weights @ self.tetra_points[0]
            error = float(np.linalg.norm(reconstructed - p))
            if not np.isfinite(error) or error > coord_tol * 10.0:
                return None
            return LinearSupport(
                local_indices=tuple(int(x) for x in self.tetra_local_indices[0].tolist()),
                weights=weights,
                reconstruction_error=error,
                condition_number=float(self.condition_numbers[0]),
                margin=float(np.min(weights)),
            )

        rhs = p[None, :] - self.origins
        first = np.einsum("tij,tj->ti", self.inverse_matrices, rhs, optimize=True)
        weights = np.empty((first.shape[0], 4), dtype=np.float64)
        weights[:, :3] = first
        weights[:, 3] = 1.0 - np.sum(first, axis=1)

        span = np.ptp(self.tetra_points.reshape(-1, 3), axis=0)
        scale = max(float(np.max(span)), 1.0)
        coord_tol = max(float(tolerance) * scale, 1.0e-12)
        weight_tol = max(float(tolerance) * 10.0, 1.0e-10)
        valid = np.all(weights >= -weight_tol, axis=1) & np.all(weights <= 1.0 + weight_tol, axis=1)
        if not np.any(valid):
            return None

        idx = np.flatnonzero(valid)
        w = weights[idx]
        tetra = self.tetra_points[idx]
        reconstructed = np.einsum("ti,tij->tj", w, tetra, optimize=True)
        errors = np.linalg.norm(reconstructed - p[None, :], axis=1)
        good = np.isfinite(errors) & (errors <= coord_tol * 10.0)
        if not np.any(good):
            return None
        idx = idx[good]
        w = weights[idx]
        errors = errors[good]
        margins = np.min(w, axis=1)
        conds = self.condition_numbers[idx]
        # max margin, then min condition number, then min reconstruction error.
        order = np.lexsort((errors, conds, -margins))
        best = int(order[0])
        tetra_index = int(idx[best])
        return LinearSupport(
            local_indices=tuple(int(x) for x in self.tetra_local_indices[tetra_index].tolist()),
            weights=np.asarray(w[best], dtype=np.float64),
            reconstruction_error=float(errors[best]),
            condition_number=float(conds[best]),
            margin=float(margins[best]),
        )


class StructureCfdCouplingEngine:
    """Map every node of one IoTDB H5 dataset to one IoTDB CFD frame."""

    def __init__(
        self,
        repo: IoTDBRepository,
        *,
        prepared_cell_cache_size: int = 8192,
    ) -> None:
        self.repo = repo
        self.runtime = MeshRuntime(repo)
        self.prepared_cell_cache_size = max(128, int(prepared_cell_cache_size))
        self._prepared_cells: "OrderedDict[int, Optional[_PreparedCell]]" = OrderedDict()

        self._cfd_dataset = ""
        self._cfd_zone = ""
        self._variables: Tuple[str, ...] = ()
        self._cell_ids = np.zeros((0,), dtype=np.int64)
        self._cell_centroids = np.zeros((0, 3), dtype=np.float64)
        self._bbox_min = np.zeros((0, 3), dtype=np.float64)
        self._bbox_max = np.zeros((0, 3), dtype=np.float64)
        self._global_min = np.zeros((3,), dtype=np.float64)
        self._global_max = np.zeros((3,), dtype=np.float64)
        self._bbox_eps = 1.0e-12
        self._node_coordinates = np.zeros((0, 3), dtype=np.float64)
        self._connectivity_ids = np.zeros((0,), dtype=np.int64)
        self._connectivity = np.zeros((0, 0), dtype=np.int64)
        self._nodal_values = np.zeros((0, 0), dtype=np.float64)
        self._mesh = None
        self._aabb_bucket_keys = np.zeros((0,), dtype=np.int64)
        self._aabb_bucket_offsets = np.zeros((0,), dtype=np.int64)
        self._aabb_bucket_cell_ids = np.zeros((0,), dtype=np.int32)
        self._aabb_oversize_cell_ids = np.zeros((0,), dtype=np.int32)
        self._aabb_index_ready = False

    @staticmethod
    def _resolve_variables(meta: dict, variables: Optional[Sequence[str]]) -> Tuple[str, ...]:
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
    def _validate_step(meta: dict, step: int) -> int:
        value = int(step)
        available = tuple(int(x) for x in meta.get("timesteps", ()))
        if available and value not in available:
            raise ValueError(f"step={value} is not available; available={list(available)}")
        return value

    def _prepare_cfd(
        self,
        dataset: str,
        step: int,
        *,
        variables: Optional[Sequence[str]],
        zone: Optional[str],
        progress: CouplingProgress,
    ) -> Tuple[int, str, Tuple[str, ...]]:
        meta = self.repo.cfd_dataset_metadata(str(dataset))
        if not meta or not bool(meta.get("is_cfd")):
            raise ValueError(f"dataset={dataset!r} is not a CFD dataset in IoTDB")
        resolved_zone = str(zone or meta.get("zone") or "0_Fluid")
        resolved_step = self._validate_step(meta, int(step))
        resolved_variables = self._resolve_variables(meta, variables)

        progress.stage("loading CFD cells and spatial index ...")
        mesh = self.runtime.ensure_cells(str(dataset), resolved_zone)
        self._mesh = mesh
        self._cfd_dataset = str(dataset)
        self._cfd_zone = resolved_zone
        self._variables = resolved_variables
        self._cell_ids = np.asarray(mesh.all_cell_ids, dtype=np.int64)
        self._cell_centroids = np.asarray(mesh.all_centroids, dtype=np.float64)
        self._bbox_min = np.asarray(mesh.all_bbox_min, dtype=np.float64)
        self._bbox_max = np.asarray(mesh.all_bbox_max, dtype=np.float64)
        if self._cell_ids.size == 0:
            raise ValueError(f"CFD dataset={dataset!r} zone={resolved_zone!r} contains no cells")
        if np.any(np.diff(self._cell_ids) < 0):
            raise RuntimeError("CFD cell IDs returned by IoTDB are not sorted")
        self._global_min = np.min(self._bbox_min, axis=0)
        self._global_max = np.max(self._bbox_max, axis=0)
        scale = max(float(np.max(self._global_max - self._global_min)), 1.0)
        self._bbox_eps = max(1.0e-10 * scale, 1.0e-12)
        self._build_aabb_bucket_index(progress)

        progress.stage("loading CFD node coordinates and connectivity ...")
        node_ids, node_coordinates = self.repo.fetch_nodes_arrays(str(dataset), resolved_zone)
        if node_ids.size == 0:
            raise ValueError(f"CFD dataset={dataset!r} zone={resolved_zone!r} contains no nodes")
        node_count = max(int(np.max(node_ids)) + 1, int(meta.get("node_count", 0) or 0))
        dense_coordinates = np.full((node_count, 3), np.nan, dtype=np.float64)
        dense_coordinates[node_ids.astype(np.int64)] = node_coordinates
        self._node_coordinates = dense_coordinates

        conn_ids, connectivity = self.repo.fetch_cell_nodes_arrays(str(dataset), resolved_zone)
        if conn_ids.size == 0:
            raise ValueError(f"CFD dataset={dataset!r} zone={resolved_zone!r} contains no connectivity")
        self._connectivity_ids = np.asarray(conn_ids, dtype=np.int64)
        self._connectivity = np.asarray(connectivity, dtype=np.int64)
        if np.any(np.diff(self._connectivity_ids) < 0):
            raise RuntimeError("CFD connectivity cell IDs returned by IoTDB are not sorted")

        valid = self._connectivity >= 0
        if not np.any(valid):
            raise ValueError("CFD connectivity contains no node references")
        repeated_cells = np.repeat(self._connectivity_ids, self._connectivity.shape[1])
        csr: NodeCellCSR = build_node_cell_csr_from_incidence(
            self._connectivity.reshape(-1)[valid.reshape(-1)],
            repeated_cells[valid.reshape(-1)],
            node_count,
        )

        progress.stage(
            f"loading CFD frame step={resolved_step} variables={list(resolved_variables)} ..."
        )
        cell_values = self.repo.fetch_cell_scalar_matrix(
            str(dataset),
            resolved_step,
            resolved_variables,
            self._cell_ids,
            zone=resolved_zone,
        )
        if cell_values.shape != (self._cell_ids.size, len(resolved_variables)):
            raise RuntimeError(
                f"CFD frame read returned shape={cell_values.shape}, expected="
                f"{(self._cell_ids.size, len(resolved_variables))}"
            )

        progress.stage("projecting cell-centred CFD values to mesh vertices ...")
        edge_cells = np.asarray(csr.cell_ids, dtype=np.int64)
        pos = np.searchsorted(self._cell_ids, edge_cells)
        valid_pos = pos < self._cell_ids.size
        if not np.all(valid_pos):
            raise RuntimeError("node-to-cell incidence references unknown CFD cell IDs")
        if not np.all(self._cell_ids[pos] == edge_cells):
            raise RuntimeError("node-to-cell incidence references unknown CFD cell IDs")
        edge_values = cell_values[pos]
        finite = np.isfinite(edge_values)
        clean = np.where(finite, edge_values, 0.0)

        incidence_counts = np.diff(csr.offsets)
        active_nodes = np.flatnonzero(incidence_counts > 0)
        starts = csr.offsets[active_nodes].astype(np.int64, copy=False)
        nodal = np.full((node_count, len(resolved_variables)), np.nan, dtype=np.float64)
        if starts.size:
            sums = np.add.reduceat(clean, starts, axis=0)
            counts = np.add.reduceat(finite.astype(np.int64), starts, axis=0)
            projected = np.full(sums.shape, np.nan, dtype=np.float64)
            np.divide(sums, counts, out=projected, where=counts > 0)
            nodal[active_nodes] = projected
        self._nodal_values = nodal
        self._prepared_cells.clear()
        return resolved_step, resolved_zone, resolved_variables

    def _build_aabb_bucket_index(self, progress: CouplingProgress) -> None:
        """Build a compact point->AABB candidate index once for coupling.

        The general benchmark runtime indexes cells by centroid and therefore
        needs a rare whole-mesh fallback when a cell spans several buckets.
        Coupling may perform millions of point lookups, so that fallback is too
        costly.  Here every normal cell is assigned to each uniform-grid bucket
        touched by its AABB.  Very large cells are kept in a tiny oversize list.

        A hard incidence cap prevents pathological meshes from trading too much
        memory for speed; those meshes transparently fall back to the existing
        centroid index + exact AABB scan.
        """

        mesh = self._mesh
        if mesh is None or self._cell_ids.size == 0:
            return
        origin = np.asarray(mesh.spatial_origin, dtype=np.float64)
        step = np.asarray(mesh.spatial_step, dtype=np.float64)
        dims = np.asarray(mesh.spatial_dims, dtype=np.int64)
        if np.any(step <= 0) or np.any(dims <= 0):
            return

        lo = np.floor((self._bbox_min - self._bbox_eps - origin[None, :]) / step[None, :]).astype(np.int64)
        hi = np.floor((self._bbox_max + self._bbox_eps - origin[None, :]) / step[None, :]).astype(np.int64)
        lo = np.clip(lo, 0, dims - 1)
        hi = np.clip(hi, 0, dims - 1)
        span = hi - lo + 1
        counts = np.prod(span, axis=1).astype(np.int64)
        max_buckets_per_cell = 512
        normal = counts <= max_buckets_per_cell
        total = int(np.sum(counts[normal], dtype=np.int64))
        max_total_incidences = 12_000_000
        if total <= 0 or total > max_total_incidences:
            progress.stage(
                "AABB bucket index skipped "
                f"(estimated incidences={total:,}; using bounded fallback)"
            )
            return

        progress.stage(f"building coupling AABB bucket index ({total:,} incidences) ...")
        keys = np.empty((total,), dtype=np.int64)
        cids = np.empty((total,), dtype=np.int32)
        oversize = []
        cursor = 0
        dy, dz = int(dims[1]), int(dims[2])
        for row, cid_value in enumerate(self._cell_ids):
            cid = int(cid_value)
            if not normal[row]:
                oversize.append(cid)
                continue
            x0, y0, z0 = (int(v) for v in lo[row])
            x1, y1, z1 = (int(v) for v in hi[row])
            n = int(counts[row])
            if n == 1:
                keys[cursor] = (x0 * dy + y0) * dz + z0
                cids[cursor] = cid
                cursor += 1
                continue
            start = cursor
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    base = (ix * dy + iy) * dz
                    width = z1 - z0 + 1
                    keys[cursor : cursor + width] = np.arange(base + z0, base + z1 + 1, dtype=np.int64)
                    cids[cursor : cursor + width] = cid
                    cursor += width
            if cursor - start != n:
                raise RuntimeError("internal AABB bucket incidence count mismatch")
        if cursor != total:
            keys = keys[:cursor]
            cids = cids[:cursor]
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        cids = cids[order]
        unique_keys, starts = np.unique(keys, return_index=True)
        offsets = np.empty((unique_keys.size + 1,), dtype=np.int64)
        offsets[:-1] = starts
        offsets[-1] = cids.size
        self._aabb_bucket_keys = unique_keys.astype(np.int64, copy=False)
        self._aabb_bucket_offsets = offsets
        self._aabb_bucket_cell_ids = cids.astype(np.int32, copy=False)
        self._aabb_oversize_cell_ids = np.asarray(oversize, dtype=np.int32)
        self._aabb_index_ready = True

    def _connectivity_for_cell(self, cell_id: int) -> np.ndarray:
        pos = int(np.searchsorted(self._connectivity_ids, int(cell_id)))
        if pos >= self._connectivity_ids.size or int(self._connectivity_ids[pos]) != int(cell_id):
            return np.zeros((0,), dtype=np.int64)
        row = self._connectivity[pos]
        return row[row >= 0].astype(np.int64, copy=False)

    def _prepare_cell(self, cell_id: int) -> Optional[_PreparedCell]:
        cid = int(cell_id)
        if cid in self._prepared_cells:
            cached = self._prepared_cells.pop(cid)
            self._prepared_cells[cid] = cached
            return cached

        node_ids = self._connectivity_for_cell(cid)
        prepared: Optional[_PreparedCell] = None
        if node_ids.size >= 4 and np.all(node_ids < self._node_coordinates.shape[0]):
            points = self._node_coordinates[node_ids]
            if np.all(np.isfinite(points)):
                combos = []
                tetra_points = []
                origins = []
                inverses = []
                conds = []
                for combo in combinations(range(node_ids.size), 4):
                    tetra = points[np.asarray(combo, dtype=np.int64)]
                    matrix = np.column_stack(
                        (tetra[0] - tetra[3], tetra[1] - tetra[3], tetra[2] - tetra[3])
                    )
                    try:
                        cond = float(np.linalg.cond(matrix))
                        if not np.isfinite(cond) or cond > 1.0e14:
                            continue
                        inv = np.linalg.inv(matrix)
                    except np.linalg.LinAlgError:
                        continue
                    combos.append(combo)
                    tetra_points.append(tetra)
                    origins.append(tetra[3])
                    inverses.append(inv)
                    conds.append(cond)
                if combos:
                    prepared = _PreparedCell(
                        node_ids=np.asarray(node_ids, dtype=np.int64),
                        tetra_local_indices=np.asarray(combos, dtype=np.int16),
                        tetra_points=np.asarray(tetra_points, dtype=np.float64),
                        origins=np.asarray(origins, dtype=np.float64),
                        inverse_matrices=np.asarray(inverses, dtype=np.float64),
                        condition_numbers=np.asarray(conds, dtype=np.float64),
                    )

        self._prepared_cells[cid] = prepared
        while len(self._prepared_cells) > self.prepared_cell_cache_size:
            self._prepared_cells.popitem(last=False)
        return prepared

    def _bucket_candidate_ids(self, point: np.ndarray) -> np.ndarray:
        mesh = self._mesh
        if mesh is None:
            return np.zeros((0,), dtype=np.int64)
        ox, oy, oz = mesh.spatial_origin
        sx, sy, sz = mesh.spatial_step
        dx, dy, dz = mesh.spatial_dims
        bx = int(np.floor((float(point[0]) - ox) / sx)) if sx > 0 else 0
        by = int(np.floor((float(point[1]) - oy) / sy)) if sy > 0 else 0
        bz = int(np.floor((float(point[2]) - oz) / sz)) if sz > 0 else 0
        bx = min(max(bx, 0), max(dx - 1, 0))
        by = min(max(by, 0), max(dy - 1, 0))
        bz = min(max(bz, 0), max(dz - 1, 0))

        if self._aabb_index_ready:
            key = (int(bx) * int(dy) + int(by)) * int(dz) + int(bz)
            pos = int(np.searchsorted(self._aabb_bucket_keys, key))
            chunks = []
            if pos < self._aabb_bucket_keys.size and int(self._aabb_bucket_keys[pos]) == key:
                a = int(self._aabb_bucket_offsets[pos])
                b = int(self._aabb_bucket_offsets[pos + 1])
                if b > a:
                    chunks.append(self._aabb_bucket_cell_ids[a:b])
            if self._aabb_oversize_cell_ids.size:
                chunks.append(self._aabb_oversize_cell_ids)
            if not chunks:
                return np.zeros((0,), dtype=np.int64)
            raw = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            return np.asarray(raw, dtype=np.int64)

        chunks = []
        keys = mesh.spatial_bucket_keys
        offsets = mesh.spatial_bucket_offsets
        bucket_cids = mesh.spatial_bucket_cell_ids
        if keys.size and offsets.size:
            for ix in range(max(0, bx - 1), min(dx, bx + 2)):
                for iy in range(max(0, by - 1), min(dy, by + 2)):
                    for iz in range(max(0, bz - 1), min(dz, bz + 2)):
                        key = (int(ix) * int(dy) + int(iy)) * int(dz) + int(iz)
                        kpos = int(np.searchsorted(keys, key))
                        if kpos < keys.size and int(keys[kpos]) == key:
                            a, b = int(offsets[kpos]), int(offsets[kpos + 1])
                            if b > a:
                                chunks.append(bucket_cids[a:b])
        elif mesh.spatial_buckets:
            for ix in range(max(0, bx - 1), min(dx, bx + 2)):
                for iy in range(max(0, by - 1), min(dy, by + 2)):
                    for iz in range(max(0, bz - 1), min(dz, bz + 2)):
                        vals = mesh.spatial_buckets.get((ix, iy, iz), ())
                        if vals:
                            chunks.append(np.asarray(vals, dtype=np.int32))
        if not chunks:
            return np.zeros((0,), dtype=np.int64)
        raw = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        return np.unique(np.asarray(raw, dtype=np.int64))

    def _candidate_cells(self, point: np.ndarray) -> Tuple[np.ndarray, bool]:
        p = np.asarray(point, dtype=np.float64).reshape(3)
        eps = self._bbox_eps
        if np.any(p < self._global_min - eps) or np.any(p > self._global_max + eps):
            return np.zeros((0,), dtype=np.int64), True

        local = self._bucket_candidate_ids(p)
        hits = np.zeros((0,), dtype=np.int64)
        if local.size:
            pos = np.searchsorted(self._cell_ids, local)
            valid = pos < self._cell_ids.size
            if np.any(valid):
                local = local[valid]
                pos = pos[valid]
                exact = self._cell_ids[pos] == local
                local = local[exact]
                pos = pos[exact]
                if local.size:
                    inside = np.all(self._bbox_min[pos] - eps <= p, axis=1) & np.all(
                        self._bbox_max[pos] + eps >= p, axis=1
                    )
                    hits = local[inside]

        if hits.size == 0 and not self._aabb_index_ready:
            # Rare correctness fallback for a cell whose centroid is farther
            # than one uniform-grid bucket from an interior point.
            mask = np.all(self._bbox_min - eps <= p, axis=1) & np.all(
                self._bbox_max + eps >= p, axis=1
            )
            hits = self._cell_ids[mask]
        if hits.size > 1:
            pos = np.searchsorted(self._cell_ids, hits)
            distances = np.linalg.norm(self._cell_centroids[pos] - p[None, :], axis=1)
            hits = hits[np.argsort(distances, kind="stable")]
        return hits, False

    def _locate_support(
        self, point: np.ndarray
    ) -> Tuple[int, Optional[LinearSupport], Optional[_PreparedCell], int]:
        candidate_ids, globally_outside = self._candidate_cells(point)
        if candidate_ids.size == 0:
            return -1, None, None, STATUS_OUTSIDE_MESH if globally_outside else STATUS_OUTSIDE_MESH

        best = None
        best_key = None
        for cid in candidate_ids.tolist():
            prepared = self._prepare_cell(int(cid))
            if prepared is None:
                continue
            support = prepared.support(point)
            if support is None:
                continue
            key = (support.margin, -support.condition_number, -int(cid))
            if best_key is None or key > best_key:
                best_key = key
                best = (int(cid), support, prepared)
        if best is None:
            return -1, None, None, STATUS_NO_CONTAINING_CELL
        return best[0], best[1], best[2], STATUS_PASS

    def _map_batch(self, points: np.ndarray, *, diagnostics: bool) -> Dict[str, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        count = pts.shape[0]
        values = np.full((count, len(self._variables)), np.nan, dtype=np.float64)
        status = np.full((count,), STATUS_INTERPOLATION_FAILED, dtype=np.uint8)
        cell_ids = np.full((count,), -1, dtype=np.int64)
        errors = np.full((count,), np.nan, dtype=np.float64)
        support_ids = np.full((count, 4), -1, dtype=np.int64) if diagnostics else None
        weights_out = np.full((count, 4), np.nan, dtype=np.float64) if diagnostics else None

        for i, point in enumerate(pts):
            if not np.all(np.isfinite(point)):
                continue
            cid, support, prepared, locate_status = self._locate_support(point)
            if locate_status != STATUS_PASS or support is None or prepared is None:
                status[i] = np.uint8(locate_status)
                continue

            local = np.asarray(support.local_indices, dtype=np.int64)
            node_ids = prepared.node_ids[local]
            node_values = self._nodal_values[node_ids]
            if node_values.shape != (4, len(self._variables)) or not np.all(np.isfinite(node_values)):
                status[i] = np.uint8(STATUS_INTERPOLATION_FAILED)
                cell_ids[i] = int(cid)
                errors[i] = float(support.reconstruction_error)
                continue

            mapped = np.asarray(support.weights, dtype=np.float64) @ node_values
            if not np.all(np.isfinite(mapped)):
                status[i] = np.uint8(STATUS_INTERPOLATION_FAILED)
                cell_ids[i] = int(cid)
                errors[i] = float(support.reconstruction_error)
                continue

            values[i] = mapped
            status[i] = np.uint8(STATUS_PASS)
            cell_ids[i] = int(cid)
            errors[i] = float(support.reconstruction_error)
            if diagnostics:
                support_ids[i] = node_ids
                weights_out[i] = support.weights

        result = {
            "values": values,
            "status": status,
            "cfd_cell_ids": cell_ids,
            "reconstruction_error": errors,
        }
        if diagnostics:
            result["support_node_ids"] = support_ids
            result["weights"] = weights_out
        return result

    @staticmethod
    def _resolve_alignment_zone(
        cfd_meta: dict,
        resolved_cfd_zone: str,
        requested_zone: Optional[str],
    ) -> str:
        if requested_zone:
            return str(requested_zone)
        hull_zones = [
            str(zone)
            for zone in cfd_meta.get("zones", ())
            if "hull" in str(zone).lower()
        ]
        if hull_zones:
            return hull_zones[0]
        raise ValueError(
            "auto-alignment requires a CFD hull/reference surface zone. "
            "No ingested zone containing 'hull' was found; ingest the hull zone "
            "or pass --alignment-cfd-zone explicitly. The full fluid-volume zone "
            f"{resolved_cfd_zone!r} is intentionally not used for automatic alignment."
        )

    def _estimate_structure_alignment(
        self,
        structure_coordinates: np.ndarray,
        *,
        cfd_dataset: str,
        resolved_cfd_zone: str,
        requested_alignment_zone: Optional[str],
        max_points: int,
        max_iterations: int,
        trim_fraction: float,
        progress: CouplingProgress,
    ) -> Tuple[AlignmentDiagnostics, str]:
        cfd_meta = self.repo.cfd_dataset_metadata(str(cfd_dataset))
        alignment_zone = self._resolve_alignment_zone(
            cfd_meta, resolved_cfd_zone, requested_alignment_zone
        )

        progress.stage(
            f"estimating optional structure/CFD similarity alignment using CFD zone={alignment_zone!r} ..."
        )
        if alignment_zone == resolved_cfd_zone and self._node_coordinates.size:
            cfd_reference = self._node_coordinates
        else:
            _ids, cfd_reference = self.repo.fetch_nodes_arrays(str(cfd_dataset), alignment_zone)
        cfd_reference = np.asarray(cfd_reference, dtype=np.float64).reshape(-1, 3)
        if cfd_reference.shape[0] < 4:
            raise ValueError(
                f"alignment CFD zone={alignment_zone!r} contains fewer than 4 nodes; "
                "ingest the hull/reference zone or specify another --alignment-cfd-zone"
            )
        result = estimate_similarity_alignment(
            np.asarray(structure_coordinates, dtype=np.float64),
            cfd_reference,
            max_points=int(max_points),
            max_iterations=int(max_iterations),
            trim_fraction=float(trim_fraction),
        )
        progress.stage(
            "alignment: "
            f"scale={result.transform.scale:.12g} "
            f"rmse={result.rmse_after:.6g} p95={result.p95_error:.6g} "
            f"confidence={result.confidence}"
        )
        if result.confidence == "low":
            progress.stage(
                "warning: automatic alignment confidence is LOW; inspect the saved transform/coordinates "
                "before using the coupled values for analysis."
            )
        return result, alignment_zone

    def couple_to_h5(
        self,
        *,
        structure_dataset: str,
        cfd_dataset: str,
        cfd_step: int,
        output_path: str | Path,
        variables: Optional[Sequence[str]] = None,
        structure_zone: Optional[str] = None,
        cfd_zone: Optional[str] = None,
        batch_size: int = 4096,
        diagnostics: bool = False,
        progress: bool = True,
        progress_interval: float = 0.25,
        auto_align: bool = False,
        alignment_cfd_zone: Optional[str] = None,
        alignment_max_points: int = 10000,
        alignment_max_iterations: int = 30,
        alignment_trim_fraction: float = 0.80,
    ) -> CouplingSummary:
        batch_size = max(1, int(batch_size))
        reporter = CouplingProgress(enabled=progress, min_interval=progress_interval)
        writer = None
        try:
            h5_meta = self.repo.h5_dataset_metadata(str(structure_dataset))
            if not h5_meta or not bool(h5_meta.get("is_h5")):
                raise ValueError(f"dataset={structure_dataset!r} is not an H5 structural dataset in IoTDB")
            resolved_structure_zone = str(structure_zone or h5_meta.get("zone") or "0_Fluid")

            reporter.stage("loading structural node coordinates from IoTDB ...")
            node_ids, source_labels, coordinates = self.repo.fetch_h5_structure_nodes(
                str(structure_dataset), resolved_structure_zone
            )
            if node_ids.size == 0:
                raise ValueError(
                    f"structural dataset={structure_dataset!r} zone={resolved_structure_zone!r} contains no nodes"
                )
            if coordinates.shape != (node_ids.size, 3) or source_labels.shape != node_ids.shape:
                raise RuntimeError("structural node arrays returned by IoTDB are inconsistent")

            resolved_step, resolved_cfd_zone, resolved_variables = self._prepare_cfd(
                str(cfd_dataset),
                int(cfd_step),
                variables=variables,
                zone=cfd_zone,
                progress=reporter,
            )

            alignment = None
            resolved_alignment_zone = None
            if bool(auto_align):
                alignment, resolved_alignment_zone = self._estimate_structure_alignment(
                    coordinates,
                    cfd_dataset=str(cfd_dataset),
                    resolved_cfd_zone=resolved_cfd_zone,
                    requested_alignment_zone=alignment_cfd_zone,
                    max_points=int(alignment_max_points),
                    max_iterations=int(alignment_max_iterations),
                    trim_fraction=float(alignment_trim_fraction),
                    progress=reporter,
                )

            alignment_metadata = None
            if alignment is not None:
                alignment_metadata = {
                    "method": alignment.method,
                    "reference_zone": str(resolved_alignment_zone),
                    "scale": float(alignment.transform.scale),
                    "rotation": np.asarray(alignment.transform.rotation, dtype=np.float64),
                    "translation": np.asarray(alignment.transform.translation, dtype=np.float64),
                    "initial_scale": float(alignment.initial_scale),
                    "rms_scale": float(alignment.rms_scale),
                    "principal_scale": float(alignment.principal_scale),
                    "principal_extent_scale": float(alignment.principal_extent_scale),
                    "scale_consistency": float(alignment.scale_consistency),
                    "rmse_before": float(alignment.rmse_before),
                    "rmse_after": float(alignment.rmse_after),
                    "median_error": float(alignment.median_error),
                    "p95_error": float(alignment.p95_error),
                    "inlier_fraction": float(alignment.inlier_fraction),
                    "iterations": int(alignment.iterations),
                    "structure_sample_count": int(alignment.structure_sample_count),
                    "cfd_sample_count": int(alignment.cfd_sample_count),
                    "confidence": str(alignment.confidence),
                }

            reporter.stage(f"creating independent coupling result: {Path(output_path)}")
            writer = CouplingH5Writer(
                output_path,
                structure_dataset=str(structure_dataset),
                structure_zone=resolved_structure_zone,
                cfd_dataset=str(cfd_dataset),
                cfd_zone=resolved_cfd_zone,
                cfd_step=resolved_step,
                variables=resolved_variables,
                node_ids=node_ids,
                source_node_labels=source_labels,
                coordinates=coordinates,
                batch_size=batch_size,
                diagnostics=diagnostics,
                alignment_metadata=alignment_metadata,
                store_coupling_coordinates=alignment is not None,
            )

            counts = {
                STATUS_PASS: 0,
                STATUS_OUTSIDE_MESH: 0,
                STATUS_NO_CONTAINING_CELL: 0,
                STATUS_INTERPOLATION_FAILED: 0,
            }
            total = int(node_ids.size)
            reporter.update(0, total)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                source_points = coordinates[start:end]
                coupling_points = (
                    alignment.transform.apply(source_points) if alignment is not None else source_points
                )
                batch = self._map_batch(coupling_points, diagnostics=diagnostics)
                writer.write_batch(
                    start,
                    end,
                    values=batch["values"],
                    status=batch["status"],
                    cfd_cell_ids=batch["cfd_cell_ids"],
                    reconstruction_error=batch["reconstruction_error"],
                    coupling_coordinates=coupling_points if alignment is not None else None,
                    support_node_ids=batch.get("support_node_ids"),
                    weights=batch.get("weights"),
                )
                unique, batch_counts = np.unique(batch["status"], return_counts=True)
                for code, n in zip(unique.tolist(), batch_counts.tolist()):
                    counts[int(code)] = counts.get(int(code), 0) + int(n)
                reporter.update(end, total)

            summary_attrs = {
                "node_count": total,
                "success_count": counts[STATUS_PASS],
                "outside_count": counts[STATUS_OUTSIDE_MESH],
                "no_containing_cell_count": counts[STATUS_NO_CONTAINING_CELL],
                "failed_count": counts[STATUS_INTERPOLATION_FAILED],
            }
            final_path = writer.finalize(summary_attrs)
            writer = None
            reporter.stage(
                "completed: "
                f"PASS={summary_attrs['success_count']} "
                f"OUTSIDE={summary_attrs['outside_count']} "
                f"NO_CELL={summary_attrs['no_containing_cell_count']} "
                f"FAILED={summary_attrs['failed_count']}"
            )
            return CouplingSummary(
                output_path=final_path,
                structure_dataset=str(structure_dataset),
                structure_zone=resolved_structure_zone,
                cfd_dataset=str(cfd_dataset),
                cfd_zone=resolved_cfd_zone,
                cfd_step=resolved_step,
                variables=resolved_variables,
                node_count=total,
                success_count=summary_attrs["success_count"],
                outside_count=summary_attrs["outside_count"],
                no_containing_cell_count=summary_attrs["no_containing_cell_count"],
                failed_count=summary_attrs["failed_count"],
                alignment_enabled=alignment is not None,
                alignment_scale=(float(alignment.transform.scale) if alignment is not None else None),
                alignment_reference_zone=resolved_alignment_zone,
                alignment_rmse=(float(alignment.rmse_after) if alignment is not None else None),
                alignment_confidence=(str(alignment.confidence) if alignment is not None else None),
            )
        except Exception:
            if writer is not None:
                writer.abort()
            raise
        finally:
            reporter.close()
