from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tiledb

from .config import TileDBConfig


class TileDBRepository:
    def __init__(self, config: Optional[TileDBConfig] = None, ctx: Optional[tiledb.Ctx] = None):
        self.config = config or TileDBConfig()
        self.ctx = ctx or tiledb.Ctx()

    # -------------------- path helpers --------------------
    def _base(self, dataset_key: str) -> str:
        return os.path.join(self.config.root_path, dataset_key)

    def path_mesh_static(self, dataset_key: str, zone: str, leaf: str) -> str:
        return os.path.join(self._base(dataset_key), "mesh_static", zone, f"{leaf}.tdb")

    def path_cell_vars(self, dataset_key: str, step: int, zone: str = "0_Fluid") -> str:
        if zone in ("1_Hull", "hull"):
            return os.path.join(
                self._base(dataset_key), "post_processing", f"step_{int(step)}", "cell_vars_hull.tdb"
            )
        return os.path.join(self._base(dataset_key), "post_processing", f"step_{int(step)}", "cell_vars.tdb")

    def path_node_vars(self, dataset_key: str, step: int) -> str:
        return os.path.join(self._base(dataset_key), "post_processing", f"step_{int(step)}", "node_vars.tdb")

    def path_derived(self, dataset_key: str, step: int, leaf: str) -> str:
        return os.path.join(self._base(dataset_key), "derived", f"step_{int(step)}", f"{leaf}.tdb")

    def array_exists(self, uri: str) -> bool:
        return tiledb.array_exists(uri, ctx=self.ctx)

    def open_array(self, uri: str, mode: str = "r"):
        if not self.array_exists(uri):
            raise FileNotFoundError(uri)
        return tiledb.open(uri, mode=mode, ctx=self.ctx)

    def probe_array(self, uri: str) -> bool:
        try:
            if not self.array_exists(uri):
                return False
            with self.open_array(uri, "r") as A:
                _ = A.nonempty_domain()
            return True
        except Exception:
            return False

    # -------------------- mesh static --------------------
    def fetch_mesh_meta(self, dataset_key: str, zone: str) -> Dict[str, float]:
        uri = self.path_mesh_static(dataset_key, zone, "mesh_meta")
        with self.open_array(uri, "r") as A:
            data = A[0]
        keys = ["node_count", "cell_count", "face_count",
                "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y", "bbox_min_z", "bbox_max_z"]
        return {k: float(data[k]) for k in keys}

    def _read_dense_chunked(self, uri: str, start: int, end: int) -> dict:
        with self.open_array(uri, "r") as A:
            return A[start:end]

    def fetch_cells(self, dataset_key: str, zone: str) -> Dict[int, Tuple[float, ...]]:
        uri = self.path_mesh_static(dataset_key, zone, "cells")
        out: Dict[int, Tuple[float, ...]] = {}
        cell_count = None
        try:
            meta = self.fetch_mesh_meta(dataset_key, zone)
            cell_count = int(meta.get("cell_count", 0))
        except Exception:
            cell_count = None

        if cell_count is None or cell_count <= 0:
            with self.open_array(uri, "r") as A:
                data = A[:]
            n = len(data["cx"])
            for i in range(n):
                out[i] = tuple(float(data[k][i]) for k in (
                    "cx", "cy", "cz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cell_type"
                ))
            return out

        chunk = self.config.read_chunk
        for start in range(0, cell_count, chunk):
            end = min(start + chunk, cell_count)
            data = self._read_dense_chunked(uri, start, end)
            for j in range(end - start):
                cid = start + j
                out[cid] = tuple(float(data[k][j]) for k in (
                    "cx", "cy", "cz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cell_type"
                ))
        return out

    def fetch_nodes(self, dataset_key: str, zone: str) -> Dict[int, Tuple[float, float, float]]:
        uri = self.path_mesh_static(dataset_key, zone, "nodes")
        out: Dict[int, Tuple[float, float, float]] = {}
        try:
            meta = self.fetch_mesh_meta(dataset_key, zone)
            node_count = int(meta.get("node_count", 0))
        except Exception:
            node_count = 0

        if node_count <= 0:
            with self.open_array(uri, "r") as A:
                data = A[:]
            for i in range(len(data["x"])):
                out[i] = (float(data["x"][i]), float(data["y"][i]), float(data["z"][i]))
            return out

        chunk = self.config.read_chunk
        for start in range(0, node_count, chunk):
            end = min(start + chunk, node_count)
            data = self._read_dense_chunked(uri, start, end)
            for j in range(end - start):
                nid = start + j
                out[nid] = (float(data["x"][j]), float(data["y"][j]), float(data["z"][j]))
        return out

    def fetch_cell_nodes(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        uri = self.path_mesh_static(dataset_key, zone, "cell_nodes")
        out: Dict[int, List[int]] = {}
        node_cols = [f"node_id_{i}" for i in range(16)]

        try:
            meta = self.fetch_mesh_meta(dataset_key, zone)
            cell_count = int(meta.get("cell_count", 0))
        except Exception:
            cell_count = 0

        def _parse_chunk(data, offset: int, count: int):
            for j in range(count):
                cid = offset + j
                ids = [int(data[col][j]) for col in node_cols if int(data[col][j]) >= 0]
                out[cid] = ids

        if cell_count <= 0:
            with self.open_array(uri, "r") as A:
                data = A[:]
            _parse_chunk(data, 0, len(data[node_cols[0]]))
            return out

        chunk = self.config.read_chunk
        for start in range(0, cell_count, chunk):
            end = min(start + chunk, cell_count)
            data = self._read_dense_chunked(uri, start, end)
            _parse_chunk(data, start, end - start)
        return out

    def fetch_cell_adjacency(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        uri = self.path_mesh_static(dataset_key, zone, "cell_adjacency")
        out: Dict[int, List[int]] = {}
        adj_cols = [f"neighbor_id_{i}" for i in range(16)]

        try:
            meta = self.fetch_mesh_meta(dataset_key, zone)
            cell_count = int(meta.get("cell_count", 0))
        except Exception:
            cell_count = 0

        def _parse_chunk(data, offset: int, count: int):
            for j in range(count):
                cid = offset + j
                ids = [int(data[col][j]) for col in adj_cols if int(data[col][j]) >= 0]
                out[cid] = ids

        if cell_count <= 0:
            with self.open_array(uri, "r") as A:
                data = A[:]
            _parse_chunk(data, 0, len(data[adj_cols[0]]))
            return out

        chunk = self.config.read_chunk
        for start in range(0, cell_count, chunk):
            end = min(start + chunk, cell_count)
            data = self._read_dense_chunked(uri, start, end)
            _parse_chunk(data, start, end - start)
        return out

    def fetch_face_planes(self, dataset_key: str, zone: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
        uri = self.path_mesh_static(dataset_key, zone, "face_planes")
        out: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        if not self.array_exists(uri):
            return out
        with self.open_array(uri, "r") as A:
            data = A[:]
        n = len(data["cell_id"])
        for j in range(n):
            cid = int(data["cell_id"][j])
            out.setdefault(cid, []).append((
                int(data["neighbor_id"][j]),
                float(data["nx"][j]), float(data["ny"][j]), float(data["nz"][j]), float(data["d"][j]),
            ))
        return out

    def fetch_boundary_faces(
        self, dataset_key: str, zone: str, patch_name: str = "*"
    ) -> List[Tuple[int, float, float, float, float]]:
        uri = self.path_mesh_static(dataset_key, zone, "boundary_faces")
        if not self.array_exists(uri):
            return []
        rows: List[Tuple[int, float, float, float, float]] = []
        with self.open_array(uri, "r") as A:
            data = A[:]
        n = len(data["cell_id"])
        for j in range(n):
            rows.append((
                int(data["cell_id"][j]),
                float(data["nx"][j]), float(data["ny"][j]), float(data["nz"][j]), float(data["area"][j]),
            ))
        return rows

    # -------------------- scalar queries --------------------
    def _resolve_cell_vars_uri(self, dataset_key: str, step: int, zone: str) -> str:
        uri = self.path_cell_vars(dataset_key, step, zone)
        if self.array_exists(uri):
            return uri
        raise FileNotFoundError(f"cell vars not found for {dataset_key} step={step} zone={zone}")

    def fetch_cell_scalar_map(
        self, dataset_key: str, step: int, var: str, cell_ids: Sequence[int], zone: str = "0_Fluid"
    ) -> Dict[int, float]:
        norm_ids = [int(i) for i in cell_ids]
        if not norm_ids:
            return {}
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            data = A[:]
            arr = np.asarray(data[var], dtype=np.float64)
            return {int(cid): float(arr[cid]) for cid in norm_ids}

    def fetch_all_cell_scalars(
        self, dataset_key: str, step: int, var: str, zone: str = "0_Fluid"
    ) -> np.ndarray:
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            return np.asarray(A[:][var], dtype=np.float64)

    def fetch_cell_ids_by_var_range(
        self, dataset_key: str, step: int, var: str, lower: float, upper: float, zone: str = "0_Fluid"
    ) -> List[int]:
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            data = A[:]
            index_arr = np.arange(len(data[var]), dtype=np.int32)
            var_arr = np.asarray(data[var], dtype=np.float64)
        mask = (var_arr >= float(lower)) & (var_arr <= float(upper))
        return [int(i) for i in index_arr[mask]]

    def fetch_velocity_map(
        self, dataset_key: str, step: int, cell_ids: Sequence[int], zone: str = "0_Fluid"
    ) -> Dict[int, Tuple[float, float, float]]:
        norm_ids = [int(i) for i in cell_ids]
        if not norm_ids:
            return {}
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            data = A[:]
            return {
                int(cid): (float(data["U"][cid]), float(data["V"][cid]), float(data["W"][cid]))
                for cid in norm_ids
            }

    def fetch_qcriterion_by_roi(
        self,
        dataset_key: str,
        step: int,
        tau: float,
        roi_bounds: Sequence[float],
        zone: str = "0_Fluid",
    ) -> List[Tuple[int, float]]:
        x0, x1, y0, y1, z0, z1 = map(float, roi_bounds)
        qc_uri = self.path_derived(dataset_key, step, "cell_qcriterion")
        cells_uri = self.path_mesh_static(dataset_key, zone, "cells")
        if not self.array_exists(qc_uri):
            return []
        cells = self.fetch_cells(dataset_key, zone)
        roi_ids = [
            cid for cid, v in cells.items()
            if v[3] >= x0 and v[4] <= x1 and v[5] >= y0 and v[6] <= y1 and v[7] >= z0 and v[8] <= z1
        ]
        if not roi_ids:
            return []
        with self.open_array(qc_uri, "r") as A:
            data = A[:]
            q_arr = np.asarray(data["q"], dtype=np.float64)
        out = []
        for cid in roi_ids:
            qf = float(q_arr[cid])
            if qf >= float(tau):
                out.append((int(cid), qf))
        out.sort(key=lambda x: -x[1])
        return out

    def list_mesh_static_zones(self, dataset_key: str):
        """
        List available mesh_static zones for a dataset.

        Example:
            mesh_static/
                0_Fluid
                0_Symmetry_sym
                1_Hull
        """

        import os

        root = os.path.join(
            self.config.root_path,
            dataset_key,
            "mesh_static",
        )

        if not os.path.exists(root):
            return []

        zones = []

        for name in os.listdir(root):
            path = os.path.join(root, name)

            if not os.path.isdir(path):
                continue

            # must contain cells.tdb
            if os.path.exists(
                    os.path.join(path, "cells.tdb")
            ):
                zones.append(name)

        return sorted(zones)
