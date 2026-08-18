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
        self._h5_meta_cache: Dict[str, Dict[str, object]] = {}
        self._h5_steps_cache: Dict[str, List[int]] = {}
        self._node_source_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._cell_source_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._var_range_cache: Dict[Tuple[str, int, str, str], Tuple[float, float]] = {}

    # -------------------- path helpers --------------------
    def _base(self, dataset_key: str) -> str:
        return os.path.join(self.config.root_path, dataset_key)

    def path_mesh_static(self, dataset_key: str, zone: str, leaf: str) -> str:
        return os.path.join(self._base(dataset_key), "mesh_static", zone, f"{leaf}.tdb")

    def path_cell_vars(self, dataset_key: str, step: int, zone: str = "0_Fluid") -> str:
        z = str(zone or "0_Fluid").strip().lower()
        if "hull" in z or "wall" in z or z in ("1_hull", "hull"):
            return os.path.join(
                self._base(dataset_key), "post_processing", f"step_{int(step)}", "cell_vars_hull.tdb"
            )
        return os.path.join(self._base(dataset_key), "post_processing", f"step_{int(step)}", "cell_vars.tdb")

    def path_node_vars(self, dataset_key: str, step: int) -> str:
        return os.path.join(self._base(dataset_key), "post_processing", f"step_{int(step)}", "node_vars.tdb")

    def path_derived(self, dataset_key: str, step: int, leaf: str) -> str:
        return os.path.join(self._base(dataset_key), "derived", f"step_{int(step)}", f"{leaf}.tdb")

    def path_h5_metadata(self, dataset_key: str, leaf: str = "dataset_meta") -> str:
        return os.path.join(self._base(dataset_key), "h5_metadata", f"{leaf}.tdb")

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


    def fetch_cells_arrays(self, dataset_key: str, zone: str):
        """Return compact NumPy cell geometry arrays for runtime hot paths."""
        uri = self.path_mesh_static(dataset_key, zone, "cells")
        attrs = ["cx", "cy", "cz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cell_type"]
        try:
            meta = self.fetch_mesh_meta(dataset_key, zone)
            cell_count = int(meta.get("cell_count", 0))
        except Exception:
            cell_count = 0

        if cell_count <= 0:
            with self.open_array(uri, "r") as A:
                try:
                    data = A.query(attrs=attrs)[:]
                except Exception:
                    data = A[:]
            n = len(np.asarray(data["cx"]).reshape(-1))
            ids = np.arange(n, dtype=np.int32)
            centers = np.column_stack([data["cx"], data["cy"], data["cz"]]).astype(np.float64, copy=False)
            mins = np.column_stack([data["xmin"], data["ymin"], data["zmin"]]).astype(np.float64, copy=False)
            maxs = np.column_stack([data["xmax"], data["ymax"], data["zmax"]]).astype(np.float64, copy=False)
            types = np.asarray(data["cell_type"], dtype=np.int32).reshape(-1)
            return ids, centers, mins, maxs, types

        ids = np.arange(cell_count, dtype=np.int32)
        centers = np.empty((cell_count, 3), dtype=np.float64)
        mins = np.empty((cell_count, 3), dtype=np.float64)
        maxs = np.empty((cell_count, 3), dtype=np.float64)
        types = np.empty(cell_count, dtype=np.int32)
        chunk = self.config.read_chunk
        with self.open_array(uri, "r") as A:
            for start in range(0, cell_count, chunk):
                end = min(start + chunk, cell_count)
                try:
                    data = A.query(attrs=attrs)[start:end]
                except Exception:
                    data = A[start:end]
                centers[start:end, 0] = np.asarray(data["cx"], dtype=np.float64).reshape(-1)
                centers[start:end, 1] = np.asarray(data["cy"], dtype=np.float64).reshape(-1)
                centers[start:end, 2] = np.asarray(data["cz"], dtype=np.float64).reshape(-1)
                mins[start:end, 0] = np.asarray(data["xmin"], dtype=np.float64).reshape(-1)
                mins[start:end, 1] = np.asarray(data["ymin"], dtype=np.float64).reshape(-1)
                mins[start:end, 2] = np.asarray(data["zmin"], dtype=np.float64).reshape(-1)
                maxs[start:end, 0] = np.asarray(data["xmax"], dtype=np.float64).reshape(-1)
                maxs[start:end, 1] = np.asarray(data["ymax"], dtype=np.float64).reshape(-1)
                maxs[start:end, 2] = np.asarray(data["zmax"], dtype=np.float64).reshape(-1)
                types[start:end] = np.asarray(data["cell_type"], dtype=np.int32).reshape(-1)
        return ids, centers, mins, maxs, types

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

    def fetch_nodes_subset(
        self, dataset_key: str, zone: str, node_ids: Sequence[int]
    ) -> Dict[int, Tuple[float, float, float]]:
        ids = sorted(set(int(x) for x in node_ids if int(x) >= 0))
        if not ids:
            return {}
        uri = self.path_mesh_static(dataset_key, zone, "nodes")
        raw = self._read_dense_attrs_at_ids(uri, ["x", "y", "z"], ids)
        return {
            nid: (float(raw["x"][nid]), float(raw["y"][nid]), float(raw["z"][nid]))
            for nid in ids
            if nid in raw["x"] and nid in raw["y"] and nid in raw["z"]
        }

    def fetch_cell_nodes_subset(
        self, dataset_key: str, zone: str, cell_ids: Sequence[int]
    ) -> Dict[int, List[int]]:
        ids = sorted(set(int(x) for x in cell_ids if int(x) >= 0))
        if not ids:
            return {}
        uri = self.path_mesh_static(dataset_key, zone, "cell_nodes")
        cols = [f"node_id_{i}" for i in range(16)]
        raw = self._read_dense_attrs_at_ids(uri, cols, ids)
        out = {}
        for cid in ids:
            vals = [int(raw[col][cid]) for col in cols if cid in raw[col] and int(raw[col][cid]) >= 0]
            out[cid] = vals
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

    def _read_dense_attrs_at_ids(
        self, uri: str, attrs: Sequence[str], cell_ids: Sequence[int]
    ) -> Dict[str, Dict[int, object]]:
        """Read only requested dense-array rows, coalescing contiguous ids.

        The historical implementation used ``A[:]`` even for a single cell.
        On multi-million-cell CFD frames that turned W4/W5 point lookups into
        full-frame scans.  Coalesced slices keep the same schema/API while
        letting TileDB read only the tiles/ranges that are actually needed.
        """
        ids = sorted(set(int(i) for i in cell_ids if int(i) >= 0))
        out = {str(attr): {} for attr in attrs}
        if not ids:
            return out

        ranges = []
        a = b = ids[0]
        for cid in ids[1:]:
            if cid == b + 1:
                b = cid
            else:
                ranges.append((a, b + 1))
                a = b = cid
        ranges.append((a, b + 1))

        with self.open_array(uri, "r") as A:
            # Newer TileDB-Py releases support multi-range indexing directly.
            # Use it when possible so scattered line/plane hits do not become
            # thousands of Python slice calls.
            if len(ranges) > 1:
                try:
                    query = A.query(attrs=list(attrs))
                    data = query.multi_index[ids]
                    for attr in attrs:
                        arr = np.asarray(data[str(attr)]).reshape(-1)
                        for cid, value in zip(ids, arr):
                            out[str(attr)][int(cid)] = value.item() if hasattr(value, "item") else value
                    return out
                except Exception:
                    pass
            for start, end in ranges:
                try:
                    data = A.query(attrs=list(attrs))[start:end]
                except Exception:
                    data = A[start:end]
                count = end - start
                for attr in attrs:
                    arr = np.asarray(data[str(attr)]).reshape(-1)
                    for offset in range(min(count, arr.size)):
                        value = arr[offset]
                        out[str(attr)][start + offset] = value.item() if hasattr(value, "item") else value
        return out

    def fetch_cell_scalar_map(
        self, dataset_key: str, step: int, var: str, cell_ids: Sequence[int], zone: str = "0_Fluid"
    ) -> Dict[int, float]:
        norm_ids = [int(i) for i in cell_ids]
        if not norm_ids:
            return {}
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        raw = self._read_dense_attrs_at_ids(uri, [str(var)], norm_ids)[str(var)]
        return {cid: float(value) for cid, value in raw.items()}

    def fetch_all_cell_scalars(
        self, dataset_key: str, step: int, var: str, zone: str = "0_Fluid"
    ) -> np.ndarray:
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            try:
                data = A.query(attrs=[str(var)])[:]
            except Exception:
                data = A[:]
            return np.asarray(data[str(var)], dtype=np.float64)

    def fetch_cell_ids_by_var_range(
        self, dataset_key: str, step: int, var: str, lower: float, upper: float, zone: str = "0_Fluid"
    ) -> List[int]:
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        lo, hi = sorted((float(lower), float(upper)))
        # Prefer TileDB's native QueryCondition so filtering runs in the C++
        # engine and only matching coordinates cross the Python boundary.
        with self.open_array(uri, "r") as A:
            cond = f"{str(var)} >= {lo} and {str(var)} <= {hi}"
            try:
                result = A.query(attrs=[str(var)], dims=["cell_id"], cond=cond)[:]
                if "cell_id" in result:
                    return [int(x) for x in np.asarray(result["cell_id"]).reshape(-1)]
            except Exception:
                pass
            try:
                data = A.query(attrs=[str(var)])[:]
            except Exception:
                data = A[:]
            var_arr = np.asarray(data[str(var)], dtype=np.float64).reshape(-1)
        return np.flatnonzero((var_arr >= lo) & (var_arr <= hi)).astype(np.int64).tolist()

    def fetch_velocity_map(
        self, dataset_key: str, step: int, cell_ids: Sequence[int], zone: str = "0_Fluid"
    ) -> Dict[int, Tuple[float, float, float]]:
        norm_ids = [int(i) for i in cell_ids]
        if not norm_ids:
            return {}
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        values = self._read_dense_attrs_at_ids(uri, ["U", "V", "W"], norm_ids)
        return {
            cid: (float(values["U"][cid]), float(values["V"][cid]), float(values["W"][cid]))
            for cid in sorted(set(norm_ids))
            if cid in values["U"] and cid in values["V"] and cid in values["W"]
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

    # -------------------- H5 metadata / W9-W11 --------------------
    @staticmethod
    def _meta_text(value, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _array_attr_names(A) -> List[str]:
        try:
            return [A.schema.attr(i).name for i in range(A.schema.nattr)]
        except Exception:
            try:
                return [x.name for x in A.schema]
            except Exception:
                return []

    def h5_dataset_metadata(self, dataset_key: str) -> Dict[str, object]:
        if dataset_key in self._h5_meta_cache:
            return dict(self._h5_meta_cache[dataset_key])
        uri = self.path_h5_metadata(dataset_key)
        if not self.array_exists(uri):
            return {}
        with self.open_array(uri, "r") as A:
            data = A[0]
            def scalar(name, default=0):
                try:
                    v = data[name]
                    arr = np.asarray(v).reshape(-1)
                    return arr[0] if arr.size else default
                except Exception:
                    return default
            def m(name, default=""):
                try:
                    return self._meta_text(A.meta[name], default)
                except Exception:
                    return default
            def csv(name):
                return tuple(x for x in (p.strip().upper() for p in m(name).split(",")) if x)
            result = {
                "is_h5": bool(int(scalar("is_h5", 0))),
                "zone": m("zone", "0_Fluid"),
                "part_name": m("part_name"),
                "instance_name": m("instance_name"),
                "variables": csv("variables_csv"),
                "nodal_variables": csv("nodal_variables_csv"),
                "common_variables": csv("common_variables_csv"),
                "common_nodal_variables": csv("common_nodal_variables_csv"),
                "element_types": tuple(x for x in m("element_types_csv").split(",") if x),
                "timesteps": tuple(int(x) for x in m("timesteps_csv").split(",") if x.strip()),
                "node_count": int(scalar("node_count", 0)),
                "cell_count": int(scalar("cell_count", 0)),
            }
        self._h5_meta_cache[dataset_key] = dict(result)
        return result

    def h5_frame_timesteps(self, dataset_key: str) -> List[int]:
        if dataset_key in self._h5_steps_cache:
            return list(self._h5_steps_cache[dataset_key])
        meta = self.h5_dataset_metadata(dataset_key)
        steps = [int(x) for x in meta.get("timesteps", ())]
        if not steps:
            root = os.path.join(self._base(dataset_key), "post_processing")
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if name.startswith("step_"):
                        try:
                            steps.append(int(name.split("_", 1)[1]))
                        except ValueError:
                            pass
        result = sorted(set(steps))
        self._h5_steps_cache[dataset_key] = list(result)
        return result

    def is_h5_dataset(self, dataset_key: str) -> bool:
        try:
            return bool(self.h5_dataset_metadata(dataset_key).get("is_h5"))
        except Exception:
            return False

    def list_cell_variables(
        self, dataset_key: str, step: int, zone: str = "0_Fluid"
    ) -> List[str]:
        """List attributes stored in the cell-variable array for a frame/zone."""
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            return sorted(str(name).upper() for name in self._array_attr_names(A))

    def fetch_var_value_range(
        self, dataset_key: str, step: int, var: str, zone: str = "0_Fluid"
    ) -> Tuple[float, float]:
        key = (str(dataset_key), int(step), str(zone), str(var).upper())
        cached = self._var_range_cache.get(key)
        if cached is not None:
            return cached
        uri = self._resolve_cell_vars_uri(dataset_key, step, zone)
        with self.open_array(uri, "r") as A:
            try:
                data = A.query(attrs=[str(var)])[:]
            except Exception:
                data = A[:]
            arr = np.asarray(data[str(var)], dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            raise ValueError(f"no values for {dataset_key} step={step} var={var}")
        result = (float(np.min(arr)), float(np.max(arr)))
        self._var_range_cache[key] = result
        return result

    def fetch_max_diffs(
        self, dataset_key: str, step: int, variables: Sequence[str]
    ) -> Dict[str, float]:
        uri = self.path_derived(dataset_key, step, "max_diff")
        if not self.array_exists(uri):
            return {}
        wanted = [str(v).upper() for v in variables]
        with self.open_array(uri, "r") as A:
            attrs = set(self._array_attr_names(A))
            selected = [v for v in wanted if not attrs or v in attrs]
            if not selected:
                return {}
            try:
                data = A.query(attrs=selected)[0]
            except Exception:
                data = A[0]
            out: Dict[str, float] = {}
            for var in selected:
                try:
                    arr = np.asarray(data[var], dtype=np.float64).reshape(-1)
                    if arr.size and np.isfinite(arr[0]):
                        out[var] = float(arr[0])
                except Exception:
                    continue
            return out

    def _source_labels(self, dataset_key: str, zone: str, kind: str) -> np.ndarray:
        cache = self._node_source_cache if kind == "node" else self._cell_source_cache
        key = (dataset_key, zone)
        if key in cache:
            return cache[key]
        leaf = "node_source" if kind == "node" else "cell_source"
        uri = self.path_mesh_static(dataset_key, zone, leaf)
        with self.open_array(uri, "r") as A:
            data = A.query(attrs=["source_label"])[:] if hasattr(A, "query") else A[:]
            arr = np.asarray(data["source_label"], dtype=np.int64).reshape(-1)
        cache[key] = arr
        return arr

    def fetch_h5_element_labels(
        self, dataset_key: str, zone: str, dense_cell_ids: Sequence[int]
    ) -> List[int]:
        ids = np.asarray([int(x) for x in dense_cell_ids], dtype=np.int64)
        if not ids.size:
            return []
        labels = self._source_labels(dataset_key, zone, "cell")
        valid = ids[(ids >= 0) & (ids < len(labels))]
        return sorted(int(x) for x in labels[valid])

    def fetch_h5_element_ids_in_coordinate_range(
        self, dataset_key: str, zone: str, lower: Sequence[float], upper: Sequence[float]
    ) -> List[int]:
        lo = np.asarray(lower, dtype=np.float64).reshape(3)
        hi = np.asarray(upper, dtype=np.float64).reshape(3)
        uri = self.path_mesh_static(dataset_key, zone, "cells")
        dense_ids: np.ndarray
        with self.open_array(uri, "r") as A:
            cond = (
                f"cx >= {float(lo[0])} and cx <= {float(hi[0])} and "
                f"cy >= {float(lo[1])} and cy <= {float(hi[1])} and "
                f"cz >= {float(lo[2])} and cz <= {float(hi[2])}"
            )
            try:
                result = A.query(attrs=[str(var)], dims=["cell_id"], cond=cond)[:]
                dense_ids = np.asarray(result["cell_id"], dtype=np.int64).reshape(-1)
            except Exception:
                data = A.query(attrs=["cx", "cy", "cz"])[:] if hasattr(A, "query") else A[:]
                x = np.asarray(data["cx"], dtype=np.float64).reshape(-1)
                y = np.asarray(data["cy"], dtype=np.float64).reshape(-1)
                z = np.asarray(data["cz"], dtype=np.float64).reshape(-1)
                mask = (
                    (x >= lo[0]) & (x <= hi[0]) &
                    (y >= lo[1]) & (y <= hi[1]) &
                    (z >= lo[2]) & (z <= hi[2])
                )
                dense_ids = np.flatnonzero(mask)
        return self.fetch_h5_element_labels(dataset_key, zone, dense_ids)

    def fetch_h5_point_ids(self, dataset_key: str, zone: str) -> List[int]:
        return sorted(int(x) for x in self._source_labels(dataset_key, zone, "node"))

    def _dense_node_ids_for_source_labels(
        self, dataset_key: str, zone: str, source_labels: Sequence[int]
    ) -> Dict[int, int]:
        wanted = {int(x) for x in source_labels}
        if not wanted:
            return {}
        labels = self._source_labels(dataset_key, zone, "node")
        return {int(label): int(i) for i, label in enumerate(labels) if int(label) in wanted}

    def _read_attr_at_dense_ids(self, uri: str, var: str, dim_name: str, ids: Sequence[int]):
        ids = sorted(set(int(x) for x in ids))
        if not ids:
            return {}
        with self.open_array(uri, "r") as A:
            cond = " or ".join(f"{dim_name} == {x}" for x in ids)
            try:
                result = A.query(attrs=[var], dims=[dim_name], cond=cond)[:]
                coords = np.asarray(result[dim_name], dtype=np.int64).reshape(-1)
                vals = np.asarray(result[var], dtype=np.float64).reshape(-1)
                return {int(i): float(v) for i, v in zip(coords, vals)}
            except Exception:
                try:
                    data = A.query(attrs=[var])[:]
                except Exception:
                    data = A[:]
                arr = np.asarray(data[var], dtype=np.float64).reshape(-1)
                return {i: float(arr[i]) for i in ids if 0 <= i < len(arr)}

    def fetch_h5_point_frame_extrema(
        self, dataset_key: str, zone: str, source_labels: Sequence[int], var: str
    ) -> Dict[int, Tuple[float, float]]:
        mapping = self._dense_node_ids_for_source_labels(dataset_key, zone, source_labels)
        if not mapping:
            return {}
        dense_to_source = {dense: src for src, dense in mapping.items()}
        mins = {src: np.inf for src in mapping}
        maxs = {src: -np.inf for src in mapping}
        for step in self.h5_frame_timesteps(dataset_key):
            uri = self.path_node_vars(dataset_key, step)
            if not self.array_exists(uri):
                continue
            try:
                values = self._read_attr_at_dense_ids(uri, str(var).upper(), "node_id", dense_to_source)
            except Exception:
                continue
            for dense_id, value in values.items():
                if dense_id not in dense_to_source or not np.isfinite(value):
                    continue
                src = dense_to_source[dense_id]
                mins[src] = min(mins[src], value)
                maxs[src] = max(maxs[src], value)
        return {
            src: (float(mins[src]), float(maxs[src]))
            for src in mapping
            if np.isfinite(mins[src]) and np.isfinite(maxs[src])
        }

    def fetch_frame_statistics(
        self, dataset_key: str, zone: str, step: int, attribute_name: Optional[str] = None
    ) -> Dict[str, Dict[str, object]]:
        meta = self.h5_dataset_metadata(dataset_key)
        variables = [str(v).upper() for v in meta.get("variables", ())]
        nodal = {str(v).upper() for v in meta.get("nodal_variables", ())}
        if attribute_name is not None:
            wanted = str(attribute_name).upper()
            variables = [wanted] if wanted in variables else []
        out: Dict[str, Dict[str, object]] = {}
        for var in variables:
            candidates = []
            if var in nodal:
                candidates.append(("node", self.path_node_vars(dataset_key, step)))
            candidates.append(("cell", self.path_cell_vars(dataset_key, step, zone)))
            for position, uri in candidates:
                if not self.array_exists(uri):
                    continue
                try:
                    with self.open_array(uri, "r") as A:
                        attrs = set(self._array_attr_names(A))
                        if attrs and var not in attrs:
                            continue
                        try:
                            data = A.query(attrs=[var])[:]
                        except Exception:
                            data = A[:]
                        arr = np.asarray(data[var], dtype=np.float64).reshape(-1)
                except Exception:
                    continue
                arr = arr[np.isfinite(arr)]
                if not arr.size:
                    continue
                out[var] = {
                    "position": position,
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "stddev": float(np.std(arr)),
                }
                break
        if not out:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no values for frame={step}{suffix}")
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
