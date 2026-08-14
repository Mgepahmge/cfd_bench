from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from iotdb.Session import Session

from .config import IoTDBConfig


def _field_to_value(field):
    if field is None:
        return None
    text = field.get_string_value()
    if text is None:
        return None
    return text


def _to_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v, default=-1):
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    text = str(v).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(default)


class IoTDBRepository:
    def __init__(self, config: IoTDBConfig):
        self.config = config
        self.session: Optional["Session"] = None
        self._cell_var_path_cache: Dict[Tuple[str, int, str], str] = {}
        self._h5_meta_cache: Dict[str, Dict[str, object]] = {}
        self._h5_steps_cache: Dict[str, List[int]] = {}

    def open(self):
        if self.session is not None:
            return
        try:
            from iotdb.Session import Session
        except ImportError as exc:
            raise RuntimeError(
                "IoTDB backend requires apache-iotdb. "
                "Install with: pip install 'cfd_bench[iotdb]'"
            ) from exc
        self.session = Session(self.config.host, self.config.port, self.config.user, self.config.password)
        self.session.open()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    # -------------------- path helpers --------------------
    def path_cell_vars(self, dataset_key: str, step: int, leaf: str = "cell_vars") -> str:
        return f"{self.config.root_path}.post_processing_management.{dataset_key}.step_{step}.{leaf}"

    @staticmethod
    def cell_vars_leaf_for_zone(zone: str) -> str:
        z = (zone or "0_Fluid").strip().lower()
        if "hull" in z or "wall" in z or z in ("1_hull", "hull"):
            return "cell_vars_hull"
        return "cell_vars"

    def path_cell_vars_for_zone(self, dataset_key: str, step: int, zone: str = "0_Fluid") -> str:
        return self.path_cell_vars(dataset_key, step, self.cell_vars_leaf_for_zone(zone))

    def path_node_vars(self, dataset_key: str, step: int) -> str:
        return f"{self.config.root_path}.post_processing_management.{dataset_key}.step_{step}.node_vars"

    def path_mesh_static(self, dataset_key: str, zone: str, leaf: str) -> str:
        return f"{self.config.root_path}.mesh_static.{dataset_key}.{zone}.{leaf}"

    def path_derived(self, dataset_key: str, step: int, leaf: str) -> str:
        return f"{self.config.root_path}.derived.{dataset_key}.step_{step}.{leaf}"

    def path_h5_metadata(self, dataset_key: str, leaf: str) -> str:
        return f"{self.config.root_path}.h5_metadata.{dataset_key}.{leaf}"

    # -------------------- generic query --------------------
    def query(self, sql: str):
        if self.session is None:
            raise RuntimeError("IoTDB session is not open")
        return self.session.execute_query_statement(sql)

    def query_rows(self, sql: str) -> List[Tuple[int, List[str]]]:
        ds = self.query(sql)
        out: List[Tuple[int, List[str]]] = []
        while ds.has_next():
            row = ds.next()
            fields = [_field_to_value(x) for x in row.get_fields()]
            out.append((int(row.get_timestamp()), fields))
        return out

    def resolve_cell_var_path(
        self, dataset_key: str, step: int, zone: str = "0_Fluid", probe_var: str = "P"
    ) -> str:
        key = (dataset_key, int(step), zone)
        if key in self._cell_var_path_cache:
            return self._cell_var_path_cache[key]
        path = self.path_cell_vars_for_zone(dataset_key, step, zone)
        try:
            rows = self.query_rows(f"SELECT {probe_var} FROM {path} LIMIT 1;")
            if rows:
                self._cell_var_path_cache[key] = path
                return path
        except Exception:
            pass
        raise RuntimeError(
            f"未找到 IoTDB 变量路径: {path}. "
            "请先运行 cfd-bench ingest 导入 step_xxx.cell_vars。"
        )

    # -------------------- scalar queries --------------------
    def fetch_cell_scalar_map(
        self,
        dataset_key: str,
        step: int,
        var: str,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> Dict[int, float]:
        if cell_ids is None:
            norm_ids: List[int] = []
        else:
            norm_ids = [int(i) for i in list(cell_ids)]
        if len(norm_ids) == 0:
            return {}
        idx = ",".join(str(i) for i in norm_ids)
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        sql = f"SELECT {var} FROM {path} WHERE Time IN ({idx});"
        rows = self.query_rows(sql)
        return {cid: _to_float(vals[0]) for cid, vals in rows}

    def fetch_cell_ids_by_var_range(
        self,
        dataset_key: str,
        step: int,
        var: str,
        lower: float,
        upper: float,
        zone: str = "0_Fluid",
    ) -> List[int]:
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        sql = (
            f"SELECT {var} FROM {path} "
            f"WHERE {var} >= {float(lower)} AND {var} <= {float(upper)};"
        )
        return [cid for cid, _ in self.query_rows(sql)]

    def fetch_velocity_map(
        self,
        dataset_key: str,
        step: int,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> Dict[int, Tuple[float, float, float]]:
        norm_ids = [int(i) for i in cell_ids]
        if not norm_ids:
            return {}
        idx = ",".join(str(i) for i in norm_ids)
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var="U")
        sql = f"SELECT U,V,W FROM {path} WHERE Time IN ({idx});"
        rows = self.query_rows(sql)
        return {
            cid: (_to_float(vals[0]), _to_float(vals[1]), _to_float(vals[2]))
            for cid, vals in rows
            if len(vals) >= 3
        }

    # -------------------- mesh static --------------------
    def fetch_cells(self, dataset_key: str, zone: str) -> Dict[int, Tuple[float, ...]]:
        path = self.path_mesh_static(dataset_key, zone, "cells")
        out: Dict[int, Tuple[float, ...]] = {}
        # Read mesh_meta first to enable chunked loading, avoiding full-table timeout.
        cell_count = None
        try:
            meta_path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
            rows = self.query_rows(f"SELECT cell_count FROM {meta_path} LIMIT 1;")
            if rows and rows[0][1]:
                cell_count = _to_int(rows[0][1][0], -1)
        except Exception:
            cell_count = None

        if cell_count is None or cell_count <= 0:
            sql = (
                "SELECT cx,cy,cz,xmin,xmax,ymin,ymax,zmin,zmax,cell_type "
                f"FROM {path};"
            )
            for cid, vals in self.query_rows(sql):
                if len(vals) < 10:
                    continue
                out[cid] = (
                    _to_float(vals[0]),
                    _to_float(vals[1]),
                    _to_float(vals[2]),
                    _to_float(vals[3]),
                    _to_float(vals[4]),
                    _to_float(vals[5]),
                    _to_float(vals[6]),
                    _to_float(vals[7]),
                    _to_float(vals[8]),
                    _to_int(vals[9], 0),
                )
            return out

        chunk = 50000
        for start in range(0, cell_count, chunk):
            end = start + chunk
            sql = (
                "SELECT cx,cy,cz,xmin,xmax,ymin,ymax,zmin,zmax,cell_type "
                f"FROM {path} WHERE Time >= {start} AND Time < {end};"
            )
            for cid, vals in self.query_rows(sql):
                if len(vals) < 10:
                    continue
                out[cid] = (
                    _to_float(vals[0]),
                    _to_float(vals[1]),
                    _to_float(vals[2]),
                    _to_float(vals[3]),
                    _to_float(vals[4]),
                    _to_float(vals[5]),
                    _to_float(vals[6]),
                    _to_float(vals[7]),
                    _to_float(vals[8]),
                    _to_int(vals[9], 0),
                )
        return out

    def fetch_nodes(self, dataset_key: str, zone: str) -> Dict[int, Tuple[float, float, float]]:
        sql = f"SELECT x,y,z FROM {self.path_mesh_static(dataset_key, zone, 'nodes')};"
        return {nid: (_to_float(v[0]), _to_float(v[1]), _to_float(v[2])) for nid, v in self.query_rows(sql)}

    def _mesh_meta_int(self, dataset_key: str, zone: str, field: str, default: int) -> int:
        try:
            path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
            rows = self.query_rows(f"SELECT {field} FROM {path} LIMIT 1;")
            if rows and rows[0][1]:
                value = _to_int(rows[0][1][0], default)
                return value if value > 0 else int(default)
        except Exception:
            pass
        return int(default)

    def fetch_cell_nodes(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        width = self._mesh_meta_int(dataset_key, zone, "max_nodes_per_cell", 16)
        fields = ",".join(f"node_id_{i}" for i in range(width))
        sql = f"SELECT {fields} FROM {self.path_mesh_static(dataset_key, zone, 'cell_nodes')};"
        out: Dict[int, List[int]] = {}
        for cid, vals in self.query_rows(sql):
            ids = [_to_int(v, -1) for v in vals]
            out[cid] = [x for x in ids if x >= 0]
        return out

    def fetch_cell_adjacency(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        width = self._mesh_meta_int(dataset_key, zone, "max_neighbors_per_cell", 16)
        fields = ",".join(f"neighbor_id_{i}" for i in range(width))
        sql = f"SELECT {fields} FROM {self.path_mesh_static(dataset_key, zone, 'cell_adjacency')};"
        out: Dict[int, List[int]] = {}
        try:
            rows = self.query_rows(sql)
        except Exception:
            return out
        for cid, vals in rows:
            ids = [_to_int(v, -1) for v in vals]
            out[cid] = [x for x in ids if x >= 0]
        return out

    def fetch_face_planes(self, dataset_key: str, zone: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
        sql = f"SELECT neighbor_id,nx,ny,nz,d FROM {self.path_mesh_static(dataset_key, zone, 'face_planes')};"
        out: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        for cid, vals in self.query_rows(sql):
            if len(vals) < 5:
                continue
            out.setdefault(cid, []).append(
                (_to_int(vals[0]), _to_float(vals[1]), _to_float(vals[2]), _to_float(vals[3]), _to_float(vals[4]))
            )
        return out

    def fetch_boundary_faces(self, dataset_key: str, zone: str, patch_name: str = "*") -> List[Tuple[int, float, float, float, float]]:
        path = self.path_mesh_static(dataset_key, zone, "boundary_faces")
        sql = f"SELECT nx,ny,nz,area,patch_name FROM {path};"
        rows = self.query_rows(sql)
        out: List[Tuple[int, float, float, float, float]] = []
        for cid, vals in rows:
            if len(vals) < 5:
                continue
            p = vals[4] or ""
            if patch_name != "*" and p != patch_name:
                continue
            out.append((cid, _to_float(vals[0]), _to_float(vals[1]), _to_float(vals[2]), _to_float(vals[3])))
        return out

    # -------------------- H5 metadata / W9-W11 --------------------
    def h5_dataset_metadata(self, dataset_key: str) -> Dict[str, object]:
        if dataset_key in self._h5_meta_cache:
            return dict(self._h5_meta_cache[dataset_key])
        path = self.path_h5_metadata(dataset_key, "dataset_meta")
        fields = [
            "is_h5", "zone", "part_name", "instance_name", "variables_csv",
            "nodal_variables_csv", "common_variables_csv", "common_nodal_variables_csv",
            "element_types_csv", "node_count", "cell_count",
        ]
        rows = self.query_rows(f"SELECT {','.join(fields)} FROM {path} LIMIT 1;")
        if not rows or not rows[0][1]:
            return {}
        vals = list(rows[0][1]) + [None] * len(fields)
        def csv_at(i):
            text = str(vals[i] or "").strip()
            return tuple(x for x in (p.strip().upper() for p in text.split(",")) if x)
        result = {
            "is_h5": _to_bool(vals[0]),
            "zone": str(vals[1] or "0_Fluid"),
            "part_name": str(vals[2] or ""),
            "instance_name": str(vals[3] or ""),
            "variables": csv_at(4),
            "nodal_variables": csv_at(5),
            "common_variables": csv_at(6),
            "common_nodal_variables": csv_at(7),
            "element_types": tuple(x for x in str(vals[8] or "").split(",") if x),
            "node_count": _to_int(vals[9], 0),
            "cell_count": _to_int(vals[10], 0),
        }
        self._h5_meta_cache[dataset_key] = dict(result)
        return result

    def h5_frame_timesteps(self, dataset_key: str) -> List[int]:
        if dataset_key in self._h5_steps_cache:
            return list(self._h5_steps_cache[dataset_key])
        path = self.path_h5_metadata(dataset_key, "frames")
        rows = self.query_rows(f"SELECT frame_index FROM {path};")
        result = sorted(int(ts) for ts, _ in rows)
        self._h5_steps_cache[dataset_key] = list(result)
        return result

    def is_h5_dataset(self, dataset_key: str) -> bool:
        try:
            return bool(self.h5_dataset_metadata(dataset_key).get("is_h5"))
        except Exception:
            return False

    def fetch_var_value_range(
        self, dataset_key: str, step: int, var: str, zone: str = "0_Fluid"
    ) -> Tuple[float, float]:
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        try:
            rows = self.query_rows(
                f"SELECT MIN_VALUE({var}),MAX_VALUE({var}) FROM {path};"
            )
            if rows and len(rows[0][1]) >= 2:
                return _to_float(rows[0][1][0]), _to_float(rows[0][1][1])
        except Exception:
            pass
        values = [
            _to_float(vals[0])
            for _, vals in self.query_rows(f"SELECT {var} FROM {path};")
            if vals and np.isfinite(_to_float(vals[0]))
        ]
        if not values:
            raise ValueError(f"no values for {dataset_key} step={step} var={var}")
        return float(np.min(values)), float(np.max(values))

    def fetch_max_diffs(
        self, dataset_key: str, step: int, variables: Sequence[str]
    ) -> Dict[str, float]:
        vars_ = [str(v).upper() for v in variables]
        if not vars_:
            return {}
        path = self.path_derived(dataset_key, step, "max_diff")
        try:
            rows = self.query_rows(f"SELECT {','.join(vars_)} FROM {path} WHERE Time=0;")
        except Exception:
            return {}
        if not rows:
            return {}
        vals = rows[0][1]
        out = {}
        for var, value in zip(vars_, vals):
            fv = _to_float(value)
            if np.isfinite(fv):
                out[var] = float(fv)
        return out

    def fetch_h5_element_ids_in_coordinate_range(
        self, dataset_key: str, zone: str, lower: Sequence[float], upper: Sequence[float]
    ) -> List[int]:
        lo = [float(x) for x in lower]
        hi = [float(x) for x in upper]
        cells = self.path_mesh_static(dataset_key, zone, "cells")
        rows = self.query_rows(
            "SELECT cx,cy,cz FROM " + cells +
            f" WHERE cx >= {lo[0]} AND cx <= {hi[0]}"
            f" AND cy >= {lo[1]} AND cy <= {hi[1]}"
            f" AND cz >= {lo[2]} AND cz <= {hi[2]};"
        )
        cell_ids = [int(ts) for ts, _ in rows]
        if not cell_ids:
            return []
        src = self.path_mesh_static(dataset_key, zone, "cell_source")
        idx = ",".join(str(x) for x in cell_ids)
        labels = self.query_rows(f"SELECT source_label FROM {src} WHERE Time IN ({idx});")
        return sorted(_to_int(vals[0]) for _, vals in labels if vals)

    def fetch_h5_point_ids(self, dataset_key: str, zone: str) -> List[int]:
        path = self.path_mesh_static(dataset_key, zone, "node_source")
        rows = self.query_rows(f"SELECT source_label FROM {path};")
        return sorted(_to_int(vals[0]) for _, vals in rows if vals)

    def _dense_node_ids_for_source_labels(
        self, dataset_key: str, zone: str, source_labels: Sequence[int]
    ) -> Dict[int, int]:
        wanted = {int(x) for x in source_labels}
        if not wanted:
            return {}
        # H5 source labels need not be dense.  Filter by label in IoTDB so W11
        # does not scan the full node-source mapping on every transaction.
        path = self.path_mesh_static(dataset_key, zone, "node_source")
        labels = ",".join(str(x) for x in sorted(wanted))
        try:
            rows = self.query_rows(
                f"SELECT source_label FROM {path} WHERE source_label IN ({labels});"
            )
        except Exception:
            rows = self.query_rows(f"SELECT source_label FROM {path};")
        out: Dict[int, int] = {}
        for dense_id, vals in rows:
            if not vals:
                continue
            label = _to_int(vals[0])
            if label in wanted:
                out[label] = int(dense_id)
        return out

    def fetch_h5_point_frame_extrema(
        self, dataset_key: str, zone: str, source_labels: Sequence[int], var: str
    ) -> Dict[int, Tuple[float, float]]:
        mapping = self._dense_node_ids_for_source_labels(dataset_key, zone, source_labels)
        if not mapping:
            return {}
        dense_to_source = {dense: src for src, dense in mapping.items()}
        dense_ids = sorted(dense_to_source)
        idx = ",".join(str(x) for x in dense_ids)
        mins = {src: np.inf for src in mapping}
        maxs = {src: -np.inf for src in mapping}
        for step in self.h5_frame_timesteps(dataset_key):
            path = self.path_node_vars(dataset_key, step)
            try:
                rows = self.query_rows(f"SELECT {var} FROM {path} WHERE Time IN ({idx});")
            except Exception:
                continue
            for dense_id, vals in rows:
                if not vals:
                    continue
                value = _to_float(vals[0])
                if not np.isfinite(value) or dense_id not in dense_to_source:
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
        all_vars = list(meta.get("variables", ()))
        nodal = set(meta.get("nodal_variables", ()))
        if attribute_name is not None:
            wanted = str(attribute_name).upper()
            all_vars = [wanted] if wanted in all_vars else []
        out: Dict[str, Dict[str, object]] = {}
        for var in all_vars:
            position = "node" if var in nodal else "cell"
            path = self.path_node_vars(dataset_key, step) if position == "node" else self.path_cell_vars_for_zone(dataset_key, step, zone)
            rows = []
            try:
                rows = self.query_rows(
                    f"SELECT COUNT({var}),MIN_VALUE({var}),MAX_VALUE({var}),"
                    f"AVG({var}),STDDEV_POP({var}) FROM {path};"
                )
            except Exception:
                rows = []
            if rows and len(rows[0][1]) >= 5:
                vals = rows[0][1]
                out[var] = {
                    "position": position,
                    "count": _to_int(vals[0], 0),
                    "min": _to_float(vals[1]),
                    "max": _to_float(vals[2]),
                    "mean": _to_float(vals[3]),
                    "stddev": _to_float(vals[4], 0.0),
                }
                continue
            # Compatibility fallback for older IoTDB servers lacking
            # STDDEV_POP: fetch this one series and compute the same statistics.
            try:
                raw = self.query_rows(f"SELECT {var} FROM {path};")
            except Exception:
                continue
            arr = np.asarray(
                [_to_float(vals[0]) for _, vals in raw if vals], dtype=np.float64
            )
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[var] = {
                    "position": position,
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "stddev": float(np.std(arr)),
                }
        if not out:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no values for frame={step}{suffix}")
        return out

