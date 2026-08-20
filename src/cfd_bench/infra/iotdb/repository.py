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
        self._cfd_meta_cache: Dict[str, Dict[str, object]] = {}
        self._var_range_cache: Dict[Tuple[str, int, str, str], Tuple[float, float]] = {}

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

    def path_cfd_metadata(self, dataset_key: str, leaf: str = "dataset_meta") -> str:
        return f"{self.config.root_path}.cfd_metadata.{dataset_key}.{leaf}"

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
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        out: Dict[int, float] = {}
        for start in range(0, len(norm_ids), 5000):
            ids = norm_ids[start:start + 5000]
            idx = ",".join(str(i) for i in ids)
            sql = f"SELECT {var} FROM {path} WHERE Time IN ({idx});"
            for cid, vals in self.query_rows(sql):
                if vals:
                    out[int(cid)] = _to_float(vals[0])
        return out

    def fetch_cell_scalar_values_bulk(
        self,
        dataset_key: str,
        step: int,
        var: str,
        cell_ids: Sequence[int],
        zone: str = "0_Fluid",
    ) -> np.ndarray:
        """High-throughput scalar fetch for large CFD cell selections.

        W2/W6 can select tens of thousands of cells.  Building many
        ``Time IN (...)`` statements is dominated by SQL parsing and network
        round-trips in IoTDB.  For a large selection, query one contiguous
        time window (using the native raw-data API when available) and keep
        only the requested timestamps locally.  Small selections retain the
        historical point-query path.

        This helper is intentionally opt-in; existing workloads continue to
        use :meth:`fetch_cell_scalar_map` unless their benchmark explicitly
        requests the bulk path.
        """
        ids = np.asarray(list(cell_ids), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.float64)

        # Preserve duplicate/order semantics while doing the backend query on
        # sorted unique IDs.
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        if unique_ids.size <= 4096:
            values = self.fetch_cell_scalar_map(
                dataset_key, step, var, unique_ids.tolist(), zone=zone
            )
            unique_values = np.asarray(
                [values.get(int(cid), np.nan) for cid in unique_ids],
                dtype=np.float64,
            )
            return unique_values[inverse]

        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        lo = int(unique_ids[0])
        hi = int(unique_ids[-1])
        span = hi - lo + 1

        # A large/local ROI is much faster as one sequential time-window query
        # than as dozens of IN-list statements.  Cap the absolute window and
        # sparsity so a selection spread across a multi-million-cell mesh does
        # not accidentally turn into a full-frame scan.
        use_window = span <= max(500_000, int(unique_ids.size) * 8)
        if use_window:
            wanted = {int(cid): i for i, cid in enumerate(unique_ids.tolist())}
            unique_values = np.full(unique_ids.size, np.nan, dtype=np.float64)
            ds = None
            try:
                if self.session is not None and hasattr(self.session, "execute_raw_data_query"):
                    ds = self.session.execute_raw_data_query(
                        [f"{path}.{var}"], lo, hi + 1
                    )
                else:
                    ds = self.query(
                        f"SELECT {var} FROM {path} "
                        f"WHERE Time >= {lo} AND Time <= {hi};"
                    )
                while ds.has_next():
                    row = ds.next()
                    pos = wanted.get(int(row.get_timestamp()))
                    if pos is None:
                        continue
                    fields = row.get_fields()
                    if fields:
                        unique_values[pos] = _to_float(_field_to_value(fields[0]))
                return unique_values[inverse]
            except Exception:
                # Older IoTDB clients/servers may not expose the raw range API
                # with the same signature. Fall back to fewer, larger IN-list
                # requests rather than the historical 5000-ID fragmentation.
                pass
            finally:
                if ds is not None:
                    close = getattr(ds, "close_operation_handle", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass

        # Sparse large selections: keep exact semantics but reduce round trips.
        out: Dict[int, float] = {}
        chunk = 20_000
        for start in range(0, unique_ids.size, chunk):
            part = unique_ids[start:start + chunk]
            idx = ",".join(str(int(i)) for i in part)
            sql = f"SELECT {var} FROM {path} WHERE Time IN ({idx});"
            for cid, vals in self.query_rows(sql):
                if vals:
                    out[int(cid)] = _to_float(vals[0])
        unique_values = np.asarray(
            [out.get(int(cid), np.nan) for cid in unique_ids], dtype=np.float64
        )
        return unique_values[inverse]

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
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var="U")
        # Large ROIs can make a single ``Time IN (...)`` query too long for
        # some IoTDB server/client versions.  Chunking also makes W7's halo
        # expansion safe on large CFD meshes without changing the schema.
        out: Dict[int, Tuple[float, float, float]] = {}
        chunk = 5000
        for start in range(0, len(norm_ids), chunk):
            ids = norm_ids[start : start + chunk]
            idx = ",".join(str(i) for i in ids)
            sql = f"SELECT U,V,W FROM {path} WHERE Time IN ({idx});"
            for cid, vals in self.query_rows(sql):
                if len(vals) < 3:
                    continue
                out[int(cid)] = (
                    _to_float(vals[0]),
                    _to_float(vals[1]),
                    _to_float(vals[2]),
                )
        return out

    # -------------------- mesh static --------------------

    def fetch_cells_arrays(self, dataset_key: str, zone: str):
        """Return compact NumPy cell geometry arrays for runtime hot paths."""
        path = self.path_mesh_static(dataset_key, zone, "cells")
        cell_count = None
        try:
            meta_path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
            rows = self.query_rows(f"SELECT cell_count FROM {meta_path} LIMIT 1;")
            if rows and rows[0][1]:
                cell_count = _to_int(rows[0][1][0], -1)
        except Exception:
            cell_count = None

        fields = "cx,cy,cz,xmin,xmax,ymin,ymax,zmin,zmax,cell_type"
        parts_ids = []
        parts_centers = []
        parts_mins = []
        parts_maxs = []
        parts_types = []

        def consume(rows):
            good = [(cid, vals) for cid, vals in rows if len(vals) >= 10]
            if not good:
                return
            parts_ids.append(np.fromiter((int(cid) for cid, _ in good), dtype=np.int64, count=len(good)))
            parts_centers.append(np.asarray([
                [_to_float(v[0]), _to_float(v[1]), _to_float(v[2])] for _, v in good
            ], dtype=np.float64))
            parts_mins.append(np.asarray([
                [_to_float(v[3]), _to_float(v[5]), _to_float(v[7])] for _, v in good
            ], dtype=np.float64))
            parts_maxs.append(np.asarray([
                [_to_float(v[4]), _to_float(v[6]), _to_float(v[8])] for _, v in good
            ], dtype=np.float64))
            parts_types.append(np.fromiter((_to_int(v[9], 0) for _, v in good), dtype=np.int32, count=len(good)))

        if cell_count is None or cell_count <= 0:
            consume(self.query_rows(f"SELECT {fields} FROM {path};"))
        else:
            chunk = 50000
            for start in range(0, cell_count, chunk):
                end = start + chunk
                consume(self.query_rows(
                    f"SELECT {fields} FROM {path} WHERE Time >= {start} AND Time < {end};"
                ))

        if not parts_ids:
            return (
                np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
            )
        ids = np.concatenate(parts_ids)
        centers = np.concatenate(parts_centers, axis=0)
        mins = np.concatenate(parts_mins, axis=0)
        maxs = np.concatenate(parts_maxs, axis=0)
        types = np.concatenate(parts_types)
        order = np.argsort(ids, kind="stable")
        return (
            ids[order].astype(np.int32, copy=False),
            np.ascontiguousarray(centers[order], dtype=np.float64),
            np.ascontiguousarray(mins[order], dtype=np.float64),
            np.ascontiguousarray(maxs[order], dtype=np.float64),
            types[order].astype(np.int32, copy=False),
        )

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

    def fetch_mesh_meta(self, dataset_key: str, zone: str) -> Dict[str, float]:
        path = self.path_mesh_static(dataset_key, zone, "mesh_meta")
        fields = [
            "node_count", "cell_count", "face_count",
            "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y",
            "bbox_min_z", "bbox_max_z",
        ]
        rows = self.query_rows(f"SELECT {','.join(fields)} FROM {path} LIMIT 1;")
        if not rows or not rows[0][1]:
            return {}
        vals = rows[0][1]
        return {name: _to_float(vals[i]) for i, name in enumerate(fields) if i < len(vals)}

    def fetch_nodes_subset(
        self, dataset_key: str, zone: str, node_ids: Sequence[int]
    ) -> Dict[int, Tuple[float, float, float]]:
        ids = sorted(set(int(x) for x in node_ids if int(x) >= 0))
        if not ids:
            return {}
        path = self.path_mesh_static(dataset_key, zone, "nodes")
        out = {}
        for start in range(0, len(ids), 5000):
            chunk = ids[start:start + 5000]
            idx = ",".join(str(x) for x in chunk)
            for nid, vals in self.query_rows(f"SELECT x,y,z FROM {path} WHERE Time IN ({idx});"):
                if len(vals) >= 3:
                    out[int(nid)] = (_to_float(vals[0]), _to_float(vals[1]), _to_float(vals[2]))
        return out

    def fetch_cell_nodes_subset(
        self, dataset_key: str, zone: str, cell_ids: Sequence[int]
    ) -> Dict[int, List[int]]:
        ids = sorted(set(int(x) for x in cell_ids if int(x) >= 0))
        if not ids:
            return {}
        width = self._mesh_meta_int(dataset_key, zone, "max_nodes_per_cell", 16)
        fields = ",".join(f"node_id_{i}" for i in range(width))
        path = self.path_mesh_static(dataset_key, zone, "cell_nodes")
        out = {}
        for start in range(0, len(ids), 5000):
            chunk = ids[start:start + 5000]
            idx = ",".join(str(x) for x in chunk)
            for cid, vals in self.query_rows(f"SELECT {fields} FROM {path} WHERE Time IN ({idx});"):
                row = [_to_int(v, -1) for v in vals]
                out[int(cid)] = [x for x in row if x >= 0]
        return out

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

    def fetch_boundary_faces(
        self, dataset_key: str, zone: str, patch_name: str = "*"
    ) -> List[Tuple[int, float, float, float, float, float, float, float, float]]:
        """Read legacy CFD boundary-face rows.

        ``load_topology.py`` stores boundary faces with the face-row index as
        IoTDB Time and the *owning cell id* in the ``cell_id`` measurement.
        Older code incorrectly treated Time as cell_id and queried a
        non-existent ``patch_name`` field even though ingest writes
        ``patch_code``.  W6 therefore could not use IoTDB-native geometry
        reliably.  Keep the persisted layout unchanged and read it correctly.
        """
        path = self.path_mesh_static(dataset_key, zone, "boundary_faces")
        sql = f"SELECT cell_id,patch_code,nx,ny,nz,area,cx,cy,cz FROM {path};"
        try:
            rows = self.query_rows(sql)
        except Exception:
            return []
        out: List[Tuple[int, float, float, float, float, float, float, float, float]] = []
        for _face_row, vals in rows:
            if len(vals) < 9:
                continue
            cid = _to_int(vals[0], -1)
            patch_code = _to_float(vals[1], 0.0)
            if cid < 0:
                continue
            if patch_name != "*":
                # Legacy DAT ingest only has a numeric patch code.  Accept
                # either the numeric string or the all-patches wildcard.
                try:
                    if float(patch_name) != patch_code:
                        continue
                except Exception:
                    continue
            out.append(
                (
                    cid,
                    patch_code,
                    _to_float(vals[2]),
                    _to_float(vals[3]),
                    _to_float(vals[4]),
                    _to_float(vals[5]),
                    _to_float(vals[6]),
                    _to_float(vals[7]),
                    _to_float(vals[8]),
                )
            )
        return out

    # -------------------- legacy CFD discovery metadata --------------------
    def cfd_dataset_metadata(self, dataset_key: str) -> Dict[str, object]:
        """Read metadata written by the canonical DAT ingest path.

        This lives under ``cfd_metadata`` so the frozen H5 metadata contract is
        untouched.  Older IoTDB databases simply return an empty mapping and
        retain the historical CLI overrides/defaults.
        """
        if dataset_key in self._cfd_meta_cache:
            return dict(self._cfd_meta_cache[dataset_key])
        path = self.path_cfd_metadata(dataset_key, "dataset_meta")
        fields = [
            "is_cfd", "zone", "zones_csv", "variables_csv",
            "timesteps_csv", "node_count", "cell_count",
        ]
        try:
            rows = self.query_rows(f"SELECT {','.join(fields)} FROM {path} LIMIT 1;")
        except Exception:
            return {}
        if not rows or not rows[0][1]:
            return {}
        vals = list(rows[0][1]) + [None] * len(fields)

        def csv_at(i, *, upper=False):
            text = str(vals[i] or "").strip()
            items = [x.strip() for x in text.split(",") if x.strip()]
            return tuple(x.upper() for x in items) if upper else tuple(items)

        steps = []
        for token in csv_at(4):
            try:
                steps.append(int(token))
            except Exception:
                pass
        result = {
            "is_cfd": _to_bool(vals[0]),
            "zone": str(vals[1] or "0_Fluid"),
            "zones": csv_at(2),
            "variables": csv_at(3, upper=True),
            "timesteps": tuple(sorted(set(steps))),
            "node_count": _to_int(vals[5], 0),
            "cell_count": _to_int(vals[6], 0),
        }
        self._cfd_meta_cache[dataset_key] = dict(result)
        return result

    def fetch_cfd_element_ids_in_coordinate_range(
        self, dataset_key: str, zone: str, lower: Sequence[float], upper: Sequence[float]
    ) -> List[int]:
        """Return implicit one-based Tecplot element IDs by centroid range."""
        lo = [float(x) for x in lower]
        hi = [float(x) for x in upper]
        path = self.path_mesh_static(dataset_key, zone, "cells")
        rows = self.query_rows(
            "SELECT cx FROM " + path +
            f" WHERE cx >= {lo[0]} AND cx <= {hi[0]}"
            f" AND cy >= {lo[1]} AND cy <= {hi[1]}"
            f" AND cz >= {lo[2]} AND cz <= {hi[2]};"
        )
        return sorted(int(cell_id) + 1 for cell_id, _ in rows)

    def fetch_cfd_frame_statistics(
        self,
        dataset_key: str,
        zone: str,
        step: int,
        attribute_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Cell-centered statistics for one legacy CFD frame."""
        meta = self.cfd_dataset_metadata(dataset_key)
        variables = [str(v).upper() for v in meta.get("variables", ())]
        if attribute_name is not None:
            wanted = str(attribute_name).upper()
            variables = [wanted] if wanted in variables else []
        out: Dict[str, Dict[str, object]] = {}
        if not variables:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no CFD values for frame={step}{suffix}")
        path = self.resolve_cell_var_path(
            dataset_key, int(step), zone=zone, probe_var=variables[0]
        )
        for var in variables:
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
                    "position": "cell",
                    "count": _to_int(vals[0], 0),
                    "min": _to_float(vals[1]),
                    "max": _to_float(vals[2]),
                    "mean": _to_float(vals[3]),
                    "stddev": _to_float(vals[4], 0.0),
                }
                continue
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
                    "position": "cell",
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "stddev": float(np.std(arr)),
                }
        if not out:
            suffix = f" variable={attribute_name}" if attribute_name else ""
            raise ValueError(f"no CFD values for frame={step}{suffix}")
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
        key = (str(dataset_key), int(step), str(zone), str(var).upper())
        cached = self._var_range_cache.get(key)
        if cached is not None:
            return cached
        path = self.resolve_cell_var_path(dataset_key, step, zone=zone, probe_var=var)
        result = None
        try:
            rows = self.query_rows(
                f"SELECT MIN_VALUE({var}),MAX_VALUE({var}) FROM {path};"
            )
            if rows and len(rows[0][1]) >= 2:
                result = (_to_float(rows[0][1][0]), _to_float(rows[0][1][1]))
        except Exception:
            pass
        if result is None:
            values = [
                _to_float(vals[0])
                for _, vals in self.query_rows(f"SELECT {var} FROM {path};")
                if vals and np.isfinite(_to_float(vals[0]))
            ]
            if not values:
                raise ValueError(f"no values for {dataset_key} step={step} var={var}")
            result = (float(np.min(values)), float(np.max(values)))
        self._var_range_cache[key] = (float(result[0]), float(result[1]))
        return self._var_range_cache[key]

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

