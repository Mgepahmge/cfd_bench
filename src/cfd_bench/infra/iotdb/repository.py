from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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


class IoTDBRepository:
    def __init__(self, config: IoTDBConfig):
        self.config = config
        self.session: Optional[Session] = None
        self._cell_var_path_cache: Dict[Tuple[str, int, str], str] = {}

    def open(self):
        if self.session is not None:
            return
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

    def fetch_cell_nodes(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        fields = ",".join(f"node_id_{i}" for i in range(16))
        sql = f"SELECT {fields} FROM {self.path_mesh_static(dataset_key, zone, 'cell_nodes')};"
        out: Dict[int, List[int]] = {}
        for cid, vals in self.query_rows(sql):
            ids = [_to_int(v, -1) for v in vals]
            out[cid] = [x for x in ids if x >= 0]
        return out

    def fetch_cell_adjacency(self, dataset_key: str, zone: str) -> Dict[int, List[int]]:
        fields = ",".join(f"neighbor_id_{i}" for i in range(16))
        sql = f"SELECT {fields} FROM {self.path_mesh_static(dataset_key, zone, 'cell_adjacency')};"
        out: Dict[int, List[int]] = {}
        for cid, vals in self.query_rows(sql):
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
