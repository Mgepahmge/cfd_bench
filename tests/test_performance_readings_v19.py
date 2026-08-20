from __future__ import annotations

import numpy as np
import pytest

from cfd_bench.core.context import MeshContext
from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.infra.iotdb.repository import IoTDBRepository
from cfd_bench.workloads.w2.run import _bench_iotdb_cfd
from cfd_bench.workloads.w6.run import _bench_iotdb_native, _bench_pg_native
from cfd_bench.workloads.w8 import run as w8_run


class _Field:
    def __init__(self, value):
        self.value = value

    def get_string_value(self):
        return str(self.value)


class _Row:
    def __init__(self, timestamp, values):
        self.timestamp = int(timestamp)
        self.values = list(values) if isinstance(values, (tuple, list)) else [values]

    def get_timestamp(self):
        return self.timestamp

    def get_fields(self):
        return [_Field(v) for v in self.values]


class _DataSet:
    def __init__(self, rows):
        self.rows = list(rows)
        self.index = 0
        self.closed = False

    def has_next(self):
        return self.index < len(self.rows)

    def next(self):
        row = self.rows[self.index]
        self.index += 1
        return row

    def close_operation_handle(self):
        self.closed = True


class _QuerySession:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []
        self.last_ds = None

    def execute_query_statement(self, sql):
        self.calls.append(sql)
        self.last_ds = _DataSet(self.rows)
        return self.last_ds


def test_iotdb_query_rows_closes_server_operation_handle():
    repo = IoTDBRepository(IoTDBConfig())
    session = _QuerySession([_Row(0, 1.0)])
    repo.session = session

    rows = repo.query_rows("SELECT P FROM root.demo")

    assert rows == [(0, ["1.0"])]
    assert session.last_ds is not None and session.last_ds.closed


def test_iotdb_w2_selection_uses_server_side_aggregates_not_raw_frame_reads(monkeypatch):
    repo = IoTDBRepository(IoTDBConfig())
    repo.resolve_cell_var_path = lambda *args, **kwargs: "root.demo.step_200.cell_vars"
    seen = []

    def fake_query_rows(sql):
        seen.append(sql)
        # COUNT, SUM, MIN_VALUE, MAX_VALUE
        return [(0, ["4", "10.0", "1.0", "4.0"])]

    monkeypatch.setattr(repo, "query_rows", fake_query_rows)
    result = repo.aggregate_cell_scalar_selection(
        "demo", 200, "P", np.array([10, 11, 12, 13], dtype=np.int32)
    )

    assert result == (4, 10.0, 1.0, 4.0)
    assert len(seen) == 1
    assert "COUNT(P)" in seen[0]
    assert "SUM(P)" in seen[0]
    assert "Time >= 10 AND Time <= 13" in seen[0]
    assert "execute_raw_data_query" not in seen[0]


def test_iotdb_w2_cfd_benchmark_consumes_aggregate_callback():
    calls = []

    def coord_fn(lo, hi):
        return np.array([1, 2, 3, 4], dtype=np.int32)

    def aggregate_fn(cells, var, step):
        calls.append((tuple(int(x) for x in cells), var, int(step)))
        return 4, 10.0, 1.0, 4.0

    _bench_iotdb_cfd(
        coord_fn,
        aggregate_fn,
        [0, 1, 0, 1, 0, 1],
        [200, 400],
        0.005,
        ["P"],
        max_hit_attempts=1,
    )

    assert calls
    assert {step for _cells, _var, step in calls} == {200, 400}


class _W6PGClient:
    def __init__(self):
        self.ctx = MeshContext("demo", 0, "surface")
        self.prepare_calls = 0
        self.force_calls = 0
        self.point_calls = 0

    def prepare_surface_force_query(self):
        self.prepare_calls += 1
        return True

    def surface_force_query(self, var):
        self.force_calls += 1
        return np.array([1.0, 2.0, 3.0])

    def surface_cells_and_normals(self):
        return np.array([0, 1], dtype=np.int32), np.eye(2, 3, dtype=np.float64)

    def point_query(self, cells, var):
        self.point_calls += 1
        return np.ones(len(cells), dtype=np.float64)


def test_w6_pg_prepares_static_normals_then_aggregates_force_in_database():
    pg = _W6PGClient()
    _bench_pg_native(pg, "P", 0.005)

    assert pg.prepare_calls == 1
    assert pg.force_calls > 0
    assert pg.point_calls == 0


class _W6IoTDBClient:
    def __init__(self):
        self.ctx = MeshContext("demo", 0, "surface")
        self.contiguous_calls = 0
        self.point_calls = 0

    def surface_cells_and_normals(self):
        return np.array([0, 1], dtype=np.int32), np.eye(2, 3, dtype=np.float64)

    def contiguous_point_query(self, cells, var):
        self.contiguous_calls += 1
        return np.ones(len(cells), dtype=np.float64)

    def point_query(self, cells, var):
        self.point_calls += 1
        return np.ones(len(cells), dtype=np.float64)


def test_w6_iotdb_cfd_uses_one_contiguous_sql_path_but_h5_keeps_point_path():
    cfd = _W6IoTDBClient()
    _bench_iotdb_native(cfd, "P", 0.005, contiguous_scalar=True)
    assert cfd.contiguous_calls > 0
    assert cfd.point_calls == 0

    h5 = _W6IoTDBClient()
    _bench_iotdb_native(h5, "U", 0.005, contiguous_scalar=False)
    assert h5.point_calls > 0
    assert h5.contiguous_calls == 0


class _VTKArray:
    def GetRange(self):
        return (0.0, 1.0)


class _VTKCellData:
    def GetArray(self, name):
        return _VTKArray()


class _VTKMesh:
    def GetCellData(self):
        return _VTKCellData()


class _VTKW8Client:
    def __init__(self, seen):
        self.vtk_mesh = _VTKMesh()
        self.seen = seen

    def var_value_range(self, var, step=None):
        self.seen.append(("range", step))
        return (0.0, 1.0)

    def range_query_var(self, lo, hi, var, step=None):
        self.seen.append(("query", step))
        return np.zeros((0,), dtype=np.int32)

    def close(self):
        pass


class _Cfg:
    vtk_dir = "unused"
    duration_sec = 0.001
    progress = False
    progress_interval_sec = 5.0

    def fluid_zone(self, ship):
        return "0_Fluid"

    def valid_variables(self, ship):
        return ["P"]


def test_vtk_w8_keeps_v19_file_backed_baseline_path(monkeypatch):
    seen = []
    monkeypatch.setattr(w8_run, "make_vtk", lambda *a, **k: _VTKW8Client(seen))

    w8_run.run_ship_step(_Cfg(), "demo", 400, {"vtk"})

    assert seen
    assert {kind for kind, _step in seen} == {"range", "query"}
    assert {_step for _kind, _step in seen} == {400}
