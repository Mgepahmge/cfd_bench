from __future__ import annotations

import numpy as np
import pytest

from cfd_bench.core.context import MeshContext
from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.infra.iotdb.repository import IoTDBRepository
from cfd_bench.workloads.w6.run import _bench_iotdb_native, _bench_pg_native
from cfd_bench.workloads.w8 import run as w8_run


class _Field:
    def __init__(self, value):
        self.value = value

    def get_string_value(self):
        return str(self.value)


class _Row:
    def __init__(self, timestamp, value):
        self.timestamp = int(timestamp)
        self.value = value

    def get_timestamp(self):
        return self.timestamp

    def get_fields(self):
        return [_Field(self.value)]


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


class _RawSession:
    def __init__(self):
        self.calls = []

    def execute_raw_data_query(self, paths, start, end):
        self.calls.append((list(paths), int(start), int(end)))
        return _DataSet(_Row(i, i * 0.5) for i in range(start, end))


def test_iotdb_large_bulk_scalar_uses_one_raw_time_window():
    repo = IoTDBRepository(IoTDBConfig())
    session = _RawSession()
    repo.session = session
    repo.resolve_cell_var_path = lambda *args, **kwargs: "root.demo.step_200.cell_vars"

    ids = np.arange(20_000, dtype=np.int32)
    values = repo.fetch_cell_scalar_values_bulk("demo", 200, "P", ids)

    assert session.calls == [(["root.demo.step_200.cell_vars.P"], 0, 20_000)]
    np.testing.assert_allclose(values[[0, 123, -1]], [0.0, 61.5, 9999.5])


class _PGCursor:
    def __init__(self, owner):
        self.owner = owner
        self.rows = []

    def execute(self, sql, params):
        self.owner.sql = sql
        self.owner.params = params
        lo, hi = int(params[-2]), int(params[-1])
        self.rows = [(i, float(i) + 0.25) for i in range(lo, hi + 1)]

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


class _PGConn:
    def __init__(self):
        self.sql = ""
        self.params = None

    def cursor(self):
        return _PGCursor(self)


class _PGInner:
    def __init__(self):
        self.conn = _PGConn()
        self.timestep = 200


def test_postgresql_contiguous_bulk_scalar_uses_primary_key_range_scan():
    pytest.importorskip("psycopg2")
    from cfd_bench.API.postgresql_api.client import PostgreSQLMeshClient
    from cfd_bench.core.context import DatasetKey

    client = PostgreSQLMeshClient()
    client._inner = _PGInner()
    client._key = DatasetKey("Kvlcc", "351K_Small", "0_Symmetry_sym", 200)
    client.ctx = MeshContext("Kvlcc_351K_Small", 200, "0_Symmetry_sym")

    values = client.bulk_point_query(np.arange(8, dtype=np.int32), "P")

    assert "cell_id BETWEEN" in client._inner.conn.sql
    assert "ANY(" not in client._inner.conn.sql
    np.testing.assert_allclose(values, np.arange(8, dtype=np.float64) + 0.25)


class _W6Client:
    def __init__(self):
        self.ctx = MeshContext("demo", 0, "surface")
        self.bulk_calls = 0
        self.point_calls = 0

    def surface_cells_and_normals(self):
        return np.array([0, 1], dtype=np.int32), np.eye(2, 3, dtype=np.float64)

    def bulk_point_query(self, cells, var):
        self.bulk_calls += 1
        return np.ones(len(cells), dtype=np.float64)

    def point_query(self, cells, var):
        self.point_calls += 1
        return np.ones(len(cells), dtype=np.float64)


def test_w6_pg_uses_bulk_scalar_but_iotdb_h5_can_keep_point_path():
    pg = _W6Client()
    _bench_pg_native(pg, "P", 0.01)
    assert pg.bulk_calls > 0
    assert pg.point_calls == 0

    h5 = _W6Client()
    _bench_iotdb_native(h5, "U", 0.01, bulk_scalar=False)
    assert h5.point_calls > 0
    assert h5.bulk_calls == 0


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


def test_vtk_w8_passes_explicit_step_so_baseline_query_reloads_frame(monkeypatch):
    seen = []
    monkeypatch.setattr(w8_run, "make_vtk", lambda *a, **k: _VTKW8Client(seen))

    w8_run.run_ship_step(_Cfg(), "demo", 400, {"vtk"})

    assert seen
    assert {kind for kind, _step in seen} == {"range", "query"}
    assert {_step for _kind, _step in seen} == {400}
