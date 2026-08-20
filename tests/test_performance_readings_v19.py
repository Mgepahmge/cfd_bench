from __future__ import annotations

import numpy as np
import pandas as pd
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


def test_iotdb_numeric_query_uses_dataframe_batches_and_large_fetch_size():
    class _DFDataSet:
        def __init__(self):
            self.frames = [
                pd.DataFrame({"Time": [10, 11], "root.demo.P": [1.5, 2.5]}),
                pd.DataFrame({"Time": [12], "root.demo.P": [3.5]}),
            ]
            self.index = 0
            self.fetch_size = None
            self.closed = False

        def set_fetch_size(self, value):
            self.fetch_size = int(value)

        def has_next_df(self):
            return self.index < len(self.frames)

        def next_df(self):
            frame = self.frames[self.index]
            self.index += 1
            return frame

        def close_operation_handle(self):
            self.closed = True

    class _DFSession:
        def __init__(self):
            self.ds = None

        def execute_query_statement(self, sql):
            self.ds = _DFDataSet()
            return self.ds

    repo = IoTDBRepository(IoTDBConfig())
    session = _DFSession()
    repo.session = session
    timestamps, values = repo.query_numeric_arrays("SELECT P FROM root.demo", 1)

    assert timestamps.tolist() == [10, 11, 12]
    assert values[:, 0].tolist() == [1.5, 2.5, 3.5]
    assert session.ds.fetch_size == 50000
    assert session.ds.closed


def test_iotdb_dense_cell_selection_uses_one_bounded_time_scan(monkeypatch):
    repo = IoTDBRepository(IoTDBConfig())
    repo.resolve_cell_var_path = lambda *args, **kwargs: "root.demo.step_200.cell_vars"
    seen = []

    def fake_query(sql, value_count):
        seen.append(sql)
        # Return the complete enclosing range; repository must filter it back
        # to the exact requested IDs and restore request order.
        ts = np.arange(0, 999, dtype=np.int64)
        return ts, ts.astype(np.float64).reshape(-1, 1)

    monkeypatch.setattr(repo, "query_numeric_arrays", fake_query)
    ids = np.arange(0, 999, 2, dtype=np.int32)
    requested = ids[::-1]
    values = repo.fetch_cell_scalar_values("demo", 200, "P", requested)

    assert len(seen) == 1
    assert "Time >= 0 AND Time <= 998" in seen[0]
    assert "Time IN" not in seen[0]
    assert np.array_equal(values, requested.astype(np.float64))


def test_iotdb_contiguous_runs_are_compacted_instead_of_large_in_lists():
    repo = IoTDBRepository(IoTDBConfig())
    ids = []
    for block in range(100):
        base = block * 20
        ids.extend(range(base, base + 10))

    predicates = repo._selection_predicates(ids)

    assert len(predicates) == 2
    assert all(not needs_filter for _predicate, needs_filter in predicates)
    assert all("Time IN" not in predicate for predicate, _ in predicates)
    assert sum(predicate.count("Time >=") for predicate, _ in predicates) == 100


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
