from __future__ import annotations

import numpy as np

from cfd_bench.core.cfd_nodal_projection import (
    build_node_cell_csr,
    point_frame_extrema_from_cell_values,
)
from cfd_bench.workloads.w9.run import _element_range_fn
from cfd_bench.workloads.w11.run import _select_cfd_variables, _select_variables


class _Cfg:
    def __init__(self, variables=None):
        self.variables = variables
        self.duration_sec = 1.0


class _BranchClient:
    def __init__(self, is_h5):
        self._is_h5 = bool(is_h5)

    def is_h5_dataset(self):
        return self._is_h5

    def h5_element_ids_in_coordinate_range(self, lo, hi):
        return np.asarray([101, 205], dtype=np.int64)

    def cfd_element_ids_in_coordinate_range(self, lo, hi):
        return np.asarray([1, 3], dtype=np.int64)

    def h5_nodal_variables(self):
        return ("U", "V")

    def cfd_variables(self):
        return ("U", "V", "P")


def test_w9_keeps_h5_source_label_branch_and_adds_cfd_implicit_ids():
    h5 = _BranchClient(True)
    cfd = _BranchClient(False)
    assert _element_range_fn(h5)([0, 0, 0], [1, 1, 1]).tolist() == [101, 205]
    assert _element_range_fn(cfd)([0, 0, 0], [1, 1, 1]).tolist() == [1, 3]


def test_w11_variable_selection_keeps_h5_and_cfd_contracts_separate():
    h5 = _BranchClient(True)
    cfd = _BranchClient(False)
    assert _select_variables(h5, _Cfg(), "beam") == ["U", "V"]
    assert _select_cfd_variables(cfd, _Cfg(), "ship") == ["U", "V", "P"]
    assert _select_variables(h5, _Cfg(["v"]), "beam") == ["V"]
    assert _select_cfd_variables(cfd, _Cfg(["p"]), "ship") == ["P"]


def test_cfd_runtime_projection_uses_one_based_source_nodes_and_cell_averaging():
    # Dense topology:
    # cell 0 -> nodes 0,1
    # cell 1 -> nodes 1,2
    # cell 2 -> nodes 2,3
    csr = build_node_cell_csr({0: [0, 1], 1: [1, 2], 2: [2, 3]}, node_count=4)
    by_step = {
        0: {0: 0.0, 1: 2.0, 2: 4.0},
        1: {0: 2.0, 1: 4.0, 2: 6.0},
    }

    def fetch(step, cell_ids):
        return np.asarray([by_step[int(step)][int(cid)] for cid in cell_ids], dtype=np.float64)

    # Source node 2 == dense node 1: averages cells 0,1 -> 1.0 then 3.0.
    # Source node 3 == dense node 2: averages cells 1,2 -> 3.0 then 5.0.
    got = point_frame_extrema_from_cell_values(csr, [2, 3], [0, 1], fetch)
    assert got == {2: (1.0, 3.0), 3: (3.0, 5.0)}


def test_cfd_runtime_projection_ignores_invalid_or_disconnected_source_nodes():
    csr = build_node_cell_csr({0: [0, 1]}, node_count=3)

    def fetch(step, cell_ids):
        return np.ones(len(cell_ids), dtype=np.float64)

    assert point_frame_extrema_from_cell_values(csr, [0, 3, 99], [0], fetch) == {}


def _scalar_values():
    return {
        0: {0: 0.0, 1: 2.0, 2: 4.0},
        1: {0: 2.0, 1: 4.0, 2: 6.0},
    }


class _FakeCfdRepo:
    def cfd_dataset_metadata(self, dataset_key):
        return {
            "node_count": 4,
            "cell_count": 3,
            "timesteps": (0, 1),
            "variables": ("U", "P"),
        }

    def fetch_mesh_meta(self, dataset_key, zone):
        return {"node_count": 4, "cell_count": 3}

    def fetch_cell_nodes(self, dataset_key, zone):
        return {0: [0, 1], 1: [1, 2], 2: [2, 3]}

    def fetch_cell_scalar_map(self, dataset_key, step, var, cell_ids, zone="0_Fluid"):
        values = _scalar_values()[int(step)]
        return {int(cid): values[int(cid)] for cid in cell_ids}


def test_iotdb_client_cfd_w11_uses_runtime_projection_without_h5_metadata():
    from cfd_bench.API.iotdb_api.client import IoTDBMeshClient
    from cfd_bench.core.context import MeshContext
    from cfd_bench.infra.iotdb.config import IoTDBConfig

    client = IoTDBMeshClient(IoTDBConfig())
    client.repo = _FakeCfdRepo()
    client.ctx = MeshContext(dataset_key="ship_3", step=0, zone="0_Fluid")
    client._cfd_node_cell_csr = None
    client.prepare_cfd_point_queries()
    assert list(client.cfd_point_ids()) == [1, 2, 3, 4]
    assert client.cfd_variables() == ("U", "P")
    assert client.cfd_point_frame_extrema([2, 3], "U") == {
        2: (1.0, 3.0),
        3: (3.0, 5.0),
    }


def test_tiledb_client_cfd_w11_uses_runtime_projection_without_h5_metadata():
    import pytest

    pytest.importorskip("tiledb")
    from cfd_bench.API.tiledb_api.client import TileDBMeshClient
    from cfd_bench.core.context import MeshContext
    from cfd_bench.infra.tiledb.config import TileDBConfig

    client = TileDBMeshClient(TileDBConfig(root_path="/tmp/not-used"))
    client.repo = _FakeCfdRepo()
    client.ctx = MeshContext(dataset_key="ship_3", step=0, zone="0_Fluid")
    client._cfd_node_cell_csr = None
    client.prepare_cfd_point_queries()
    assert list(client.cfd_point_ids()) == [1, 2, 3, 4]
    assert client.cfd_variables() == ("U", "P")
    assert client.cfd_point_frame_extrema([2, 3], "U") == {
        2: (1.0, 3.0),
        3: (3.0, 5.0),
    }


def test_iotdb_repository_cfd_w9_returns_one_based_element_ids():
    from cfd_bench.infra.iotdb.config import IoTDBConfig
    from cfd_bench.infra.iotdb.repository import IoTDBRepository

    repo = IoTDBRepository(IoTDBConfig())
    repo.path_mesh_static = lambda dataset, zone, leaf: "root.cells"
    seen = {}

    def query_rows(sql):
        seen["sql"] = sql
        return [(0, ["0.0"]), (2, ["0.0"])]

    repo.query_rows = query_rows
    got = repo.fetch_cfd_element_ids_in_coordinate_range(
        "ship", "0_Fluid", [0, 0, 0], [1, 1, 1]
    )
    assert got == [1, 3]
    assert "cx >= 0.0" in seen["sql"]
    assert "cz <= 1.0" in seen["sql"]


def test_iotdb_repository_cfd_w10_uses_cell_centered_metadata_variables():
    from cfd_bench.infra.iotdb.config import IoTDBConfig
    from cfd_bench.infra.iotdb.repository import IoTDBRepository

    repo = IoTDBRepository(IoTDBConfig())
    repo.cfd_dataset_metadata = lambda dataset: {"variables": ("U", "P")}
    repo.resolve_cell_var_path = lambda dataset, step, zone="0_Fluid", probe_var="P": "root.vars"

    def query_rows(sql):
        if "COUNT(U)" in sql:
            return [(0, ["3", "1", "5", "3", "1.632993"])]
        if "COUNT(P)" in sql:
            return [(0, ["3", "10", "30", "20", "8.164966"])]
        raise AssertionError(sql)

    repo.query_rows = query_rows
    got = repo.fetch_cfd_frame_statistics("ship", "0_Fluid", 200)
    assert got["U"]["position"] == "cell"
    assert got["U"]["count"] == 3
    assert got["U"]["min"] == 1.0
    assert got["P"]["max"] == 30.0


class _FakePGCursor:
    def __init__(self):
        self.sql = ""
        self.params = None

    def execute(self, sql, params=None):
        self.sql = " ".join(str(sql).split())
        self.params = params

    def fetchall(self):
        if "SELECT cell_id + 1" in self.sql:
            return [(1,), (3,)]
        if "COUNT(*) AS n" in self.sql:
            return [("P", 3, 10.0, 30.0, 20.0, 8.0), ("U", 3, 1.0, 5.0, 3.0, 1.5)]
        return []

    def fetchone(self):
        return None

    def close(self):
        pass


class _FakePGConn:
    def cursor(self):
        return _FakePGCursor()


class _FakePGInner:
    def __init__(self):
        self.conn = _FakePGConn()


def _fake_pg_client():
    import pytest

    pytest.importorskip("psycopg2")
    from cfd_bench.API.postgresql_api.client import PostgreSQLMeshClient
    from cfd_bench.core.context import DatasetKey, MeshContext

    client = PostgreSQLMeshClient()
    client._key = DatasetKey("ship", "3", zone="0_Fluid", step=200)
    client._inner = _FakePGInner()
    client.ctx = MeshContext(dataset_key="ship_3", step=200, zone="0_Fluid")
    return client


def test_postgresql_client_cfd_w9_returns_one_based_element_ids():
    client = _fake_pg_client()
    got = client.cfd_element_ids_in_coordinate_range([0, 0, 0], [1, 1, 1])
    assert got.tolist() == [1, 3]


def test_postgresql_client_cfd_w10_is_cell_centered_only():
    client = _fake_pg_client()
    got = client.cfd_frame_statistics(step=200)
    assert got["P"] == {
        "position": "cell",
        "count": 3,
        "min": 10.0,
        "max": 30.0,
        "mean": 20.0,
        "stddev": 8.0,
    }
    assert got["U"]["position"] == "cell"


class _FakeW11Client:
    def __init__(self, is_h5):
        self._is_h5 = bool(is_h5)
        self.calls = []
        self.closed = False

    def is_h5_dataset(self):
        return self._is_h5

    def h5_point_ids(self):
        self.calls.append("h5_point_ids")
        return np.asarray([10, 20, 30], dtype=np.int64)

    def h5_nodal_variables(self):
        self.calls.append("h5_nodal_variables")
        return ("U",)

    def h5_point_frame_extrema(self, ids, var):
        self.calls.append(("h5_extrema", tuple(ids), var))
        return {int(i): (0.0, 1.0) for i in ids}

    def cfd_point_ids(self):
        self.calls.append("cfd_point_ids")
        return np.asarray([1, 2, 3], dtype=np.int64)

    def cfd_variables(self):
        self.calls.append("cfd_variables")
        return ("U",)

    def prepare_cfd_point_queries(self):
        self.calls.append("prepare_cfd")

    def cfd_point_frame_extrema(self, ids, var):
        self.calls.append(("cfd_extrema", tuple(ids), var))
        return {int(i): (0.0, 1.0) for i in ids}

    def close(self):
        self.closed = True


def _one_iteration_clock(monkeypatch, module):
    values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(module.time, "time", lambda: next(values))


def test_w11_h5_branch_stays_on_original_h5_methods(monkeypatch):
    import cfd_bench.workloads.w11.run as w11

    client = _FakeW11Client(True)
    _one_iteration_clock(monkeypatch, w11)
    w11._run_client("PG", client, _Cfg(), "beam")
    assert "h5_point_ids" in client.calls
    assert "h5_nodal_variables" in client.calls
    assert any(isinstance(x, tuple) and x[0] == "h5_extrema" for x in client.calls)
    assert "cfd_point_ids" not in client.calls
    assert "prepare_cfd" not in client.calls
    assert client.closed


def test_w11_cfd_branch_uses_runtime_projection_methods(monkeypatch):
    import cfd_bench.workloads.w11.run as w11

    client = _FakeW11Client(False)
    _one_iteration_clock(monkeypatch, w11)
    w11._run_client("PG", client, _Cfg(), "ship")
    assert "cfd_point_ids" in client.calls
    assert "cfd_variables" in client.calls
    assert "prepare_cfd" in client.calls
    assert any(isinstance(x, tuple) and x[0] == "cfd_extrema" for x in client.calls)
    assert "h5_point_ids" not in client.calls
    assert client.closed
