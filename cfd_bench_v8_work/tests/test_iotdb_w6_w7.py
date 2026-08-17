import numpy as np

from cfd_bench.API.iotdb_api.client import IoTDBMeshClient
from cfd_bench.core.context import MeshContext
from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.infra.iotdb.repository import IoTDBRepository


class _BoundaryRepo:
    def __init__(self, rows):
        self.rows = rows

    def fetch_boundary_faces(self, dataset, zone):
        return list(self.rows)


class _FallbackRepo:
    def fetch_boundary_faces(self, dataset, zone):
        return []


class _Runtime:
    def __init__(self, data):
        self.data = data

    def ensure_cells(self, dataset, zone):
        return self.data

    def ensure_cell_nodes(self, dataset, zone):
        return self.data

    def ensure_adjacency(self, dataset, zone):
        return self.data


class _W7Repo:
    def __init__(self, velocities):
        self.velocities = velocities
        self.requested = None

    def fetch_velocity_map(self, dataset, step, ids, zone="0_Fluid"):
        self.requested = list(ids)
        return {int(i): self.velocities[int(i)] for i in ids if int(i) in self.velocities}


def _client_with_ctx():
    client = IoTDBMeshClient(IoTDBConfig())
    client.ctx = MeshContext(dataset_key="demo", step=0, zone="0_Fluid")
    return client


def test_iotdb_boundary_normals_preserve_owner_cell_ids():
    client = _client_with_ctx()
    client.repo = _BoundaryRepo([
        (7, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (7, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (9, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ])
    ids, normals = client.surface_cells_and_normals()
    assert ids.tolist() == [7, 9]
    np.testing.assert_allclose(normals[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(normals[1], [0.0, 1.0, 0.0])


def test_iotdb_w6_topology_fallback_handles_beam_elements():
    data = RuntimeMeshData(
        cells={
            0: (0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2),
            1: (1.5, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2),
        },
        nodes={0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0)},
        cell_nodes={0: [0, 1], 1: [1, 2]},
    )
    client = _client_with_ctx()
    client.repo = _FallbackRepo()
    client.runtime = _Runtime(data)
    ids, normals = client.surface_cells_and_normals()
    assert ids.tolist() == [0, 1]
    assert normals.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), [1.0, 1.0])


def test_iotdb_w7_fetches_adjacency_halo_and_returns_low_dimensional_q():
    data = RuntimeMeshData(
        cells={
            0: (0.0, 0.0, 0.0, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 2),
            1: (1.0, 0.0, 0.0, 0.9, 1.1, 0.0, 0.0, 0.0, 0.0, 2),
            2: (2.0, 0.0, 0.0, 1.9, 2.1, 0.0, 0.0, 0.0, 0.0, 2),
        },
        adjacency={0: [1], 1: [0, 2], 2: [1]},
    )
    repo = _W7Repo({0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0)})
    client = _client_with_ctx()
    client.repo = repo
    client.runtime = _Runtime(data)
    client.range_query_coord = lambda lo, hi: np.asarray([1], dtype=np.int32)

    ids, q = client.compute_qcriterion_roi([0, 0, 0], [2, 0, 0])
    assert repo.requested == [0, 1, 2]
    assert ids.tolist() == [1]
    assert q.shape == (1,)
    assert np.isfinite(q[0])


def test_iotdb_boundary_face_reader_uses_persisted_cell_id_and_patch_code():
    repo = IoTDBRepository(IoTDBConfig())
    seen = []

    def query_rows(sql):
        seen.append(sql)
        # Time=123 is face-row id; persisted cell_id is 8.
        return [(123, ["8", "0", "1", "0", "0", "2.5", "3", "4", "5"])]

    repo.query_rows = query_rows
    rows = repo.fetch_boundary_faces("demo", "0_Wall_hull")
    assert "cell_id,patch_code" in seen[0]
    assert rows[0][0] == 8
    assert rows[0][5] == 2.5


class _SplitLoadRuntime:
    """Mimic the real MeshRuntime lazy-loading contract."""
    def __init__(self):
        self.data = RuntimeMeshData()
        self.ensure_cells_called = False

    def ensure_cells(self, dataset, zone):
        self.ensure_cells_called = True
        self.data.cells = {
            0: (0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2),
            1: (1.5, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2),
        }
        return self.data

    def ensure_cell_nodes(self, dataset, zone):
        self.data.nodes = {
            0: (0.0, 0.0, 0.0),
            1: (1.0, 0.0, 0.0),
            2: (2.0, 0.0, 0.0),
        }
        self.data.cell_nodes = {0: [0, 1], 1: [1, 2]}
        return self.data


def test_iotdb_w6_fallback_explicitly_loads_cells_like_real_runtime():
    client = _client_with_ctx()
    client.repo = _FallbackRepo()
    runtime = _SplitLoadRuntime()
    client.runtime = runtime

    ids, normals = client.surface_cells_and_normals()

    assert runtime.ensure_cells_called
    assert ids.tolist() == [0, 1]
    assert normals.shape == (2, 3)

class _ScalarRepo:
    def __init__(self, available):
        self.available = set(available)

    def is_h5_dataset(self, dataset):
        return True

    def h5_dataset_metadata(self, dataset):
        return {"common_variables": tuple(sorted(self.available))}

    def resolve_cell_var_path(self, dataset, step, zone="0_Fluid", probe_var="P"):
        if probe_var not in self.available:
            raise RuntimeError("missing")
        return "root.demo.vars"

    def query_rows(self, sql):
        for var in self.available:
            if f"SELECT {var} " in sql:
                return [(0, ["1.0"])]
        return []


def test_iotdb_w6_prefers_pressure_but_falls_back_to_available_structural_scalar():
    client = _client_with_ctx()
    client.repo = _ScalarRepo({"U", "V", "W"})
    assert client.resolve_w6_scalar(["P", "U", "V", "W"]) == "U"


def test_iotdb_w7_recovers_regular_3d_qcriterion_from_halo():
    cells = {
        0: (0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 4),
        1: (1.0, 0.0, 0.0, 1, 1, 0, 0, 0, 0, 4),
        2: (-1.0, 0.0, 0.0, -1, -1, 0, 0, 0, 0, 4),
        3: (0.0, 1.0, 0.0, 0, 0, 1, 1, 0, 0, 4),
        4: (0.0, -1.0, 0.0, 0, 0, -1, -1, 0, 0, 4),
        5: (0.0, 0.0, 1.0, 0, 0, 0, 0, 1, 1, 4),
        6: (0.0, 0.0, -1.0, 0, 0, 0, 0, -1, -1, 4),
    }
    data = RuntimeMeshData(cells=cells, adjacency={0: [1, 2, 3, 4, 5, 6]})
    velocities = {}
    for cid, row in cells.items():
        x, y, z = row[:3]
        velocities[cid] = (-y, x, 0.0)
    repo = _W7Repo(velocities)
    client = _client_with_ctx()
    client.repo = repo
    client.runtime = _Runtime(data)
    client.range_query_coord = lambda lo, hi: np.asarray([0], dtype=np.int32)

    ids, q = client.compute_qcriterion_roi([-1, -1, -1], [1, 1, 1])
    assert ids.tolist() == [0]
    np.testing.assert_allclose(q, [1.0], atol=1e-12)
