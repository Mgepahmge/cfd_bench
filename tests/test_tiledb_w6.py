from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from cfd_bench.core.context import MeshContext
from cfd_bench.core.runtime_mesh import RuntimeMeshData


def _load_client_class(monkeypatch):
    # TileDB-Py is optional in the test environment. The client methods under
    # test use a fake repository/runtime, so a module stub is sufficient.
    if "tiledb" not in sys.modules:
        fake = types.ModuleType("tiledb")
        fake.Ctx = object
        monkeypatch.setitem(sys.modules, "tiledb", fake)
    mod = importlib.import_module("cfd_bench.API.tiledb_api.client")
    return mod.TileDBMeshClient


class _StructuralRepo:
    def list_mesh_static_zones(self, dataset_key):
        return ["0_Fluid"]

    def fetch_boundary_faces(self, dataset_key, zone):
        return []

    def list_cell_variables(self, dataset_key, step, zone):
        return ["U", "V", "W"]


class _StructuralRuntime:
    def __init__(self):
        self.data = RuntimeMeshData()
        self.data.cells = {
            0: (0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }
        self.data.nodes = {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)}
        self.data.cell_nodes = {0: [0, 1]}

    def ensure_cells(self, dataset_key, zone):
        return self.data

    def ensure_cell_nodes(self, dataset_key, zone):
        return self.data


class _CFDRepo:
    def list_mesh_static_zones(self, dataset_key):
        return ["0_Fluid", "0_Wall_hull"]

    def fetch_boundary_faces(self, dataset_key, zone):
        if zone == "0_Wall_hull":
            return [(7, 0.0, 1.0, 0.0, 2.0), (7, 0.0, 1.0, 0.0, 1.0)]
        return []

    def list_cell_variables(self, dataset_key, step, zone):
        return ["P", "U", "V", "W"] if zone == "0_Wall_hull" else ["U", "V", "W"]


class _UnusedRuntime:
    def ensure_cells(self, *args, **kwargs):
        raise AssertionError("boundary-face CFD path should not need runtime cells")

    def ensure_cell_nodes(self, *args, **kwargs):
        raise AssertionError("boundary-face CFD path should not need runtime topology")


def _client(cls, repo, runtime, zone):
    client = cls.__new__(cls)
    client.repo = repo
    client.runtime = runtime
    client.ctx = MeshContext(dataset_key="case", step=0, zone=zone)
    return client


def test_tiledb_w6_structural_only_fluid_zone_falls_back_and_uses_available_scalar(monkeypatch):
    cls = _load_client_class(monkeypatch)
    client = _client(cls, _StructuralRepo(), _StructuralRuntime(), "0_Fluid")

    assert client.w6_zone_candidates("case", preferred_zone="0_Fluid", hull_hint="0_Wall_hull") == ["0_Fluid"]
    assert client.resolve_hull_zone("case") == "0_Fluid"
    assert client.resolve_w6_scalar(["P", "U", "V", "W"]) == "U"

    cells, normals = client.surface_cells_and_normals()
    assert cells.tolist() == [0]
    assert normals.shape == (1, 3)
    assert np.isfinite(normals).all()
    assert np.isclose(np.linalg.norm(normals[0]), 1.0)


def test_tiledb_w6_cfd_prefers_hull_boundary_faces_and_pressure(monkeypatch):
    cls = _load_client_class(monkeypatch)
    repo = _CFDRepo()
    client = _client(cls, repo, _UnusedRuntime(), "0_Wall_hull")

    zones = client.w6_zone_candidates("case", preferred_zone="0_Fluid", hull_hint="0_Wall_hull")
    assert zones[0] == "0_Wall_hull"
    assert "0_Fluid" in zones
    assert client.resolve_w6_scalar(["P", "U", "V", "W"]) == "P"

    cells, normals = client.surface_cells_and_normals()
    assert cells.tolist() == [7]
    assert normals.shape == (1, 3)
    assert np.allclose(normals[0], [0.0, 1.0, 0.0])
