from __future__ import annotations

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.infra.iotdb.mesh_runtime import MeshRuntime as IoTMeshRuntime
from cfd_bench.mesh_ops.geometry_ops import (
    _line_intersects_bbox,
    _plane_hits_bbox,
    cells_in_coordinate_range,
    iotdb_line_intersection,
    iotdb_plane_intersection,
    iotdb_point_intersection,
)
from cfd_bench.workloads.common.geom_resolver import cell_count


def _mesh(n=200):
    cells = {}
    for cid in range(n):
        x = float(cid % 20)
        y = float((cid // 20) % 10)
        z = float(cid // 200)
        cells[cid] = (x + 0.4, y + 0.4, z + 0.4, x, x + 0.8, y, y + 0.8, z, z + 0.8, 4)
    data = RuntimeMeshData(cells=cells)
    IoTMeshRuntime._build_spatial_index(data)
    return data


def test_runtime_bbox_view_is_cached_not_rebuilt_per_access():
    data = _mesh(20)
    first = data.cell_bbox
    second = data.cell_bbox
    assert first is second


def test_vectorized_coordinate_range_matches_bbox_semantics():
    data = _mesh(200)
    lo = np.array([3.0, 2.0, -1.0])
    hi = np.array([8.8, 6.8, 2.0])
    expected = [
        cid for cid, row in data.cells.items()
        if row[3] >= lo[0] and row[4] <= hi[0]
        and row[5] >= lo[1] and row[6] <= hi[1]
        and row[7] >= lo[2] and row[8] <= hi[2]
    ]
    assert cells_in_coordinate_range(data, lo, hi).tolist() == expected


def test_vectorized_line_and_plane_match_scalar_reference():
    data = _mesh(200)
    p0 = np.array([-1.0, 2.4, 0.4])
    p1 = np.array([21.0, 2.4, 0.4])
    expected_line = []
    for cid, row in data.cells.items():
        bb = (row[3], row[4], row[5], row[6], row[7], row[8])
        ok, t = _line_intersects_bbox(p0, p1, bb)
        if ok:
            expected_line.append((t, cid))
    expected_line.sort()
    assert iotdb_line_intersection(data, p0, p1).tolist() == [cid for _, cid in expected_line]

    origin = np.array([5.3, 0.0, 0.0])
    normal = np.array([1.0, 0.3, -0.2])
    expected_plane = []
    for cid, row in data.cells.items():
        bb = (row[3], row[4], row[5], row[6], row[7], row[8])
        if _plane_hits_bbox(origin, normal, bb):
            expected_plane.append(cid)
    assert iotdb_plane_intersection(data, origin, normal).tolist() == expected_plane


def test_point_miss_outside_global_bbox_does_not_materialize_bbox_dict():
    data = _mesh(200)
    assert data._cell_bbox_cache is None
    hits = iotdb_point_intersection(data, np.array([[1e6, 1e6, 1e6]], dtype=np.float64))
    assert hits.size == 0
    assert data._cell_bbox_cache is None


class _Cfg:
    host = "perf-test-host"
    port = "6667"
    root_path = "root.perf_test"


class _Repo:
    config = _Cfg()

    def __init__(self):
        self.fetch_count = 0

    def fetch_cells(self, dataset, zone):
        self.fetch_count += 1
        return _mesh(5).cells


def test_iotdb_static_runtime_is_reused_across_clients():
    dataset = "unique_perf_cache_dataset"
    r1 = _Repo()
    r2 = _Repo()
    m1 = IoTMeshRuntime(r1)
    m2 = IoTMeshRuntime(r2)
    assert len(m1.ensure_cells(dataset, "0_Fluid").cells) == 5
    assert len(m2.ensure_cells(dataset, "0_Fluid").cells) == 5
    assert r1.fetch_count == 1
    assert r2.fetch_count == 0


class _CountOnlyClient:
    def __init__(self):
        self.runtime = type("R", (), {"ensure_cells": lambda *args: (_ for _ in ()).throw(AssertionError("must not load mesh"))})()
        self.ctx = type("C", (), {"dataset_key": "d", "zone": "z"})()

    def get_cell_count(self):
        return 123456


def test_cell_count_prefers_metadata_api_over_runtime_materialization():
    assert cell_count(_CountOnlyClient()) == 123456
