import time

import numpy as np
import pytest

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.infra.postgresql import spatial as pg_spatial
from cfd_bench.mesh_ops.geometry_ops import iotdb_point_intersection
from cfd_bench.workloads.common.random_geom import random_start_point
from cfd_bench.workloads.w5.run import _integrate_one_streamline


def test_postgresql_point_intersection_does_not_snap_miss_to_nearest(monkeypatch):
    monkeypatch.setattr(pg_spatial, "_bucket_candidates", lambda *args, **kwargs: [])
    centroids = {7: (0.0, 0.0, 0.0)}

    hits = pg_spatial.point_intersection(
        None,
        "dataset",
        "default",
        "0_Fluid",
        np.array([[100.0, 100.0, 100.0]], dtype=np.float64),
        centroids=centroids,
    )

    assert hits.dtype == np.int32
    assert hits.size == 0


def test_iotdb_tiledb_shared_point_intersection_omits_miss_sentinel():
    data = RuntimeMeshData(
        cells={0: (0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)},
        all_cell_ids=np.array([0], dtype=np.int32),
        all_bbox_min=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        all_bbox_max=np.array([[1.0, 1.0, 1.0]], dtype=np.float64),
    )

    hits = iotdb_point_intersection(
        data,
        np.array([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]], dtype=np.float64),
    )

    assert hits.tolist() == [0]
    assert -1 not in hits


def test_w5_streamline_stops_when_point_leaves_mesh():
    scalar_calls = []

    def scalar_fn(cid):
        scalar_calls.append(cid)
        return 1.0, 0.0, 0.0

    def intersect_fn(points):
        return np.array([], dtype=np.int32)

    completed = _integrate_one_streamline(
        scalar_fn,
        intersect_fn,
        0,
        np.array([0.0, 0.0, 0.0]),
        deadline=time.monotonic() + 1.0,
        delta_t=1.0,
    )

    assert completed is True
    assert scalar_calls == [0]


def test_w5_streamline_has_independent_max_step_guard():
    scalar_calls = 0

    def scalar_fn(cid):
        nonlocal scalar_calls
        scalar_calls += 1
        return 1.0, 0.0, 0.0

    def intersect_fn(points):
        return np.array([0], dtype=np.int32)

    completed = _integrate_one_streamline(
        scalar_fn,
        intersect_fn,
        0,
        np.array([0.0, 0.0, 0.0]),
        deadline=time.monotonic() + 10.0,
        delta_t=1.0,
        max_steps=3,
    )

    assert completed is True
    assert scalar_calls == 3


def test_w5_streamline_honors_global_deadline_before_query():
    def scalar_fn(cid):
        raise AssertionError("scalar query must not run after deadline")

    completed = _integrate_one_streamline(
        scalar_fn,
        lambda points: np.array([0], dtype=np.int32),
        0,
        np.array([0.0, 0.0, 0.0]),
        deadline=time.monotonic() - 1.0,
        delta_t=1.0,
    )

    assert completed is False


def test_random_start_point_can_expire_on_mesh_without_hits():
    with pytest.raises(TimeoutError):
        random_start_point(
            lambda points: np.array([], dtype=np.int32),
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            deadline=time.monotonic() - 1.0,
        )
