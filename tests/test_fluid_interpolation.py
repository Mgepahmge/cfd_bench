from __future__ import annotations

import numpy as np
import pytest

from cfd_bench.features.fluid_interpolation import FluidInterpolationEngine, find_linear_support


def test_find_linear_support_reproduces_affine_field_in_tetrahedron():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.asarray([0.2, 0.3, 0.1], dtype=np.float64)
    support = find_linear_support(target, vertices)
    assert support is not None
    assert np.all(support.weights >= -1e-12)
    assert np.isclose(np.sum(support.weights), 1.0)
    assert support.reconstruction_error < 1e-12

    nodal = 4.0 + vertices[:, 0] + 2.0 * vertices[:, 1] + 3.0 * vertices[:, 2]
    interpolated = float(np.dot(support.weights, nodal[list(support.local_indices)]))
    expected = 4.0 + target[0] + 2.0 * target[1] + 3.0 * target[2]
    assert np.isclose(interpolated, expected)


def test_find_linear_support_cube_is_independent_of_vertex_order():
    cube = np.asarray(
        [
            [1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0],
            [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0],
        ],
        dtype=np.float64,
    )
    target = np.asarray([0.31, 0.47, 0.62], dtype=np.float64)
    support = find_linear_support(target, cube)
    assert support is not None
    selected = cube[list(support.local_indices)]
    assert np.linalg.norm(support.weights @ selected - target) < 1e-10
    assert np.min(support.weights) >= -1e-9


class _FakeRepo:
    def __init__(self):
        self.config = type("Cfg", (), {"host": "fake", "port": "0", "root_path": "root.fake"})()
        self.nodes = {
            0: (0.0, 0.0, 0.0),
            1: (1.0, 0.0, 0.0),
            2: (0.0, 1.0, 0.0),
            3: (0.0, 0.0, 1.0),
        }
        self.cell_nodes = {0: [0, 1, 2, 3]}

    def cfd_dataset_metadata(self, dataset):
        return {
            "is_cfd": True,
            "zone": "0_Fluid",
            "zones": ("0_Fluid",),
            "variables": ("P", "U"),
            "timesteps": (200,),
            "node_count": 4,
            "cell_count": 1,
        }

    def fetch_mesh_meta(self, dataset, zone):
        return {"node_count": 4.0, "cell_count": 1.0}

    def fetch_cells_arrays(self, dataset, zone):
        ids = np.asarray([0], dtype=np.int32)
        centers = np.asarray([[0.25, 0.25, 0.25]], dtype=np.float64)
        mins = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        maxs = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64)
        types = np.asarray([4], dtype=np.int32)
        return ids, centers, mins, maxs, types

    def fetch_cell_nodes_subset(self, dataset, zone, cell_ids):
        return {int(cid): list(self.cell_nodes[int(cid)]) for cid in cell_ids}

    def fetch_nodes_subset(self, dataset, zone, node_ids):
        return {int(nid): self.nodes[int(nid)] for nid in node_ids}

    def fetch_cell_nodes(self, dataset, zone):
        return {k: list(v) for k, v in self.cell_nodes.items()}

    def fetch_cell_scalar_values(self, dataset, step, var, cell_ids, zone="0_Fluid"):
        value = {"P": 7.5, "U": -2.0}[str(var).upper()]
        return np.full((len(cell_ids),), value, dtype=np.float64)


def test_engine_locates_cell_projects_vertices_and_interpolates_without_ingest_changes():
    engine = FluidInterpolationEngine(_FakeRepo())
    result = engine.interpolate(
        "tiny_cfd", 200, [0.2, 0.2, 0.2], variables=["P", "U"]
    )
    assert result.cell_id == 0
    assert result.source_element_id == 1
    assert set(result.support_source_node_ids).issubset({1, 2, 3, 4})
    assert np.isclose(result.values["P"], 7.5)
    assert np.isclose(result.values["U"], -2.0)
    assert result.reconstruction_error < 1e-10
    assert result.vertex_value_source == "mean of incident cell-centered CFD values"


def test_engine_rejects_point_outside_mesh():
    engine = FluidInterpolationEngine(_FakeRepo())
    with pytest.raises(ValueError, match="outside the mesh AABB"):
        engine.interpolate("tiny_cfd", 200, [2.0, 2.0, 2.0], variables=["P"])


def test_cli_registers_interpolate_without_changing_run_command():
    from cfd_bench.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "interpolate",
            "--datasets", "tiny_cfd",
            "--step", "200",
            "--point", "0.1", "0.2", "0.3",
            "--variables", "U", "P",
        ]
    )
    assert args.datasets == ["tiny_cfd"]
    assert args.step == 200
    assert args.point == [[0.1, 0.2, 0.3]]
    assert args.variables == ["U", "P"]
