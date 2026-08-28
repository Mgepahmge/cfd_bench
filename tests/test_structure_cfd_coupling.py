from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from cfd_bench.features.structure_cfd_coupling import (
    STATUS_NO_CONTAINING_CELL,
    STATUS_OUTSIDE_MESH,
    STATUS_PASS,
    StructureCfdCouplingEngine,
)


class _FakeCouplingRepo:
    def __init__(self):
        self.config = type(
            "Cfg",
            (),
            {"host": "fake", "port": "0", "root_path": "root.fake"},
        )()
        self.frame_reads = 0
        self.structure_reads = 0
        self.node_reads = 0
        self.connectivity_reads = 0

    def h5_dataset_metadata(self, dataset):
        assert dataset == "structure"
        return {
            "is_h5": True,
            "zone": "0_Structure",
            "node_count": 4,
            "cell_count": 1,
        }

    def fetch_h5_structure_nodes(self, dataset, zone):
        self.structure_reads += 1
        assert dataset == "structure"
        assert zone == "0_Structure"
        node_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
        source_labels = np.asarray([101, 102, 103, 104], dtype=np.int64)
        coordinates = np.asarray(
            [
                [0.2, 0.2, 0.2],    # inside tetra
                [2.0, 2.0, 2.0],    # outside global mesh bbox
                [0.8, 0.8, 0.8],    # inside bbox, outside tetra
                [0.1, 0.1, 0.1],    # inside tetra
            ],
            dtype=np.float64,
        )
        return node_ids, source_labels, coordinates

    def cfd_dataset_metadata(self, dataset):
        assert dataset == "cfd"
        return {
            "is_cfd": True,
            "zone": "0_Fluid",
            "variables": ("P", "U"),
            "timesteps": (200,),
            "node_count": 4,
            "cell_count": 1,
        }

    def fetch_cells_arrays(self, dataset, zone):
        assert dataset == "cfd"
        assert zone == "0_Fluid"
        return (
            np.asarray([0], dtype=np.int32),
            np.asarray([[0.25, 0.25, 0.25]], dtype=np.float64),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
            np.asarray([4], dtype=np.int32),
        )

    def fetch_nodes_arrays(self, dataset, zone):
        self.node_reads += 1
        return (
            np.asarray([0, 1, 2, 3], dtype=np.int64),
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )

    def fetch_cell_nodes_arrays(self, dataset, zone):
        self.connectivity_reads += 1
        return (
            np.asarray([0], dtype=np.int64),
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        )

    def fetch_cell_scalar_matrix(self, dataset, step, variables, cell_ids, zone="0_Fluid"):
        self.frame_reads += 1
        assert dataset == "cfd"
        assert step == 200
        assert list(variables) == ["P", "U"]
        assert list(cell_ids) == [0]
        return np.asarray([[7.5, -2.0]], dtype=np.float64)


def test_coupling_maps_all_structure_nodes_and_writes_independent_h5(tmp_path):
    repo = _FakeCouplingRepo()
    output = tmp_path / "coupling.h5"
    engine = StructureCfdCouplingEngine(repo)
    summary = engine.couple_to_h5(
        structure_dataset="structure",
        cfd_dataset="cfd",
        cfd_step=200,
        output_path=output,
        variables=["P", "U"],
        batch_size=2,
        diagnostics=True,
        progress=False,
    )

    assert output.is_file()
    assert not (tmp_path / "coupling.h5.partial").exists()
    assert summary.node_count == 4
    assert summary.success_count == 2
    assert summary.outside_count == 1
    assert summary.no_containing_cell_count == 1
    assert summary.failed_count == 0

    # Performance contract: DB preparation is independent of structure point count.
    assert repo.structure_reads == 1
    assert repo.node_reads == 1
    assert repo.connectivity_reads == 1
    assert repo.frame_reads == 1

    with h5py.File(output, "r") as h5:
        assert h5.attrs["format"] == "cfd-bench-structure-cfd-coupling-v1"
        np.testing.assert_array_equal(h5["nodes/node_id"][:], [0, 1, 2, 3])
        np.testing.assert_array_equal(h5["nodes/source_node_label"][:], [101, 102, 103, 104])
        status = h5["diagnostics/status"][:]
        np.testing.assert_array_equal(
            status,
            [STATUS_PASS, STATUS_OUTSIDE_MESH, STATUS_NO_CONTAINING_CELL, STATUS_PASS],
        )
        p = h5["values/P"][:]
        u = h5["values/U"][:]
        assert np.isclose(p[0], 7.5) and np.isclose(p[3], 7.5)
        assert np.isclose(u[0], -2.0) and np.isclose(u[3], -2.0)
        assert np.isnan(p[1]) and np.isnan(p[2])
        assert h5["diagnostics/support_node_ids"].shape == (4, 4)
        assert h5["diagnostics/weights"].shape == (4, 4)
        meta = h5["metadata"].attrs
        assert meta["structure_dataset"] == "structure"
        assert meta["cfd_dataset"] == "cfd"
        assert int(meta["cfd_step"]) == 200
        assert int(meta["success_count"]) == 2


def test_coupling_cli_is_registered_without_changing_existing_commands():
    from cfd_bench.cli.main import build_parser

    args = build_parser().parse_args(
        [
            "couple",
            "--structure-dataset", "structure",
            "--cfd-dataset", "cfd",
            "--cfd-step", "200",
            "--output", "result.h5",
            "--variables", "P", "U",
            "--batch-size", "2048",
            "--diagnostics",
        ]
    )
    assert args.command == "couple"
    assert args.structure_dataset == "structure"
    assert args.cfd_dataset == "cfd"
    assert args.cfd_step == 200
    assert args.variables == ["P", "U"]
    assert args.batch_size == 2048
    assert args.diagnostics is True


def test_csv_export_is_separate_script(tmp_path):
    repo = _FakeCouplingRepo()
    source = tmp_path / "coupling.h5"
    StructureCfdCouplingEngine(repo).couple_to_h5(
        structure_dataset="structure",
        cfd_dataset="cfd",
        cfd_step=200,
        output_path=source,
        variables=["P", "U"],
        progress=False,
    )
    output = tmp_path / "coupling.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "coupling_h5_to_csv.py"
    subprocess.run(
        [sys.executable, str(script), str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    with output.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["source_node_label"] == "101"
    assert rows[0]["status"] == "PASS"
    assert rows[1]["status"] == "OUTSIDE_MESH"
    assert rows[2]["status"] == "NO_CONTAINING_CELL"
    assert rows[0]["P"] == "7.5"


def _rotation_z(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class _FakeAlignedCouplingRepo(_FakeCouplingRepo):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(123)
        # Random points strictly inside the unit tetrahedron.  The asymmetric
        # distribution avoids exact hull symmetries in the alignment test.
        raw = rng.gamma(shape=1.5, scale=1.0, size=(240, 4))
        bary = raw / np.sum(raw, axis=1, keepdims=True)
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.target_points = bary @ vertices
        self.true_scale = 0.0025
        self.true_rotation = _rotation_z(11.0)
        self.true_translation = np.asarray([2.4, -0.7, 0.35], dtype=np.float64)
        self.structure_points = (
            (self.target_points - self.true_translation[None, :]) @ self.true_rotation
        ) / self.true_scale

    def h5_dataset_metadata(self, dataset):
        assert dataset == "structure"
        return {
            "is_h5": True,
            "zone": "0_Structure",
            "node_count": int(self.structure_points.shape[0]),
            "cell_count": 1,
        }

    def fetch_h5_structure_nodes(self, dataset, zone):
        self.structure_reads += 1
        n = int(self.structure_points.shape[0])
        return (
            np.arange(n, dtype=np.int64),
            np.arange(1001, 1001 + n, dtype=np.int64),
            self.structure_points.copy(),
        )

    def cfd_dataset_metadata(self, dataset):
        assert dataset == "cfd"
        return {
            "is_cfd": True,
            "zone": "0_Fluid",
            "zones": ("0_Fluid", "0_Wall_hull"),
            "variables": ("P", "U"),
            "timesteps": (200,),
            "node_count": 4,
            "cell_count": 1,
        }

    def fetch_nodes_arrays(self, dataset, zone):
        self.node_reads += 1
        if zone == "0_Wall_hull":
            n = int(self.target_points.shape[0])
            return np.arange(n, dtype=np.int64), self.target_points.copy()
        return super().fetch_nodes_arrays(dataset, zone)


def test_optional_auto_alignment_recovers_uniform_transform_and_preserves_source_coordinates(tmp_path):
    repo = _FakeAlignedCouplingRepo()
    output = tmp_path / "aligned-coupling.h5"
    summary = StructureCfdCouplingEngine(repo).couple_to_h5(
        structure_dataset="structure",
        cfd_dataset="cfd",
        cfd_step=200,
        output_path=output,
        variables=["P", "U"],
        auto_align=True,
        alignment_max_points=240,
        progress=False,
    )

    assert summary.alignment_enabled is True
    assert summary.alignment_reference_zone == "0_Wall_hull"
    assert summary.alignment_scale == pytest.approx(repo.true_scale, rel=1.0e-5)
    assert summary.success_count == repo.structure_points.shape[0]
    assert summary.outside_count == 0

    with h5py.File(output, "r") as h5:
        meta = h5["metadata"].attrs
        assert bool(meta["alignment_enabled"]) is True
        assert meta["alignment_reference_zone"] == "0_Wall_hull"
        assert float(meta["alignment_scale"]) == pytest.approx(repo.true_scale, rel=1.0e-5)
        assert str(meta["coordinate_frame"]) == "similarity-aligned"
        np.testing.assert_allclose(h5["nodes/coordinates"][:], repo.structure_points)
        np.testing.assert_allclose(
            h5["nodes/coupling_coordinates"][:], repo.target_points, rtol=1.0e-5, atol=1.0e-7
        )


def test_default_coupling_keeps_auto_alignment_disabled(tmp_path):
    repo = _FakeCouplingRepo()
    output = tmp_path / "unaligned.h5"
    summary = StructureCfdCouplingEngine(repo).couple_to_h5(
        structure_dataset="structure",
        cfd_dataset="cfd",
        cfd_step=200,
        output_path=output,
        variables=["P", "U"],
        progress=False,
    )
    assert summary.alignment_enabled is False
    assert summary.alignment_scale is None
    with h5py.File(output, "r") as h5:
        assert bool(h5["metadata"].attrs["alignment_enabled"]) is False
        assert "coupling_coordinates" not in h5["nodes"]
