from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

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
