from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")
h5py = pytest.importorskip("h5py")

from cfd_bench.API.vtk_api.client import VTKMeshClient
from cfd_bench.cli.main import build_parser, main as cli_main
from cfd_bench.infra.vtk.discovery import discover_vtk_datasets
from cfd_bench.ingest.h5.postgresql import build_ingest_plan
from cfd_bench.ingest.vtk.load_vtk import write_h5_plan_to_vtk


def _tiny_cfd_text(*, u=(1.0, 2.0), p=(7.0, 8.0)) -> str:
    x = [0, 1, 0, 0, 0]
    y = [0, 0, 1, 0, 0]
    z = [0, 0, 0, 1, -1]
    fields = [x, y, z, list(u), [3, 4], [5, 6], list(p), [9, 10], [11, 12]]
    values = "\n".join(" ".join(str(v) for v in arr) for arr in fields)
    return f'''TITLE ="tiny CFD"
VARIABLES ="X" "Y" "Z" "U" "V" "W" "P" "K" "E"
ZONE T="0_Fluid"
N=5,E=2
STRANDID=1,SOLUTIONTIME=2.0
Nodes=5,Faces=7,Elements=2,ZONETYPE=FEPolyhedron
DATAPACKING=BLOCK
VARLOCATION=([4-9]=CELLCENTERED)
TotalNumFaceNodes=21,NumConnectedBoundaryFaces=0,TotalNumBoundaryConnections=0
DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)
{values}
# node count per face
3 3 3 3 3 3 3
# face nodes
1 2 3  1 4 2  2 4 3  3 4 1  1 2 5  2 3 5  3 1 5
# left elements
1 1 1 1 2 2 2
# right elements
2 0 0 0 0 0 0
'''


def _write_cfd_case(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "200_case.dat").write_text(_tiny_cfd_text(), encoding="utf-8")
    (root / "400_case.dat").write_text(
        _tiny_cfd_text(u=(2.0, 4.0), p=(9.0, 10.0)), encoding="utf-8"
    )


def _attr(group, name: str, value) -> None:
    if isinstance(value, str):
        group.attrs[name] = np.array([value], dtype=h5py.string_dtype())
    else:
        group.attrs[name] = np.array([value])


def _write_h5_case(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        part = h5.create_group("Parts/PART-1")
        nodes = part.create_group("Nodes")
        nodes.create_dataset("Labels", data=np.array([101, 102, 103], dtype=np.int32))
        nodes.create_dataset(
            "Coordinates",
            data=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        )
        block = part.create_group("Elements/ElementClass:0")
        _attr(block, "ElementType", "B33")
        _attr(block, "SectionCategory", "beam")
        block.create_dataset("Labels", data=np.array([501, 502], dtype=np.int32))
        block.create_dataset("Connectivities", data=np.array([[0, 1], [1, 2]], dtype=np.int32))

        inst = h5.create_group("Assembly/Instances/PART-1-1")
        _attr(inst, "PartName", "PART-1")
        _attr(inst, "Dependent", 1)

        step = h5.create_group("Steps/Step-1")
        _attr(step, "Index", 1)
        _attr(step, "Domain", 0)
        frame = step.create_group("Frames/Frame:0")
        _attr(frame, "Inc/Mode", 1)
        _attr(frame, "Time/Freq", 1.0)

        ug = frame.create_group("U")
        ug.attrs["ComponentLabels"] = np.array(["U1", "U2", "U3"], dtype=h5py.string_dtype())
        _attr(ug, "Position", 1)
        _attr(ug, "Type", 3)
        ui = ug.create_group("PART-1-1")
        ui.create_dataset(
            "Real",
            data=np.array([[0, 0, 0], [0, -1, 0], [0, -2, 0]], dtype=np.float32),
        )

        eg = frame.create_group("E")
        eg.attrs["ComponentLabels"] = np.array(["E11"], dtype=h5py.string_dtype())
        _attr(eg, "Position", 3)
        _attr(eg, "Type", 1)
        eb = eg.create_group("PART-1-1/ElementClass:0/LocationIndex:1")
        eb.create_dataset("Real", data=np.array([[2.0], [6.0]], dtype=np.float32))


def test_ingest_cli_exposes_vtk_as_backend_not_include_flag():
    parser = build_parser()
    args = parser.parse_args(["ingest", "--dat", "x", "--datasets", "d", "--backends", "vtk"])
    assert args.backends == ["vtk"]
    with pytest.raises(SystemExit):
        parser.parse_args([
            "ingest", "--dat", "x", "--datasets", "d", "--backends", "vtk", "--include-vtk"
        ])


def test_vtk_cfd_backend_uses_canonical_schema_and_runtime_queries(tmp_path: Path):
    dat = tmp_path / "dat"
    root = tmp_path / "vtk"
    _write_cfd_case(dat)
    assert cli_main([
        "ingest", "--dat", str(dat), "--datasets", "tiny_cfd",
        "--backends", "vtk", "--zone-indices", "0", "--vtk-root", str(root),
    ]) == 0

    infos = discover_vtk_datasets(["tiny_cfd"], root=str(root))
    assert len(infos) == 1
    assert infos[0].timesteps == (200, 400)
    assert set(infos[0].variables) == {"U", "V", "W", "P", "K", "E"}

    client = VTKMeshClient(root_path=str(root))
    client.connect("tiny_cfd", 200, "0_Fluid")
    assert not client.is_h5_dataset()
    assert client.get_cell_count() == 2
    np.testing.assert_allclose(client.point_query([0, 1], "U"), [1.0, 2.0])
    np.testing.assert_allclose(client.velocity_query([0], step=400), [[2.0, 3.0, 5.0]])
    assert client.get_max_diffs(200)["U"] == pytest.approx(1.0)
    assert client.cfd_element_ids_in_coordinate_range([-1, -1, -2], [2, 2, 2]).tolist() == [1, 2]
    assert client.cfd_frame_statistics("P", 400)["P"]["count"] == 2
    client.prepare_cfd_point_queries()
    extrema = client.cfd_point_frame_extrema([1, 2, 3], "U")
    assert set(extrema) == {1, 2, 3}
    assert extrema[1][0] <= extrema[1][1]
    client.close()


def test_vtk_h5_backend_preserves_source_ids_and_genuine_nodal_fields(tmp_path: Path):
    h5_path = tmp_path / "beam.h5"
    root = tmp_path / "vtk"
    _write_h5_case(h5_path)
    assert cli_main([
        "ingest-h5", "--h5", str(h5_path), "--datasets", "beam",
        "--backends", "vtk", "--vtk-root", str(root), "--no-max-diffs",
    ]) == 0

    client = VTKMeshClient(root_path=str(root))
    client.connect("beam", 0, "0_Fluid")
    assert client.is_h5_dataset()
    assert set(client.h5_nodal_variables()) == {"U", "V", "W"}
    assert client.h5_point_ids().tolist() == [101, 102, 103]
    assert client.h5_element_ids_in_coordinate_range([-1, -1, -1], [3, 1, 1]).tolist() == [501, 502]
    assert client.frame_statistics("V", 0)["V"]["position"] == "node"
    assert client.frame_statistics("E", 0)["E"]["position"] == "cell"
    assert client.h5_point_frame_extrema([101, 102, 103], "V") == {
        101: (0.0, 0.0),
        102: (-1.0, -1.0),
        103: (-2.0, -2.0),
    }
    client.close()


def test_vtk_tiny_h5_all_workloads_complete(tmp_path: Path, capsys):
    h5_path = tmp_path / "beam.h5"
    root = tmp_path / "vtk"
    _write_h5_case(h5_path)
    plan, mesh, frames = build_ingest_plan(str(h5_path))
    write_h5_plan_to_vtk("beam", "0_Fluid", str(root), plan, mesh, frames)

    rc = cli_main([
        "run", "--backend", "vtk",
        "--workloads", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11",
        "--steps", "0", "--duration", "0.005",
        "--datasets", "beam", "--vtk-dir", str(root), "--zone-fluid", "0_Fluid",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    for wid in range(1, 12):
        assert f"Running w{wid}" in out
    assert "VTK W11:" in out
