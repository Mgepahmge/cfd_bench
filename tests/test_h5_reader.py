from pathlib import Path

import h5py
import numpy as np
import pytest

from cfd_bench.ingest.h5.artifacts import write_max_diff_files
from cfd_bench.ingest.h5.canonical import build_canonical_frame, field_to_cells, field_to_nodes
from cfd_bench.ingest.h5.postgresql import build_ingest_plan
from cfd_bench.ingest.h5.reader import OdbH5Reader


def _attr(group, name, value):
    if isinstance(value, str):
        group.attrs[name] = np.array([value], dtype=h5py.string_dtype())
    else:
        group.attrs[name] = np.array([value])


def _make_case(path: Path, modal: bool = False) -> Path:
    with h5py.File(path, "w") as h5:
        part = h5.create_group("Parts/PART-1")
        nodes = part.create_group("Nodes")
        nodes.create_dataset("Labels", data=np.array([1, 2, 3], dtype=np.int32))
        nodes.create_dataset(
            "Coordinates",
            data=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        )
        block = part.create_group("Elements/ElementClass:0")
        _attr(block, "ElementType", "B33")
        _attr(block, "SectionCategory", "beam<RECT Profile>")
        block.create_dataset("Labels", data=np.array([1, 2], dtype=np.int32))
        # Connectivity is zero-based node row index, not source label.
        block.create_dataset("Connectivities", data=np.array([[0, 1], [1, 2]], dtype=np.int32))

        inst = h5.create_group("Assembly/Instances/PART-1-1")
        _attr(inst, "PartName", "PART-1")
        _attr(inst, "Dependent", 1)

        step = h5.create_group("Steps/Step-1")
        _attr(step, "Index", 1)
        _attr(step, "Domain", 0 if not modal else 2)
        frames = step.create_group("Frames")

        def add_u(frame, scale):
            ug = frame.create_group("U")
            ug.attrs["ComponentLabels"] = np.array(
                ["U1", "U2", "U3"], dtype=h5py.string_dtype()
            )
            _attr(ug, "Position", 1)
            _attr(ug, "Type", 3)
            ui = ug.create_group("PART-1-1")
            ui.create_dataset(
                "Real",
                data=np.array(
                    [[0, 0, 0], [0, -scale, 0], [0, -2 * scale, 0]],
                    dtype=np.float32,
                ),
            )

        if modal:
            base = frames.create_group("Frame:0")
            _attr(base, "Inc/Mode", 0)
            _attr(base, "Time/Freq", 0.0)
            _attr(base, "Description", "Base State")
            for i, freq in [(1, 10.0), (2, 20.0)]:
                frame = frames.create_group(f"Frame:{i}")
                _attr(frame, "Inc/Mode", i)
                _attr(frame, "Time/Freq", freq)
                add_u(frame, float(i))
        else:
            frame = frames.create_group("Frame:0")
            _attr(frame, "Inc/Mode", 1)
            _attr(frame, "Time/Freq", 1.0)
            add_u(frame, 1.0)

            eg = frame.create_group("E")
            eg.attrs["ComponentLabels"] = np.array(
                ["E11", "E22", "E12"], dtype=h5py.string_dtype()
            )
            _attr(eg, "Position", 3)
            _attr(eg, "Type", 7)
            eb = eg.create_group("PART-1-1/ElementClass:0/LocationIndex:1")
            # 2 elements x 2 samples per element x 3 components.
            eb.create_dataset(
                "Real",
                data=np.array(
                    [[1, 0, 0], [3, 0, 0], [5, 0, 0], [7, 0, 0]],
                    dtype=np.float32,
                ),
            )

            sg = frame.create_group("S")
            sg.attrs["ComponentLabels"] = np.array(
                ["S11", "S22", "S12"], dtype=h5py.string_dtype()
            )
            _attr(sg, "Position", 3)
            sb = sg.create_group("PART-1-1/ElementClass:0/LocationIndex:1")
            sb.create_dataset(
                "Real",
                data=np.array(
                    [[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]],
                    dtype=np.float32,
                ),
            )
    return path


def test_static_mesh_and_connectivity(tmp_path):
    path = _make_case(tmp_path / "static.h5")
    reader = OdbH5Reader(str(path))
    mesh = reader.load_mesh()
    assert mesh.instance_name == "PART-1-1"
    assert mesh.part_name == "PART-1"
    assert mesh.node_count == 3
    assert mesh.cell_count == 2
    assert set(mesh.cell_element_types) == {"B33"}
    assert mesh.cell_node_ids[0].tolist() == [0, 1]
    assert mesh.source_node_labels.tolist() == [1, 2, 3]
    assert mesh.cell_adjacency == [[1], [0]]


def test_static_field_mapping_to_benchmark_cells(tmp_path):
    path = _make_case(tmp_path / "static.h5")
    reader = OdbH5Reader(str(path))
    mesh = reader.load_mesh()
    frame = next(reader.iter_frames())
    canonical = build_canonical_frame(reader, mesh, frame, timestep=0)
    assert set(canonical.cell_scalars) == {"U", "V", "W", "E"}
    assert np.allclose(canonical.cell_scalars["U"], 0.0)
    assert np.allclose(canonical.cell_scalars["V"], [-0.5, -1.5])
    # E defaults to component magnitude, then averages the 2 samples/element.
    assert np.allclose(canonical.cell_scalars["E"], [2.0, 6.0])


def test_element_integration_field_can_select_component(tmp_path):
    path = _make_case(tmp_path / "static.h5")
    reader = OdbH5Reader(str(path))
    mesh = reader.load_mesh()
    frame = next(reader.iter_frames())
    s11 = field_to_cells(reader, mesh, frame, "S", component="S11")
    assert np.allclose(s11, [15.0, 35.0])


def test_modal_base_frame_is_skipped_and_modes_map_sequentially(tmp_path):
    path = _make_case(tmp_path / "modal.h5", modal=True)
    plan, mesh, frames = build_ingest_plan(str(path))
    assert mesh.cell_count == 2
    assert plan.frame_count == 2
    assert plan.mapped_timesteps == (0, 1)
    assert plan.skipped_frames == ("Step-1/Frame:0",)
    assert set(plan.mapped_variables) == {"U", "V", "W"}
    assert frames[0].info.inc_or_mode == 1
    assert frames[0].info.time_or_frequency == pytest.approx(10.0)


def test_max_diff_files_match_mapped_variables(tmp_path):
    path = _make_case(tmp_path / "static.h5")
    plan, mesh, frames = build_ingest_plan(str(path))
    out = tmp_path / "max"
    paths = write_max_diff_files(str(out), "beam_static", mesh, frames)
    assert len(paths) == 1
    text = Path(paths[0]).read_text()
    assert "variable,max_diff" in text
    for var in ["U", "V", "W", "E"]:
        assert f"{var}," in text


def test_explicit_mapping_augments_auto_mapping(tmp_path):
    path = _make_case(tmp_path / "static.h5")
    reader = OdbH5Reader(str(path))
    mesh = reader.load_mesh()
    frame = next(reader.iter_frames())
    canonical = build_canonical_frame(
        reader,
        mesh,
        frame,
        timestep=0,
        explicit_mapping={"P": ("S", "S11")},
    )
    assert set(canonical.cell_scalars) == {"U", "V", "W", "E", "P"}
    assert np.allclose(canonical.cell_scalars["P"], [15.0, 35.0])


def test_db_max_diff_rows_are_derived_from_canonical_frames(tmp_path):
    from cfd_bench.ingest.h5.postgresql import _max_diff_rows

    path = _make_case(tmp_path / "static.h5")
    _, mesh, frames = build_ingest_plan(str(path))
    rows = list(_max_diff_rows(mesh, frames, "beam", "static", "0_Fluid"))
    values = {(step, var): diff for _, _, _, step, var, diff in rows}
    assert set(var for _, var in values) == {"U", "V", "W", "E"}
    assert values[(0, "V")] == pytest.approx(1.0)
    assert values[(0, "E")] == pytest.approx(4.0)


def test_nodal_values_are_preserved_for_h5_workloads(tmp_path):
    path = _make_case(tmp_path / "static.h5")
    reader = OdbH5Reader(str(path))
    mesh = reader.load_mesh()
    frame = next(reader.iter_frames())
    canonical = build_canonical_frame(reader, mesh, frame, timestep=0)
    assert set(canonical.node_scalars) == {"U", "V", "W"}
    assert np.allclose(canonical.node_scalars["V"], [0.0, -1.0, -2.0])
    assert np.allclose(field_to_nodes(reader, mesh, frame, "U", component="U2"), [0.0, -1.0, -2.0])
    assert canonical.cell_scalars["V"].tolist() == pytest.approx([-0.5, -1.5])


def test_ingest_plan_reports_nodal_variables(tmp_path):
    path = _make_case(tmp_path / "modal.h5", modal=True)
    plan, _, frames = build_ingest_plan(str(path))
    assert plan.mapped_node_variables == ("U", "V", "W")
    assert all(set(frame.node_scalars) == {"U", "V", "W"} for frame in frames)


def test_insert_frames_persists_direct_nodal_values(tmp_path, monkeypatch):
    from cfd_bench.ingest.h5 import postgresql as h5pg

    path = _make_case(tmp_path / "static.h5")
    _, _, frames = build_ingest_plan(str(path))
    captured = {}

    def fake_batch_insert(cur, table, columns, rows, batch_size=5000):
        captured[table] = (list(columns), list(rows))

    monkeypatch.setattr(h5pg, "_batch_insert", fake_batch_insert)
    h5pg._insert_frames(object(), frames, "beam", "static", "0_Fluid")

    assert "cell_scalar" in captured
    assert "node_scalar" in captured
    node_rows = captured["node_scalar"][1]
    # 3 nodes x U/V/W for the one static frame.
    assert len(node_rows) == 9
    v_rows = [row for row in node_rows if row[4] == "V"]
    assert [row[6] for row in v_rows] == pytest.approx([0.0, -1.0, -2.0])


def test_c3d10_adjacency_requires_a_complete_quadratic_face():
    from cfd_bench.ingest.h5.reader import _build_adjacency

    # Cell 1 shares the complete quadratic face (1,2,3,5,6,7) with cell 0.
    # Cell 2 only shares edge (1,2,5), which the old >=3-node rule wrongly
    # classified as a face neighbor.
    cells = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
        np.array([0, 1, 2, 10, 4, 5, 6, 11, 12, 13], dtype=np.int64),
        np.array([0, 1, 20, 21, 4, 22, 23, 24, 25, 26], dtype=np.int64),
    ]
    adjacency = _build_adjacency(cells, ["C3D10", "C3D10", "C3D10"])
    assert adjacency == [[1], [0], []]


def test_c3d10_spatial_shell_uses_four_corner_nodes():
    from cfd_bench.ingest.postgresql.build_cell_geom_full import _surface_geometry

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    surface_points, simplices = _surface_geometry(points, "C3D10")
    assert surface_points.shape == (4, 3)
    assert simplices.shape == (4, 3)
    assert np.allclose(surface_points, points[:4])
