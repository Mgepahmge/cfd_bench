from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cfd_bench.ingest.cfd.canonical import (
    iter_cfd_frames,
    load_cfd_topology,
    max_neighbor_diffs,
)
from cfd_bench.ingest.decoder import CAE_Decoder
from cfd_bench.ingest.common.dat_files import iter_dat_files
from cfd_bench.workloads.common.random_geom import random_start_point
from cfd_bench.workloads.w1.run import _run_point_queries
from cfd_bench.workloads.w2.run import _bench as w2_bench
from cfd_bench.workloads.w7.run import _bench_db as w7_bench


def _dat_text(*, u=(1.0, 2.0), p=(7.0, 8.0)) -> str:
    # Two tetrahedra sharing face (1,2,3).  This intentionally uses the same
    # compact + verbose count combination as the real NaViiX/Tecplot files.
    x = [0, 1, 0, 0, 0]
    y = [0, 0, 1, 0, 0]
    z = [0, 0, 0, 1, -1]
    fields = [
        x, y, z,
        list(u), [3, 4], [5, 6], list(p), [9, 10], [11, 12],
    ]
    values = "\n".join(" ".join(str(v) for v in arr) for arr in fields)
    return f'''TITLE ="tiny CFD"
VARIABLES ="X" "Y" "Z" "U" "V" "W" "P" "K" "E"
ZONE  T="0_Fluid"
N =5, E =2
 STRANDID=1, SOLUTIONTIME=2.000000
 Nodes =5, Faces=7, Elements =2, ZONETYPE=FEPolyhedron
 DATAPACKING=BLOCK
 VARLOCATION=([4-9]=CELLCENTERED)
 TotalNumFaceNodes=21, NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0
 DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)
{values}
# node count per face
3 3 3 3 3 3 3
# face nodes
1 2 3   1 4 2   2 4 3   3 4 1   1 2 5   2 3 5   3 1 5
# left elements
1 1 1 1 2 2 2
# right elements
2 0 0 0 0 0 0
'''


def _write_case(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "200_case.dat").write_text(_dat_text(), encoding="utf-8")
    (root / "400_case.dat").write_text(_dat_text(u=(2.0, 4.0), p=(9.0, 10.0)), encoding="utf-8")


def test_realistic_fepolyhedron_decoder_and_canonical_topology(tmp_path: Path):
    _write_case(tmp_path)
    dec = CAE_Decoder(3)
    dec.Decode_dat_file(str(tmp_path / "200_case.dat"), topology=True)
    assert dec.Variables == ["X", "Y", "Z", "U", "V", "W", "P", "K", "E"]
    assert len(dec.Zones) == 1
    z = dec.Zones[0]
    assert (z.Node_count, z.Element_count, z.Face_count) == (5, 2, 7)
    assert z.Zone_name == "0_Fluid"
    assert z.construct_element_adjacency() == [[1], [0]]
    assert all(np.all(np.asarray(nodes) >= 0) for nodes in z.EN.values())
    np.testing.assert_allclose(z.Element_Coordinates[0], [0.25, 0.25])
    np.testing.assert_allclose(z.Element_Coordinates[1], [0.25, 0.25])
    np.testing.assert_allclose(z.Element_Coordinates[2], [0.25, -0.25])

    topo = load_cfd_topology(str(tmp_path), zone_indices=(0,))
    fluid = topo["0_Fluid"]
    assert fluid["node_count"] == 5
    assert fluid["cell_count"] == 2
    assert fluid["max_nodes_per_cell"] == 4
    assert fluid["max_neighbors_per_cell"] == 1
    assert fluid["adjacency"] == [[1], [0]]
    assert len(fluid["boundary_faces"]) == 6
    assert len(fluid["boundary_face_nodes"]) == 6


def test_topology_false_and_frame_streaming(tmp_path: Path):
    _write_case(tmp_path)
    dec = CAE_Decoder(3)
    dec.Decode_dat_file(str(tmp_path / "200_case.dat"), topology=False)
    z = dec.Zones[0]
    assert z.FN == []
    assert z.EN == {}
    np.testing.assert_allclose(z.Element_Variables[0], [1.0, 2.0])

    frames = list(iter_cfd_frames(str(tmp_path), zone_indices=(0,)))
    assert [f.step for f in frames] == [200, 400]
    assert list(frames[0].zones[0].variables) == ["U", "V", "W", "P", "K", "E"]
    assert max_neighbor_diffs(
        load_cfd_topology(str(tmp_path), (0,))["0_Fluid"],
        frames[0].zones[0].variables,
    )["U"] == pytest.approx(1.0)


def test_dat_files_are_sorted_by_numeric_step(tmp_path: Path):
    for name in ("1000_case.dat", "200_case.dat", "400_case.dat"):
        (tmp_path / name).write_text(_dat_text(), encoding="utf-8")
    assert [Path(p).name for p in iter_dat_files(str(tmp_path))] == [
        "200_case.dat", "400_case.dat", "1000_case.dat"
    ]


def test_bounded_cfd_retry_loops_return_on_always_empty_geometry():
    bounds = [0, 1, 0, 1, 0, 1]
    empty = lambda *_args, **_kwargs: np.zeros((0,), dtype=np.int32)
    scalar = lambda *_args, **_kwargs: np.zeros((0,), dtype=np.float64)

    # duration=0.02 is long enough to exercise many empty transactions while
    # proving the inner hit-retry loops themselves are finite.
    _run_point_queries("test", empty, scalar, bounds, 0.02, ["U"], max_hit_attempts=2)
    w2_bench("test", empty, lambda *_a: np.zeros((0,)), bounds, [200], 0.02, ["U"], max_hit_attempts=2)
    w7_bench("test", empty, lambda *_a: None, bounds, 0.02, max_hit_attempts=2)


def test_random_start_point_has_optional_finite_retry():
    with pytest.raises(LookupError):
        random_start_point(
            lambda _pts: np.zeros((0,), dtype=np.int32),
            [0, 1, 0, 1, 0, 1],
            max_attempts=3,
        )


def _surface_zone_text() -> str:
    # Minimal FEPolygon-like surface zone.  Tecplot polygon faces are edges, so
    # node-count-per-face may be omitted and the decoder must infer two nodes
    # per face while reconstructing the polygon element from its edges.
    fields = [
        [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0],
        [0.0], [0.0], [0.0], [42.0], [0.0], [0.0],
    ]
    values = "\n".join(" ".join(str(v) for v in arr) for arr in fields)
    return f'''ZONE T="0_Wall_hull"
N=4,E=1
Nodes=4,Faces=4,Elements=1,ZONETYPE=FEPolygon
DATAPACKING=BLOCK
VARLOCATION=([4-9]=CELLCENTERED)
DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)
{values}
# face nodes
1 2  2 3  3 4  4 1
# left elements
1 1 1 1
# right elements
0 0 0 0
'''


def test_decoder_keeps_multiple_cfd_zones_and_surface_polygon(tmp_path: Path):
    text = _dat_text() + "\n" + _surface_zone_text()
    path = tmp_path / "200_case.dat"
    path.write_text(text, encoding="utf-8")

    dec = CAE_Decoder(3)
    dec.Decode_dat_file(str(path), topology=True)
    assert [z.Zone_name for z in dec.Zones] == ["0_Fluid", "0_Wall_hull"]
    hull = dec.Zones[1]
    assert hull.Zone_type.lower() == "fepolygon"
    assert hull.NCPF.tolist() == [2, 2, 2, 2]
    assert hull.EN[0].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(
        [hull.Element_Coordinates[0][0], hull.Element_Coordinates[1][0], hull.Element_Coordinates[2][0]],
        [0.5, 0.5, 0.0],
    )

    topo = load_cfd_topology(str(path), zone_indices=(0, 1))
    assert list(topo) == ["0_Fluid", "0_Wall_hull"]
    assert topo["0_Wall_hull"]["cell_nodes"] == [[0, 1, 2, 3]]
    # FEPolygon edges are not 3-D boundary polygons; W6 intentionally uses
    # its per-cell topology-normal fallback for this zone.
    assert topo["0_Wall_hull"]["boundary_faces"] == []

    frames = list(iter_cfd_frames(str(path), zone_indices=(0, 1)))
    assert len(frames) == 1
    assert [z.zone_name for z in frames[0].zones] == ["0_Fluid", "0_Wall_hull"]
    assert frames[0].zones[1].variables["P"].tolist() == [42.0]
