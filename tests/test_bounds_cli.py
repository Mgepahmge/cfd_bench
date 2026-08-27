from __future__ import annotations

import json

import pytest

from cfd_bench.cli.bounds_cmd import _build_payload


def _mesh_meta():
    return {
        "bbox_min_x": -2.0,
        "bbox_max_x": 8.0,
        "bbox_min_y": -1.0,
        "bbox_max_y": 4.0,
        "bbox_min_z": 0.0,
        "bbox_max_z": 2.0,
    }


def test_bounds_payload_classifies_inside_and_outside_points_with_coupling_tolerance():
    payload = _build_payload(
        "case_default",
        "0_Fluid",
        _mesh_meta(),
        [
            [0.0, 0.0, 1.0],
            [8.0 + 5.0e-10, 0.0, 1.0],
            [8.0 + 2.0e-9, -2.0, 1.0],
        ],
    )

    assert payload["extent"] == {"x": 10.0, "y": 5.0, "z": 2.0}
    assert payload["tolerance"] == pytest.approx(1.0e-9)
    assert payload["points"][0]["inside_aabb"] is True
    assert payload["points"][1]["inside_aabb"] is True
    assert payload["points"][2]["inside_aabb"] is False
    assert payload["points"][2]["outside_axes"] == ["x+", "y-"]
    assert payload["points"][2]["distance_to_aabb"] == pytest.approx(1.0)


def test_bounds_payload_rejects_missing_or_invalid_metadata():
    with pytest.raises(ValueError, match="bbox_min_x"):
        _build_payload("bad", "0_Fluid", {}, [])

    meta = _mesh_meta()
    meta["bbox_min_x"] = 10.0
    with pytest.raises(ValueError, match="invalid bounding box"):
        _build_payload("bad", "0_Fluid", meta, [])


def test_cli_registers_bounds_and_supports_repeated_points_and_json():
    from cfd_bench.cli.main import build_parser

    args = build_parser().parse_args(
        [
            "bounds",
            "--datasets",
            "case_default",
            "--zone",
            "0_Fluid",
            "--point",
            "0",
            "1",
            "2",
            "--point",
            "3",
            "4",
            "5",
            "--json",
        ]
    )
    assert args.datasets == ["case_default"]
    assert args.zone == "0_Fluid"
    assert args.point == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert args.json is True


def test_run_bounds_reads_iotdb_metadata_without_requiring_a_step(monkeypatch, capsys):
    import cfd_bench.infra.iotdb.repository as repository_module
    from cfd_bench.cli.main import main

    opened = []
    closed = []

    class FakeRepo:
        def __init__(self, config):
            self.config = config

        def open(self):
            opened.append(True)

        def close(self):
            closed.append(True)

        def cfd_dataset_metadata(self, dataset):
            assert dataset == "case_default"
            return {"is_cfd": True, "zone": "0_Fluid"}

        def fetch_mesh_meta(self, dataset, zone):
            assert (dataset, zone) == ("case_default", "0_Fluid")
            return _mesh_meta()

    monkeypatch.setattr(repository_module, "IoTDBRepository", FakeRepo)

    rc = main(
        [
            "bounds",
            "--datasets",
            "case_default",
            "--point",
            "9",
            "0",
            "1",
            "--json",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dataset"] == "case_default"
    assert result["zone"] == "0_Fluid"
    assert result["points"][0]["inside_aabb"] is False
    assert result["points"][0]["outside_axes"] == ["x+"]
    assert opened == [True]
    assert closed == [True]
