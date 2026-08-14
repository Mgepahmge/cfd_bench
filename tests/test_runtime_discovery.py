import pytest

from cfd_bench.infra.postgresql.discovery import group_dataset_rows
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.workloads.w3.run import _find_max_diff_file



def test_postgresql_discovery_uses_common_variables_across_steps():
    rows = [
        ("beam", "static", "0_Fluid", 0, "U"),
        ("beam", "static", "0_Fluid", 0, "V"),
        ("beam", "static", "0_Fluid", 0, "E"),
        ("beam", "static", "0_Fluid", 1, "U"),
        ("beam", "static", "0_Fluid", 1, "V"),
        ("beam", "static", "0_Fluid", 1, "P"),
    ]
    info = group_dataset_rows(rows)[0]
    assert info.dataset_key == "beam_static"
    assert info.timesteps == (0, 1)
    assert info.variables == ("U", "V")
    assert info.zone_type == "0_Fluid"


def test_workload_config_prefers_discovered_metadata():
    cfg = WorkloadConfig(
        ships=["beam_static"],
        steps=None,
        variables=None,
        zone_fluid=None,
        discovered_steps={"beam_static": [0, 1]},
        discovered_variables={"beam_static": ["U", "V", "W", "E"]},
        discovered_zones={"beam_static": "RESULTS"},
    )
    assert cfg.valid_steps("beam_static") == [0, 1]
    assert cfg.valid_variables("beam_static") == ["U", "V", "W", "E"]
    assert cfg.fluid_zone("beam_static") == "RESULTS"


def test_w3_missing_sidecar_directory_is_not_an_error(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert _find_max_diff_file(str(missing), "beam_static", 0) is None


def test_run_cli_requires_explicit_datasets_and_keeps_metadata_overrides_optional():
    from cfd_bench.cli.main import build_parser

    args = build_parser().parse_args(
        ["run", "--datasets", "beam_static", "modal_case", "--duration", "0.01"]
    )
    assert args.backend == ["postgresql"]
    assert args.datasets == ["beam_static", "modal_case"]
    assert args.steps is None
    assert args.variables is None


def test_h5_ingest_requires_explicit_dataset_and_keeps_layout_overrides_optional():
    from cfd_bench.cli.main import build_parser

    args = build_parser().parse_args(
        [
            "ingest-h5",
            "--h5",
            "/tmp/example.h5",
            "--datasets",
            "beam_static",
            "--dry-run",
        ]
    )
    assert args.datasets == "beam_static"
    assert args.instance is None
    assert args.steps is None
    assert args.vector_field is None
    assert args.scalar_fields is None


def test_legacy_cfd_ingest_uses_same_datasets_option():
    from cfd_bench.cli.main import build_parser

    args = build_parser().parse_args(
        ["ingest", "--dat", "/tmp/Postprocessing", "--datasets", "JBC_615k"]
    )
    assert args.datasets == "JBC_615k"


def test_datasets_is_required_by_public_ingest_and_run_commands():
    from cfd_bench.cli.main import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "--dat", "/tmp/Postprocessing"])
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest-h5", "--h5", "/tmp/example.h5", "--dry-run"])


def test_runner_keeps_legacy_defaults_and_registers_h5_workloads():
    from cfd_bench.workloads.runner import DEFAULT_WORKLOADS, H5_ONLY_WORKLOADS

    assert DEFAULT_WORKLOADS == ("w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8")
    assert H5_ONLY_WORKLOADS == ("w9", "w10", "w11")


def test_w11_selects_only_direct_nodal_variables():
    from cfd_bench.workloads.w11.run import _select_variables

    class FakePG:
        def h5_nodal_variables(self):
            return ("U", "V", "W")

    cfg = WorkloadConfig(ships=["beam_static"], variables=None)
    assert _select_variables(FakePG(), cfg, "beam_static") == ["U", "V", "W"]

    cfg = WorkloadConfig(ships=["beam_static"], variables=["V", "E"] )
    assert _select_variables(FakePG(), cfg, "beam_static") == ["V"]


def test_w9_w10_w11_modules_have_expected_runner_hooks():
    from cfd_bench.workloads.w9 import run as w9
    from cfd_bench.workloads.w10 import run as w10
    from cfd_bench.workloads.w11 import run as w11

    assert callable(w9.run_ship)
    assert callable(w10.run_ship_step)
    assert callable(w11.run_ship)


def test_h5_ingest_backend_defaults_to_postgresql_and_can_select_iotdb():
    from cfd_bench.cli.main import build_parser

    parser = build_parser()
    default_args = parser.parse_args(
        ["ingest-h5", "--h5", "/tmp/example.h5", "--datasets", "beam_static", "--dry-run"]
    )
    assert default_args.backends == ["postgresql"]

    iotdb_args = parser.parse_args(
        [
            "ingest-h5", "--h5", "/tmp/example.h5", "--datasets", "beam_static",
            "--backends", "iotdb", "--dry-run",
        ]
    )
    assert iotdb_args.backends == ["iotdb"]


def test_workload_config_can_discover_h5_metadata_from_iotdb(monkeypatch):
    from cfd_bench.cli.main import build_parser
    from cfd_bench.infra.iotdb.discovery import IoTDBDatasetInfo
    from cfd_bench.infra.iotdb import discovery
    from cfd_bench.workloads.common.cli import workload_config_from_args

    monkeypatch.setattr(
        discovery,
        "discover_iotdb_datasets",
        lambda selected: [
            IoTDBDatasetInfo(
                dataset_key="beam_modal",
                zone_type="0_Fluid",
                timesteps=(0, 1, 2),
                variables=("U", "V", "W"),
            )
        ],
    )
    args = build_parser().parse_args(
        ["run", "--datasets", "beam_modal", "--backend", "iotdb", "--duration", "0.01"]
    )
    cfg = workload_config_from_args(args)
    assert cfg.valid_steps("beam_modal") == [0, 1, 2]
    assert cfg.valid_variables("beam_modal") == ["U", "V", "W"]
    assert cfg.fluid_zone("beam_modal") == "0_Fluid"


def test_iotdb_h5_client_exposes_w9_w11_primitives_without_importing_iotdb_driver(monkeypatch):
    import numpy as np
    from cfd_bench.API.iotdb_api.client import IoTDBMeshClient
    from cfd_bench.core.context import MeshContext

    client = IoTDBMeshClient()
    client.ctx = MeshContext(dataset_key="beam_modal", step=0, zone="0_Fluid")
    monkeypatch.setattr(client.repo, "is_h5_dataset", lambda dataset: True)
    monkeypatch.setattr(
        client.repo,
        "fetch_h5_element_ids_in_coordinate_range",
        lambda dataset, zone, lo, hi: [101, 205],
    )
    monkeypatch.setattr(
        client.repo,
        "h5_dataset_metadata",
        lambda dataset: {"common_nodal_variables": ("U", "V", "W")},
    )
    monkeypatch.setattr(client.repo, "fetch_h5_point_ids", lambda dataset, zone: [1, 7, 9])
    monkeypatch.setattr(
        client.repo,
        "fetch_h5_point_frame_extrema",
        lambda dataset, zone, ids, var: {int(ids[0]): (-2.0, 3.0)},
    )

    assert client.is_h5_dataset()
    assert client.h5_element_ids_in_coordinate_range([0, 0, 0], [1, 1, 1]).tolist() == [101, 205]
    assert client.h5_nodal_variables() == ("U", "V", "W")
    assert client.h5_point_ids().tolist() == [1, 7, 9]
    assert client.h5_point_frame_extrema([7], "V") == {7: (-2.0, 3.0)}


def test_iotdb_repository_h5_metadata_stats_and_cross_frame_extrema(monkeypatch):
    from cfd_bench.infra.iotdb.repository import IoTDBRepository
    from cfd_bench.infra.iotdb.config import IoTDBConfig

    repo = IoTDBRepository(IoTDBConfig())

    def fake_query(sql):
        if ".h5_metadata.beam.dataset_meta" in sql:
            return [(0, [
                "true", "0_Fluid", "PART-1", "PART-1-1",
                "E,U,V,W", "U,V,W", "E,U,V,W", "U,V,W",
                "B33", "3", "2",
            ])]
        if ".h5_metadata.beam.frames" in sql:
            return [(0, ["0"]), (1, ["1"])]
        if ".node_source" in sql:
            return [(0, ["10"]), (2, ["30"])]
        if ".step_0.node_vars" in sql and "COUNT(" in sql:
            return [(0, ["2", "1.0", "5.0", "3.0", "2.0"])]
        if ".step_0.node_vars" in sql:
            return [(0, ["1.0"]), (2, ["5.0"])]
        if ".step_1.node_vars" in sql:
            return [(0, ["-2.0"]), (2, ["7.0"])]
        if ".step_0.cell_vars" in sql and "COUNT(" in sql:
            return [(0, ["2", "2.0", "6.0", "4.0", "2.0"])]
        raise AssertionError(sql)

    monkeypatch.setattr(repo, "query_rows", fake_query)
    meta = repo.h5_dataset_metadata("beam")
    assert meta["common_variables"] == ("E", "U", "V", "W")
    assert repo.h5_frame_timesteps("beam") == [0, 1]
    assert repo.fetch_h5_point_frame_extrema("beam", "0_Fluid", [10, 30], "V") == {
        10: (-2.0, 1.0),
        30: (5.0, 7.0),
    }
    stats = repo.fetch_frame_statistics("beam", "0_Fluid", 0, "V")
    assert stats["V"]["position"] == "node"
    assert stats["V"]["mean"] == pytest.approx(3.0)
