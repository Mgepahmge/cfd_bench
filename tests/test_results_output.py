from __future__ import annotations

import csv
import warnings

from cfd_bench.core.results import csv_result_output, emit_benchmark_result, result_context


def test_csv_result_output_is_structured_and_console_line_is_unchanged(tmp_path, capsys):
    path = tmp_path / "results.csv"
    with csv_result_output(str(path)):
        with result_context("w1", "Kvlcc_351K_Small", 200):
            emit_benchmark_result(
                "PG point intersection: 25 txns in 10s",
                backend="PG",
                operation="point_intersection",
                transactions=25,
                duration_sec=10.0,
            )

    assert capsys.readouterr().out.strip() == "PG point intersection: 25 txns in 10s"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "Kvlcc_351K_Small"
    assert row["workload"] == "w1"
    assert row["backend"] == "postgresql"
    assert row["operation"] == "point_intersection"
    assert row["step"] == "200"
    assert row["transactions"] == "25"
    assert row["duration_sec"] == "10"
    assert float(row["txns_per_sec"]) == 2.5


def test_no_csv_output_does_not_create_file_or_change_print(capsys):
    with csv_result_output(None):
        with result_context("w8", "beam", 0):
            emit_benchmark_result(
                "TileDB W8: 7 txns in 1.0s",
                backend="TileDB",
                operation="variable_range",
                transactions=7,
                duration_sec=1.0,
            )
    assert capsys.readouterr().out.strip() == "TileDB W8: 7 txns in 1.0s"


def test_run_cli_exposes_optional_csv_output(tmp_path):
    from cfd_bench.cli.main import build_parser

    out = tmp_path / "bench.csv"
    args = build_parser().parse_args(
        ["run", "--datasets", "beam", "--output", str(out)]
    )
    assert args.output == str(out)


def test_iotdb_discovery_warning_filter_side_effect_is_isolated(monkeypatch):
    from cfd_bench.cli.main import build_parser
    from cfd_bench.infra.iotdb import discovery
    from cfd_bench.workloads.common.cli import workload_config_from_args

    before = list(warnings.filters)

    def fake_discover(selected):
        warnings.simplefilter("always", DeprecationWarning)
        return []

    monkeypatch.setattr(discovery, "discover_iotdb_datasets", fake_discover)
    args = build_parser().parse_args(
        ["run", "--datasets", "beam", "--backend", "iotdb"]
    )
    workload_config_from_args(args)
    assert warnings.filters == before


def test_iotdb_connect_warning_filter_side_effect_is_isolated(monkeypatch):
    from cfd_bench.API.iotdb_api import client as client_module
    from cfd_bench.workloads.common.backends import make_iotdb

    before = list(warnings.filters)

    class FakeClient:
        def connect(self, ship, step, zone):
            warnings.simplefilter("always", DeprecationWarning)

    monkeypatch.setattr(client_module, "IoTDBMeshClient", FakeClient)
    result = make_iotdb("beam", 0, "0_Fluid")
    assert isinstance(result, FakeClient)
    assert warnings.filters == before
