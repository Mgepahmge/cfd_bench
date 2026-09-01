from __future__ import annotations

import cfd_bench.ingest.orchestrator as orchestrator


def _fake_topology(*_args, **_kwargs):
    return {
        "0_Fluid": {
            "node_count": 4,
            "cell_count": 1,
        }
    }


def _capture_backend(calls, backend):
    def fake(dat_path, ship_type, scale, *args, **kwargs):
        calls.append((backend, ship_type, scale))
    return fake


def test_ingest_preserves_dataset_name_without_synthetic_default_suffix(tmp_path, monkeypatch):
    dat = tmp_path / "case.dat"
    dat.write_text("dummy", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "cfd_bench.ingest.cfd.canonical.load_cfd_topology",
        _fake_topology,
    )
    monkeypatch.setattr(orchestrator, "_ingest_postgresql", _capture_backend(calls, "postgresql"))
    monkeypatch.setattr(orchestrator, "_ingest_iotdb", _capture_backend(calls, "iotdb"))
    monkeypatch.setattr(orchestrator, "_ingest_tiledb", _capture_backend(calls, "tiledb"))

    report = orchestrator.ingest_all(
        str(dat),
        "test001",
        ["postgresql", "iotdb", "tiledb"],
        zone_indices=(0,),
        tiledb_root=str(tmp_path / "tiledb"),
        vtk_root=str(tmp_path / "vtk"),
        init_pg_schema=False,
        build_pg_spatial=False,
    )

    assert report.ok
    assert calls == [
        ("postgresql", "test001", ""),
        ("iotdb", "test001", ""),
        ("tiledb", "test001", ""),
    ]


def test_ingest_keeps_explicit_legacy_scale_suffix(tmp_path, monkeypatch):
    dat = tmp_path / "case.dat"
    dat.write_text("dummy", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "cfd_bench.ingest.cfd.canonical.load_cfd_topology",
        _fake_topology,
    )
    monkeypatch.setattr(orchestrator, "_ingest_iotdb", _capture_backend(calls, "iotdb"))

    report = orchestrator.ingest_all(
        str(dat),
        "JBC_615k",
        ["iotdb"],
        zone_indices=(0,),
        tiledb_root=str(tmp_path / "tiledb"),
        vtk_root=str(tmp_path / "vtk"),
    )

    assert report.ok
    assert calls == [("iotdb", "JBC", "615k")]
