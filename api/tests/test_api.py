from __future__ import annotations

import csv
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cfd_bench.cli.main import build_parser
from cfd_bench_api.commands import (
    build_benchmark_command,
    build_cfd_ingest_command,
    build_h5_ingest_command,
)
from cfd_bench_api.config import ApiConfig
from cfd_bench_api.jobs import ExecutionGate, JobManager
from cfd_bench_api.main import create_app
from cfd_bench_api.schemas import (
    BenchmarkRequest,
    CfdIngestRequest,
    H5IngestRequest,
    InterpolationPointResult,
    InterpolationResponse,
)
from cfd_bench_api.state import StateStore


@pytest.fixture
def cfg(tmp_path):
    return ApiConfig(
        data_root=tmp_path / "api-data",
        cfd_bench_executable="cfd-bench",
        recommended_chunk_size=4,
        max_chunk_size=8,
        scheduler_poll_sec=0.02,
        cancel_grace_sec=0.05,
        server_ingest_roots=(tmp_path / "share",),
    )


@pytest.fixture
def app(cfg):
    return create_app(cfg, start_worker=False)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def _create_completed_upload(client: TestClient, fmt: str, name: str, content: bytes):
    response = client.post(
        "/api/v1/uploads",
        json={"format": fmt, "files": [{"name": name, "size_bytes": len(content)}]},
    )
    assert response.status_code == 201, response.text
    upload = response.json()
    file_id = upload["files"][0]["file_id"]
    response = client.patch(
        f"/api/v1/uploads/{upload['upload_id']}/files/{file_id}",
        headers={"Upload-Offset": "0", "Content-Type": "application/octet-stream"},
        content=content,
    )
    assert response.status_code == 204, response.text
    response = client.post(f"/api/v1/uploads/{upload['upload_id']}/complete")
    assert response.status_code == 200, response.text
    return response.json()


def test_command_builders_are_accepted_by_existing_cli(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CFD_BENCH_IOTDB_HOST", "iotdb")
    monkeypatch.setenv("CFD_BENCH_IOTDB_PORT", "6667")
    parser = build_parser()

    cfd = CfdIngestRequest(
        upload_id="upl_x",
        dataset="JBC_615k",
        backends=["postgresql", "iotdb", "vtk"],
        zone_indices=[0, 1],
        init_pg_schema=False,
        build_pg_spatial=True,
    )
    cfd_cmd = build_cfd_ingest_command(cfg, cfd, tmp_path / "files")
    cfd_args = parser.parse_args(cfd_cmd[1:])
    assert cfd_args.command == "ingest"
    assert cfd_args.datasets == "JBC_615k"
    assert cfd_args.iotdb_host == "iotdb"
    assert cfd_args.init_pg_schema is False

    h5 = H5IngestRequest(
        upload_id="upl_h5",
        dataset="beam_modal",
        backends=["postgresql", "tiledb"],
        instance="PART-1-1",
        steps=["Step-1"],
        vector_field="U",
        scalar_fields=["P", "E"],
        field_mappings={"P": "S.S11"},
        timestep_mode="frame-index",
        include_empty_frames=True,
        init_schema=False,
        build_spatial=False,
        write_max_diffs=False,
    )
    h5_cmd = build_h5_ingest_command(cfg, h5, tmp_path / "beam.h5")
    h5_args = parser.parse_args(h5_cmd[1:])
    assert h5_args.command == "ingest-h5"
    assert h5_args.backends == ["postgresql", "tiledb"]
    assert h5_args.map == ["P=S.S11"]
    assert h5_args.no_init_schema is True
    assert h5_args.no_build_spatial is True
    assert h5_args.no_max_diffs is True

    bench = BenchmarkRequest(
        datasets=["JBC_615k"],
        workloads=["w1", "w6", "w11"],
        backends=["iotdb"],
        duration_sec=3.5,
        geom_engine="vtk",
        steps=[200, 400],
        variables=["U", "P"],
        zone_fluid="0_Fluid",
        progress=True,
        progress_interval_sec=2.0,
    )
    bench_cmd = build_benchmark_command(cfg, bench, tmp_path / "results.csv")
    bench_args = parser.parse_args(bench_cmd[1:])
    assert bench_args.command == "run"
    assert bench_args.workloads == ["w1", "w6", "w11"]
    assert bench_args.backend == ["iotdb"]
    assert bench_args.geom_engine == "vtk"
    assert bench_args.output == str(tmp_path / "results.csv")



def test_benchmark_default_duration_is_five_seconds():
    request = BenchmarkRequest(datasets=["beam"])
    assert request.duration_sec == 5.0


def test_cancel_waiting_benchmark_stays_queued_and_never_starts(tmp_path):
    cfg = ApiConfig(
        data_root=tmp_path / "state",
        cfd_bench_executable=sys.executable,
        scheduler_poll_sec=0.01,
        cancel_grace_sec=0.05,
    )
    cfg.ensure_directories()
    store = StateStore(cfg.state_db)
    gate = ExecutionGate()
    manager = JobManager(cfg, store, gate)
    sentinel = tmp_path / "should-not-run.txt"

    assert gate.try_begin_interactive_read()
    manager.start()
    try:
        job = manager.create_job(
            job_type="benchmark",
            dataset="beam",
            upload_id=None,
            request={},
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).write_text('ran')",
            ],
        )

        # Give the worker time to reach ExecutionGate. The regression was that
        # the worker claimed the row first, leaving `running + pid=NULL` here.
        time.sleep(0.05)
        waiting = store.get_job(job["job_id"])
        assert waiting["status"] == "queued"
        assert waiting["pid"] is None

        cancelled = manager.cancel(job["job_id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["cancel_requested"] is True

        gate.end_interactive_read()
        time.sleep(0.1)
        assert not sentinel.exists()
        assert store.get_job(job["job_id"])["status"] == "cancelled"
    finally:
        # Safe even if the assertion above failed before releasing the gate.
        gate.end_interactive_read()
        manager.stop()


def test_cancel_claimed_job_without_pid_converges_immediately(tmp_path):
    cfg = ApiConfig(
        data_root=tmp_path / "state",
        cfd_bench_executable=sys.executable,
        scheduler_poll_sec=0.01,
        cancel_grace_sec=0.05,
    )
    cfg.ensure_directories()
    store = StateStore(cfg.state_db)
    gate = ExecutionGate()
    manager = JobManager(cfg, store, gate)

    job = manager.create_job(
        job_type="benchmark",
        dataset="beam",
        upload_id=None,
        request={},
        command=[sys.executable, "-c", "raise SystemExit('must not execute')"],
    )
    assert store.claim_job(job["job_id"])
    claimed = store.get_job(job["job_id"])
    assert claimed["status"] == "running"
    assert claimed["pid"] is None

    cancelled = manager.cancel(job["job_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert cancelled["pid"] is None
    assert cancelled["finished_at"] is not None


def test_cancel_falls_back_to_persisted_pid_when_popen_handle_is_missing(tmp_path):
    cfg = ApiConfig(
        data_root=tmp_path / "state",
        cfd_bench_executable=sys.executable,
        scheduler_poll_sec=0.01,
        cancel_grace_sec=0.05,
    )
    cfg.ensure_directories()
    store = StateStore(cfg.state_db)
    gate = ExecutionGate()
    manager = JobManager(cfg, store, gate)
    command = [sys.executable, "-c", "import time; time.sleep(30)"]

    job = manager.create_job(
        job_type="benchmark",
        dataset="beam",
        upload_id=None,
        request={},
        command=command,
    )
    assert store.claim_job(job["job_id"])
    process = subprocess.Popen(command, start_new_session=True)
    try:
        store.set_job_pid(job["job_id"], process.pid)
        # Intentionally do not populate manager._processes: this simulates the
        # persisted-PID / missing-in-memory-handle recovery path.
        response = manager.cancel(job["job_id"])
        assert response["cancel_requested"] is True

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if store.get_job(job["job_id"])["status"] == "cancelled":
                break
            time.sleep(0.01)
        assert store.get_job(job["job_id"])["status"] == "cancelled"

        deadline = time.time() + 2.0
        while time.time() < deadline and process.poll() is None:
            time.sleep(0.01)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)

def test_resumable_upload_and_offset_conflict(client, cfg):
    response = client.post(
        "/api/v1/uploads",
        json={
            "format": "cfd-dat",
            "files": [{"name": "200.dat", "size_bytes": 8}],
        },
    )
    assert response.status_code == 201
    upload = response.json()
    assert upload["chunk_size"] == 4
    file_id = upload["files"][0]["file_id"]
    upload_id = upload["upload_id"]

    first = client.patch(
        f"/api/v1/uploads/{upload_id}/files/{file_id}",
        headers={"Upload-Offset": "0"},
        content=b"abcd",
    )
    assert first.status_code == 204
    assert first.headers["Upload-Offset"] == "4"

    head = client.head(f"/api/v1/uploads/{upload_id}/files/{file_id}")
    assert head.status_code == 204
    assert head.headers["Upload-Offset"] == "4"
    assert head.headers["Upload-Length"] == "8"

    wrong = client.patch(
        f"/api/v1/uploads/{upload_id}/files/{file_id}",
        headers={"Upload-Offset": "0"},
        content=b"zz",
    )
    assert wrong.status_code == 409
    assert wrong.headers["Upload-Offset"] == "4"

    second = client.patch(
        f"/api/v1/uploads/{upload_id}/files/{file_id}",
        headers={"Upload-Offset": "4"},
        content=b"efgh",
    )
    assert second.status_code == 204
    complete = client.post(f"/api/v1/uploads/{upload_id}/complete")
    assert complete.status_code == 200
    assert complete.json()["status"] == "completed"

    staged = cfg.uploads_root / upload_id / "files" / "200.dat"
    assert staged.read_bytes() == b"abcdefgh"


def test_oversized_chunk_rolls_back_staging_file(client, cfg):
    response = client.post(
        "/api/v1/uploads",
        json={"format": "cfd-dat", "files": [{"name": "200.dat", "size_bytes": 20}]},
    )
    upload = response.json()
    file_id = upload["files"][0]["file_id"]
    upload_id = upload["upload_id"]

    response = client.patch(
        f"/api/v1/uploads/{upload_id}/files/{file_id}",
        headers={"Upload-Offset": "0"},
        content=b"123456789",
    )
    assert response.status_code == 413
    state = client.get(f"/api/v1/uploads/{upload_id}").json()
    assert state["files"][0]["offset_bytes"] == 0
    staged = cfg.uploads_root / upload_id / "files" / "200.dat"
    assert staged.stat().st_size == 0


def test_upload_validation_rejects_paths_and_wrong_formats(client):
    bad_path = client.post(
        "/api/v1/uploads",
        json={"format": "cfd-dat", "files": [{"name": "../200.dat", "size_bytes": 1}]},
    )
    assert bad_path.status_code == 422

    bad_h5 = client.post(
        "/api/v1/uploads",
        json={
            "format": "h5",
            "files": [
                {"name": "a.h5", "size_bytes": 1},
                {"name": "b.h5", "size_bytes": 1},
            ],
        },
    )
    assert bad_h5.status_code == 422


def test_ingest_job_uses_completed_upload_and_existing_cli_contract(client, app):
    upload = _create_completed_upload(client, "cfd-dat", "200.dat", b"x")
    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "cfd-dat",
            "upload_id": upload["upload_id"],
            "dataset": "JBC_615k",
            "backends": ["iotdb", "vtk"],
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["type"] == "ingest"
    assert body["status"] == "queued"

    job = app.state.store.get_job(body["job_id"])
    assert job is not None
    args = build_parser().parse_args(job["command"][1:])
    assert args.command == "ingest"
    assert args.datasets == "JBC_615k"
    assert args.backends == ["iotdb", "vtk"]
    assert Path(args.dat).name == "files"



def test_server_path_cfd_ingest_uses_allowed_share_without_upload(client, app, cfg):
    case_dir = cfg.server_ingest_roots[0] / "cases" / "Kvlcc2"
    case_dir.mkdir(parents=True)
    (case_dir / "200.dat").write_text("dummy", encoding="utf-8")
    (case_dir / "400.dat").write_text("dummy", encoding="utf-8")

    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "cfd-dat",
            "server_path": str(case_dir),
            "dataset": "Kvlcc2_351k",
            "backends": ["iotdb"],
            "zone_indices": [0, 1],
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["upload_id"] is None

    job = app.state.store.get_job(body["job_id"])
    assert job is not None
    args = build_parser().parse_args(job["command"][1:])
    assert args.command == "ingest"
    assert Path(args.dat) == case_dir.resolve()
    assert args.datasets == "Kvlcc2_351k"


def test_server_path_ingest_rejects_paths_outside_allowed_roots(client, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "200.dat").write_text("dummy", encoding="utf-8")

    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "cfd-dat",
            "server_path": str(outside),
            "dataset": "forbidden",
            "backends": ["iotdb"],
        },
    )
    assert response.status_code == 403, response.text


def test_server_path_ingest_rejects_symlink_escape(client, cfg, tmp_path):
    outside = tmp_path / "outside-symlink"
    outside.mkdir()
    (outside / "200.dat").write_text("dummy", encoding="utf-8")
    share = cfg.server_ingest_roots[0]
    share.mkdir(parents=True, exist_ok=True)
    link = share / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable in this test environment")

    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "cfd-dat",
            "server_path": str(link),
            "dataset": "forbidden",
            "backends": ["iotdb"],
        },
    )
    assert response.status_code == 403, response.text


def test_server_path_h5_ingest_accepts_allowed_file(client, app, cfg):
    share = cfg.server_ingest_roots[0]
    share.mkdir(parents=True, exist_ok=True)
    source = share / "beam.h5"
    source.write_bytes(b"not-read-by-api-route")

    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "h5",
            "server_path": str(source),
            "dataset": "beam",
            "backends": ["postgresql"],
        },
    )
    assert response.status_code == 202, response.text
    job = app.state.store.get_job(response.json()["job_id"])
    args = build_parser().parse_args(job["command"][1:])
    assert args.command == "ingest-h5"
    assert Path(args.h5) == source.resolve()


def test_ingest_request_rejects_missing_or_ambiguous_source(client, cfg):
    missing = client.post(
        "/api/v1/ingests",
        json={"format": "cfd-dat", "dataset": "beam", "backends": ["iotdb"]},
    )
    assert missing.status_code == 422

    share = cfg.server_ingest_roots[0]
    share.mkdir(parents=True, exist_ok=True)
    (share / "200.dat").write_text("dummy", encoding="utf-8")
    both = client.post(
        "/api/v1/ingests",
        json={
            "format": "cfd-dat",
            "upload_id": "upl_x",
            "server_path": str(share),
            "dataset": "beam",
            "backends": ["iotdb"],
        },
    )
    assert both.status_code == 422

def test_h5_ingest_requires_h5_upload(client):
    upload = _create_completed_upload(client, "cfd-dat", "200.dat", b"x")
    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "h5",
            "upload_id": upload["upload_id"],
            "dataset": "beam",
        },
    )
    assert response.status_code == 409


def test_benchmark_csv_is_canonical_and_json_is_only_a_view(client, app):
    response = client.post(
        "/api/v1/benchmarks",
        json={
            "datasets": ["beam"],
            "workloads": ["w1"],
            "backends": ["postgresql"],
            "duration_sec": 1,
        },
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    job = app.state.store.get_job(job_id)
    assert job is not None
    result = Path(job["result_csv"])
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id", "timestamp_utc", "dataset", "workload", "backend",
                "operation", "step", "transactions", "duration_sec", "txns_per_sec", "details",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "abc",
                "timestamp_utc": "2026-08-25T00:00:00+00:00",
                "dataset": "beam",
                "workload": "w1",
                "backend": "postgresql",
                "operation": "point_intersection",
                "step": "0",
                "transactions": "2",
                "duration_sec": "1",
                "txns_per_sec": "2",
                "details": "",
            }
        )

    csv_response = client.get(f"/api/v1/jobs/{job_id}/result.csv")
    assert csv_response.status_code == 200
    assert "point_intersection" in csv_response.text

    json_response = client.get(f"/api/v1/jobs/{job_id}/result")
    assert json_response.status_code == 200
    payload = json_response.json()
    assert payload["canonical"] == "csv"
    assert payload["partial"] is True
    assert payload["rows"][0]["transactions"] == "2"


def test_queued_job_can_be_cancelled(client):
    response = client.post(
        "/api/v1/benchmarks",
        json={"datasets": ["beam"], "workloads": ["w1"], "duration_sec": 1},
    )
    job_id = response.json()["job_id"]
    cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["cancel_requested"] is True


def test_job_manager_executes_queued_processes_one_at_a_time(tmp_path):
    cfg = ApiConfig(
        data_root=tmp_path / "state",
        cfd_bench_executable=sys.executable,
        scheduler_poll_sec=0.01,
        cancel_grace_sec=0.05,
    )
    cfg.ensure_directories()
    store = StateStore(cfg.state_db)
    gate = ExecutionGate()
    manager = JobManager(cfg, store, gate)
    manager.start()
    try:
        first = manager.create_job(
            job_type="ingest",
            dataset="a",
            upload_id=None,
            request={},
            command=[sys.executable, "-c", "import time; print('one'); time.sleep(0.05)"],
        )
        second = manager.create_job(
            job_type="benchmark",
            dataset="b",
            upload_id=None,
            request={},
            command=[sys.executable, "-c", "print('two')"],
        )
        deadline = time.time() + 3
        while time.time() < deadline:
            a = store.get_job(first["job_id"])
            b = store.get_job(second["job_id"])
            if a["status"] == "succeeded" and b["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert store.get_job(first["job_id"])["status"] == "succeeded"
        assert store.get_job(second["job_id"])["status"] == "succeeded"
        a = store.get_job(first["job_id"])
        b = store.get_job(second["job_id"])
        assert a["started_at"] <= b["started_at"]
    finally:
        manager.stop()


def test_execution_gate_closes_interpolation_to_benchmark_race():
    gate = ExecutionGate()
    assert gate.try_begin_interactive_read()
    entered = threading.Event()

    def start_benchmark():
        gate.enter_job("benchmark")
        entered.set()

    thread = threading.Thread(target=start_benchmark)
    thread.start()
    time.sleep(0.03)
    assert not entered.is_set()
    gate.end_interactive_read()
    assert entered.wait(1.0)
    assert gate.try_begin_upload_chunk() is False
    assert gate.try_begin_interactive_read() is False
    gate.leave_job()
    thread.join(timeout=1.0)
    assert gate.try_begin_upload_chunk() is True
    gate.end_upload_chunk()


def test_interpolation_endpoint_returns_structured_result(client, monkeypatch):
    from cfd_bench_api import main as main_module

    def fake_interpolate(request):
        return InterpolationResponse(
            dataset=request.dataset,
            step=request.step,
            zone=request.zone or "0_Fluid",
            results=[
                InterpolationPointResult(
                    point=request.points[0],
                    values={"U": 1.25},
                    validation="PASS",
                )
            ],
        )

    monkeypatch.setattr(main_module, "interpolate_points", fake_interpolate)
    response = client.post(
        "/api/v1/interpolate",
        json={
            "dataset": "JBC_615k",
            "step": 200,
            "points": [[1, 2, 3]],
            "variables": ["U"],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["backend"] == "iotdb"
    assert response.json()["results"][0]["values"] == {"U": 1.25}


def test_capabilities_match_current_core_defaults(client):
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_workloads"] == ["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"]
    assert payload["workloads"][-3:] == ["w9", "w10", "w11"]
    assert payload["interpolation"]["backends"] == ["iotdb"]


def test_cli_option_injection_is_rejected_by_request_models(client):
    response = client.post(
        "/api/v1/benchmarks",
        json={
            "datasets": ["--output"],
            "workloads": ["w1"],
            "duration_sec": 1,
        },
    )
    assert response.status_code == 422

    upload = _create_completed_upload(client, "cfd-dat", "200.dat", b"x")
    response = client.post(
        "/api/v1/ingests",
        json={
            "format": "cfd-dat",
            "upload_id": upload["upload_id"],
            "dataset": "--help",
        },
    )
    assert response.status_code == 422
