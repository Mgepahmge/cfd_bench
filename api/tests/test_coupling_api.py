from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import h5py
from fastapi.testclient import TestClient

from cfd_bench.cli.main import build_parser
from cfd_bench_api.commands import build_coupling_command
from cfd_bench_api.config import ApiConfig
from cfd_bench_api.main import create_app
from cfd_bench_api.schemas import CouplingRequest
from cfd_bench_api.state import StateStore


def _config(tmp_path: Path) -> ApiConfig:
    return ApiConfig(
        data_root=tmp_path / "api-data",
        cfd_bench_executable="cfd-bench",
        scheduler_poll_sec=0.02,
        cancel_grace_sec=0.05,
        server_ingest_roots=(tmp_path / "share",),
    )


def test_coupling_command_builder_matches_core_cli(tmp_path):
    cfg = _config(tmp_path)
    request = CouplingRequest(
        structure_dataset="beam",
        cfd_dataset="flow_default",
        cfd_step=1000,
        variables=["U", "V", "W", "P"],
        structure_zone="0_Structure",
        cfd_zone="0_Fluid",
        batch_size=8192,
        diagnostics=True,
        progress=False,
    )
    command = build_coupling_command(cfg, request, tmp_path / "coupling.h5")
    args = build_parser().parse_args(command[1:])
    assert args.command == "couple"
    assert args.structure_dataset == "beam"
    assert args.cfd_dataset == "flow_default"
    assert args.cfd_step == 1000
    assert args.variables == ["U", "V", "W", "P"]
    assert args.structure_zone == "0_Structure"
    assert args.cfd_zone == "0_Fluid"
    assert args.batch_size == 8192
    assert args.diagnostics is True
    assert args.no_progress is True


def test_coupling_job_and_h5_result_endpoints(tmp_path):
    cfg = _config(tmp_path)
    app = create_app(cfg, start_worker=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/couplings",
            json={
                "structure_dataset": "beam",
                "cfd_dataset": "flow_default",
                "cfd_step": 1000,
                "variables": ["P", "U"],
                "batch_size": 1024,
            },
        )
        assert response.status_code == 202, response.text
        job_view = response.json()
        assert job_view["type"] == "coupling"
        assert job_view["status"] == "queued"
        job_id = job_view["job_id"]

        job = app.state.store.get_job(job_id)
        assert job is not None
        result_path = Path(job["result_h5"])
        assert result_path.name == "coupling.h5"
        assert "couple" in job["command"]

        result_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(result_path, "w") as h5:
            h5.attrs["format"] = "cfd-bench-structure-cfd-coupling-v1"
            meta = h5.create_group("metadata")
            meta.attrs["structure_dataset"] = "beam"
            meta.attrs["structure_zone"] = "0_Structure"
            meta.attrs["cfd_dataset"] = "flow_default"
            meta.attrs["cfd_zone"] = "0_Fluid"
            meta.attrs["cfd_step"] = 1000
            meta.attrs["variables_json"] = json.dumps(["P", "U"])
            meta.attrs["node_count"] = 10
            meta.attrs["success_count"] = 8
            meta.attrs["outside_count"] = 1
            meta.attrs["no_containing_cell_count"] = 1
            meta.attrs["failed_count"] = 0
        app.state.store.finish_job(job_id, status="succeeded", exit_code=0)

        summary = client.get(f"/api/v1/jobs/{job_id}/result")
        assert summary.status_code == 200, summary.text
        payload = summary.json()
        assert payload["canonical"] == "h5"
        assert payload["node_count"] == 10
        assert payload["success_count"] == 8
        assert payload["variables"] == ["P", "U"]

        download = client.get(f"/api/v1/jobs/{job_id}/result.h5")
        assert download.status_code == 200
        assert download.content[:8] == b"\x89HDF\r\n\x1a\n"


def test_state_store_migrates_existing_jobs_table_with_result_h5(tmp_path):
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                dataset TEXT,
                upload_id TEXT,
                request_json TEXT NOT NULL,
                command_json TEXT NOT NULL,
                result_csv TEXT,
                stdout_path TEXT NOT NULL,
                stderr_path TEXT NOT NULL,
                pid INTEGER,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                exit_code INTEGER,
                error TEXT
            );
            CREATE TABLE uploads (
                upload_id TEXT PRIMARY KEY,
                format TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE upload_files (
                file_id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                name TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                offset_bytes INTEGER NOT NULL DEFAULT 0
            );
            """
        )
    StateStore(db)
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert "result_h5" in columns
