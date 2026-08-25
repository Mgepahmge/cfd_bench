"""FastAPI application exposing CFD-Bench over HTTP."""

from __future__ import annotations

import csv
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse

from cfd_bench.workloads.runner import DEFAULT_WORKLOADS, H5_ONLY_WORKLOADS

from .commands import (
    build_benchmark_command,
    build_cfd_ingest_command,
    build_h5_ingest_command,
)
from .config import ApiConfig
from .datasets import discover_dataset_sources
from .interpolation import interpolate_points
from .jobs import ExecutionGate, JobManager
from .schemas import (
    BenchmarkRequest,
    CapabilitiesResponse,
    CfdIngestRequest,
    CsvResult,
    DatasetEntry,
    DatasetList,
    H5IngestRequest,
    IngestRequest,
    InterpolationRequest,
    InterpolationResponse,
    JobList,
    JobView,
    UploadCreateRequest,
    UploadFileState,
    UploadSession,
)
from .state import StateStore
from .uploads import UploadService


API_VERSION = "v1"
ALL_BACKENDS = ["postgresql", "iotdb", "tiledb", "vtk"]
ALL_WORKLOADS = list(DEFAULT_WORKLOADS) + list(H5_ONLY_WORKLOADS)


def _upload_view(upload: dict, chunk_size: int) -> UploadSession:
    return UploadSession(
        upload_id=upload["upload_id"],
        format=upload["format"],
        status=upload["status"],
        chunk_size=int(chunk_size),
        created_at=upload["created_at"],
        completed_at=upload.get("completed_at"),
        files=[
            UploadFileState(
                file_id=item["file_id"],
                name=item["name"],
                size_bytes=int(item["size_bytes"]),
                offset_bytes=int(item["offset_bytes"]),
            )
            for item in upload["files"]
        ],
    )


def _job_view(job: dict) -> JobView:
    result_path = job.get("result_csv")
    partial = bool(result_path and Path(result_path).is_file() and Path(result_path).stat().st_size > 0)
    return JobView(
        job_id=job["job_id"],
        type=job["type"],
        status=job["status"],
        dataset=job.get("dataset"),
        upload_id=job.get("upload_id"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        exit_code=job.get("exit_code"),
        cancel_requested=bool(job.get("cancel_requested")),
        partial_result_available=partial,
        error=job.get("error"),
    )


def _require_completed_upload(uploads: UploadService, upload_id: str, fmt: str) -> dict:
    upload = uploads.get(upload_id)
    if upload["status"] != "completed":
        raise HTTPException(status_code=409, detail="upload must be completed before ingest")
    if upload["format"] != fmt:
        raise HTTPException(
            status_code=409,
            detail=f"upload format is {upload['format']!r}, not {fmt!r}",
        )
    return upload



def _resolve_server_ingest_source(config: ApiConfig, raw_path: str, fmt: str) -> Path:
    """Resolve a server-side ingest path inside an explicitly allowed root.

    The API accepts container-visible paths (normally /share/...).  resolve() is
    intentionally used before the allow-root check so ``..`` components and
    symlinks cannot escape the configured shared directories.
    """

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(
            status_code=422,
            detail="server_path must be an absolute path inside the API container (for example /share/case1)",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"server_path does not exist: {raw_path}") from exc

    allowed = []
    for root in config.server_ingest_roots:
        try:
            allowed.append(root.expanduser().resolve(strict=True))
        except (FileNotFoundError, OSError):
            # A configured but currently unmounted root simply cannot match.
            continue
    if not any(resolved == root or root in resolved.parents for root in allowed):
        roots_text = ", ".join(str(root) for root in config.server_ingest_roots)
        raise HTTPException(
            status_code=403,
            detail=f"server_path is outside the allowed ingest roots: {roots_text}",
        )

    if fmt == "cfd-dat":
        if resolved.is_file():
            if resolved.suffix.lower() != ".dat":
                raise HTTPException(status_code=422, detail="cfd-dat server_path file must end in .dat")
        elif resolved.is_dir():
            if not any(item.is_file() and item.suffix.lower() == ".dat" for item in resolved.iterdir()):
                raise HTTPException(
                    status_code=422,
                    detail="cfd-dat server_path directory contains no direct .dat files",
                )
        else:
            raise HTTPException(status_code=422, detail="cfd-dat server_path must be a file or directory")
    elif fmt == "h5":
        if not resolved.is_file() or resolved.suffix.lower() not in {".h5", ".hdf5"}:
            raise HTTPException(status_code=422, detail="h5 server_path must be an .h5 or .hdf5 file")
    return resolved

def _tail(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            prefix = "[...truncated to tail...]\n"
        else:
            prefix = ""
        return prefix + fh.read().decode("utf-8", errors="replace")


def create_app(config: Optional[ApiConfig] = None, *, start_worker: bool = True) -> FastAPI:
    cfg = config or ApiConfig.from_env()
    cfg.ensure_directories()
    store = StateStore(cfg.state_db)
    gate = ExecutionGate()
    manager = JobManager(cfg, store, gate)
    uploads = UploadService(cfg, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            manager.start()
        yield
        manager.stop()

    app = FastAPI(
        title="CFD-Bench API",
        version="0.1.0",
        description=(
            "HTTP adapter around the existing CFD-Bench CLI. Long-running ingest and "
            "benchmark operations are queued; benchmark CSV remains the canonical result."
        ),
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.store = store
    app.state.gate = gate
    app.state.jobs = manager
    app.state.uploads = uploads

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "active_job_type": gate.active_job_type,
        }

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    def capabilities():
        return CapabilitiesResponse(
            api_version=API_VERSION,
            core_cli=cfg.cfd_bench_executable,
            backends=ALL_BACKENDS,
            workloads=ALL_WORKLOADS,
            default_workloads=list(DEFAULT_WORKLOADS),
            geometry_engines=["db", "vtk"],
            upload_formats=["cfd-dat", "h5"],
            interpolation={
                "backends": ["iotdb"],
                "dataset_types": ["cfd"],
                "batch_points": True,
                "diagnostics": True,
            },
            scheduling={
                "heavy_job_workers": 1,
                "benchmark_exclusive": True,
                "interpolation_blocked_during_heavy_jobs": True,
                "upload_chunks_blocked_during_benchmark": True,
                "server_path_ingest": True,
                "server_ingest_roots": [str(path) for path in cfg.server_ingest_roots],
            },
        )

    @app.get("/api/v1/datasets", response_model=DatasetList)
    async def datasets():
        if gate.benchmark_running():
            raise HTTPException(
                status_code=423,
                detail="dataset discovery is temporarily disabled while a benchmark is running",
            )
        found = await run_in_threadpool(discover_dataset_sources, store)
        return DatasetList(
            datasets=[
                DatasetEntry(dataset=key, sources=sorted(sources))
                for key, sources in sorted(found.items())
            ],
            note=(
                "Best-effort inventory from successful API ingests, PostgreSQL, TileDB and VTK. "
                "The current IoTDB core discovery API requires candidate dataset names, so the "
                "HTTP adapter does not invent a separate wildcard metadata contract."
            ),
        )

    @app.post(
        "/api/v1/uploads",
        response_model=UploadSession,
        status_code=status.HTTP_201_CREATED,
    )
    def create_upload(payload: UploadCreateRequest):
        return _upload_view(uploads.create(payload), cfg.recommended_chunk_size)

    @app.get("/api/v1/uploads/{upload_id}", response_model=UploadSession)
    def get_upload(upload_id: str):
        return _upload_view(uploads.get(upload_id), cfg.recommended_chunk_size)

    @app.head("/api/v1/uploads/{upload_id}/files/{file_id}")
    def head_upload_file(upload_id: str, file_id: str):
        uploads.get(upload_id)
        item = store.get_upload_file(upload_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="upload file not found")
        return Response(
            status_code=204,
            headers={
                "Upload-Offset": str(int(item["offset_bytes"])),
                "Upload-Length": str(int(item["size_bytes"])),
            },
        )

    @app.patch("/api/v1/uploads/{upload_id}/files/{file_id}", status_code=204)
    async def patch_upload_file(
        upload_id: str,
        file_id: str,
        request: Request,
        upload_offset: int = Header(alias="Upload-Offset", ge=0),
    ):
        if not gate.try_begin_upload_chunk():
            raise HTTPException(
                status_code=423,
                detail=(
                    "a benchmark is running; upload chunks are temporarily blocked "
                    "to avoid perturbing benchmark disk I/O"
                ),
            )
        try:
            new_offset = await uploads.append_chunk(upload_id, file_id, request, upload_offset)
        finally:
            gate.end_upload_chunk()
        return Response(status_code=204, headers={"Upload-Offset": str(new_offset)})

    @app.post("/api/v1/uploads/{upload_id}/complete", response_model=UploadSession)
    def complete_upload(upload_id: str):
        return _upload_view(uploads.complete(upload_id), cfg.recommended_chunk_size)

    @app.delete("/api/v1/uploads/{upload_id}", status_code=204)
    def delete_upload(upload_id: str):
        uploads.delete(upload_id)
        return Response(status_code=204)

    @app.post("/api/v1/uploads/{upload_id}/inspect-h5")
    async def inspect_h5(upload_id: str):
        if not gate.try_begin_interactive_read():
            raise HTTPException(
                status_code=423,
                detail="H5 inspection is temporarily disabled while ingest/benchmark is running",
            )
        try:
            upload = _require_completed_upload(uploads, upload_id, "h5")
            source = uploads.file_path(upload_id, upload["files"][0]["name"])

            def do_inspect():
                from cfd_bench.ingest.h5.reader import OdbH5Reader

                return OdbH5Reader(str(source)).inspect()

            try:
                return await run_in_threadpool(do_inspect)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            gate.end_interactive_read()

    @app.post("/api/v1/ingests", response_model=JobView, status_code=202)
    def create_ingest(payload: IngestRequest = Body(discriminator="format")):
        if isinstance(payload, CfdIngestRequest):
            if payload.server_path is not None:
                source = _resolve_server_ingest_source(cfg, payload.server_path, "cfd-dat")
            else:
                upload = _require_completed_upload(uploads, payload.upload_id, "cfd-dat")
                source = uploads.files_dir(payload.upload_id)
            command = build_cfd_ingest_command(cfg, payload, source)
        elif isinstance(payload, H5IngestRequest):
            if payload.server_path is not None:
                source = _resolve_server_ingest_source(cfg, payload.server_path, "h5")
            else:
                upload = _require_completed_upload(uploads, payload.upload_id, "h5")
                source = uploads.file_path(payload.upload_id, upload["files"][0]["name"])
            command = build_h5_ingest_command(cfg, payload, source)
        else:  # pragma: no cover - discriminator already enforces this
            raise HTTPException(status_code=422, detail="unsupported ingest format")
        job = manager.create_job(
            job_type="ingest",
            dataset=payload.dataset,
            upload_id=payload.upload_id,
            request=payload.model_dump(mode="json"),
            command=command,
        )
        return _job_view(job)

    @app.post("/api/v1/benchmarks", response_model=JobView, status_code=202)
    def create_benchmark(payload: BenchmarkRequest):
        job_id = manager.new_job_id()
        job_dir = cfg.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        result_csv = job_dir / "results.csv"
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        stdout_path.touch()
        stderr_path.touch()
        command = build_benchmark_command(cfg, payload, result_csv)
        store.create_job(
            job_id=job_id,
            job_type="benchmark",
            dataset=",".join(payload.datasets),
            upload_id=None,
            request=payload.model_dump(mode="json"),
            command=command,
            result_csv=str(result_csv),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        manager.notify()
        job = store.get_job(job_id)
        assert job is not None
        return _job_view(job)

    @app.get("/api/v1/jobs", response_model=JobList)
    def list_jobs(
        job_status: Optional[Literal["queued", "running", "succeeded", "failed", "cancelled"]] = Query(default=None, alias="status"),
        job_type: Optional[Literal["ingest", "benchmark"]] = Query(default=None, alias="type"),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        return JobList(
            jobs=[
                _job_view(job)
                for job in store.list_jobs(status=job_status, job_type=job_type, limit=limit)
            ]
        )

    @app.get("/api/v1/jobs/{job_id}", response_model=JobView)
    def get_job(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_view(job)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobView, status_code=202)
    def cancel_job(job_id: str):
        job = manager.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_view(job)

    @app.get("/api/v1/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def job_logs(
        job_id: str,
        stream: Literal["stdout", "stderr"] = "stdout",
        tail_bytes: int = Query(default=1024 * 1024, ge=1, le=16 * 1024 * 1024),
    ):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        path = Path(job["stdout_path"] if stream == "stdout" else job["stderr_path"])
        return PlainTextResponse(_tail(path, tail_bytes), media_type="text/plain; charset=utf-8")

    @app.get("/api/v1/jobs/{job_id}/result.csv")
    def result_csv(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["type"] != "benchmark" or not job.get("result_csv"):
            raise HTTPException(status_code=409, detail="job has no benchmark CSV result")
        path = Path(job["result_csv"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="benchmark CSV is not available yet")
        return FileResponse(
            path,
            media_type="text/csv",
            filename=f"{job_id}-results.csv",
        )

    @app.get("/api/v1/jobs/{job_id}/result", response_model=CsvResult)
    def result_json(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["type"] != "benchmark" or not job.get("result_csv"):
            raise HTTPException(status_code=409, detail="job has no benchmark CSV result")
        path = Path(job["result_csv"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="benchmark CSV is not available yet")
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [dict(row) for row in reader]
            columns = list(reader.fieldnames or [])
        return CsvResult(
            job_id=job_id,
            partial=job["status"] != "succeeded",
            columns=columns,
            rows=rows,
        )

    @app.post("/api/v1/interpolate", response_model=InterpolationResponse)
    async def interpolate(payload: InterpolationRequest):
        if not gate.try_begin_interactive_read():
            raise HTTPException(
                status_code=423,
                detail=(
                    "ingest/benchmark job is running; interpolation is temporarily disabled "
                    "to avoid resource interference"
                ),
            )
        try:
            try:
                return await run_in_threadpool(interpolate_points, payload)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            gate.end_interactive_read()

    return app


app = create_app()
