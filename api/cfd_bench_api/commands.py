"""Translate typed API requests into the existing CFD-Bench CLI contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .config import ApiConfig
from .schemas import BenchmarkRequest, CfdIngestRequest, H5IngestRequest


def _extend_list(command: List[str], flag: str, values) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def build_cfd_ingest_command(
    config: ApiConfig,
    request: CfdIngestRequest,
    source_dir: Path,
) -> List[str]:
    command = [
        config.cfd_bench_executable,
        "ingest",
        "--dat",
        str(source_dir),
        "--datasets",
        request.dataset,
        "--backends",
        *request.backends,
        "--zone-indices",
        *(str(x) for x in request.zone_indices),
        "--iotdb-host",
        os.getenv("CFD_BENCH_IOTDB_HOST", "127.0.0.1"),
        "--iotdb-port",
        os.getenv("CFD_BENCH_IOTDB_PORT", "6667"),
        "--iotdb-user",
        os.getenv("CFD_BENCH_IOTDB_USER", "root"),
        "--iotdb-password",
        os.getenv("CFD_BENCH_IOTDB_PASSWORD", "root"),
    ]
    command.append("--init-pg-schema" if request.init_pg_schema else "--no-init-pg-schema")
    command.append("--build-pg-spatial" if request.build_pg_spatial else "--no-build-pg-spatial")
    return command


def build_h5_ingest_command(
    config: ApiConfig,
    request: H5IngestRequest,
    source_file: Path,
) -> List[str]:
    command = [
        config.cfd_bench_executable,
        "ingest-h5",
        "--h5",
        str(source_file),
        "--datasets",
        request.dataset,
        "--backends",
        *request.backends,
        "--zone",
        request.zone,
        "--timestep-mode",
        request.timestep_mode,
    ]
    if request.instance:
        command.extend(["--instance", request.instance])
    _extend_list(command, "--steps", request.steps)
    if request.vector_field:
        command.extend(["--vector-field", request.vector_field])
    _extend_list(command, "--scalar-fields", request.scalar_fields)
    for target, source in request.field_mappings.items():
        command.extend(["--map", f"{target}={source}"])
    if request.include_empty_frames:
        command.append("--include-empty-frames")
    if not request.init_schema:
        command.append("--no-init-schema")
    if not request.build_spatial:
        command.append("--no-build-spatial")
    if not request.write_max_diffs:
        command.append("--no-max-diffs")
    return command


def build_benchmark_command(
    config: ApiConfig,
    request: BenchmarkRequest,
    result_csv: Path,
) -> List[str]:
    command = [
        config.cfd_bench_executable,
        "run",
        "--datasets",
        *request.datasets,
        "--workloads",
        *request.workloads,
        "--backend",
        *request.backends,
        "--duration",
        str(request.duration_sec),
        "--geom-engine",
        request.geom_engine,
        "--zone-hull",
        request.zone_hull,
        "--output",
        str(result_csv),
    ]
    _extend_list(command, "--steps", request.steps)
    _extend_list(command, "--variables", request.variables)
    if request.zone_fluid:
        command.extend(["--zone-fluid", request.zone_fluid])
    if request.progress:
        command.extend(
            ["--progress", "--progress-interval", str(request.progress_interval_sec)]
        )
    return command
