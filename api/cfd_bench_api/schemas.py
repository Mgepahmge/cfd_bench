"""Pydantic request/response models for API v1."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Backend = Literal["postgresql", "iotdb", "tiledb", "vtk"]
UploadFormat = Literal["cfd-dat", "h5"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

def _reject_cli_option_like(value: str, field_name: str) -> str:
    text = str(value)
    if "\x00" in text or text.startswith("-"):
        raise ValueError(f"{field_name} must not start with '-' or contain NUL")
    return text


def _reject_cli_option_like_list(values, field_name: str):
    if values is None:
        return None
    return [_reject_cli_option_like(value, field_name) for value in values]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UploadFileSpec(StrictModel):
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def validate_basename(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("file name must be a plain basename without path separators")
        return value


class UploadCreateRequest(StrictModel):
    format: UploadFormat
    files: List[UploadFileSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_files_for_format(self):
        names = [item.name for item in self.files]
        if len(set(names)) != len(names):
            raise ValueError("duplicate file names are not allowed in one upload session")
        if self.format == "cfd-dat":
            bad = [name for name in names if not name.lower().endswith(".dat")]
            if bad:
                raise ValueError(f"cfd-dat uploads accept only .dat files: {bad}")
        else:
            if len(names) != 1:
                raise ValueError("h5 uploads must contain exactly one file")
            if not names[0].lower().endswith((".h5", ".hdf5")):
                raise ValueError("h5 uploads require a .h5 or .hdf5 file")
        return self


class UploadFileState(StrictModel):
    file_id: str
    name: str
    size_bytes: int
    offset_bytes: int


class UploadSession(StrictModel):
    upload_id: str
    format: UploadFormat
    status: Literal["uploading", "completed"]
    chunk_size: int
    created_at: str
    completed_at: Optional[str] = None
    files: List[UploadFileState]


class CfdIngestRequest(StrictModel):
    format: Literal["cfd-dat"] = "cfd-dat"
    upload_id: str
    dataset: str = Field(min_length=1, max_length=200)
    backends: List[Backend] = Field(
        default_factory=lambda: ["postgresql", "iotdb", "tiledb"]
    )
    zone_indices: List[int] = Field(default_factory=lambda: [0, 1], min_length=1)
    init_pg_schema: bool = True
    build_pg_spatial: bool = True

    @field_validator("dataset")
    @classmethod
    def validate_dataset_token(cls, value: str) -> str:
        return _reject_cli_option_like(value, "dataset")


class H5IngestRequest(StrictModel):
    format: Literal["h5"] = "h5"
    upload_id: str
    dataset: str = Field(min_length=1, max_length=200)
    backends: List[Backend] = Field(default_factory=lambda: ["postgresql"])
    instance: Optional[str] = None
    zone: str = "0_Fluid"
    steps: Optional[List[str]] = None
    vector_field: Optional[str] = None
    scalar_fields: Optional[List[str]] = None
    field_mappings: Dict[str, str] = Field(default_factory=dict)
    timestep_mode: Literal["sequence", "frame-index", "inc-mode"] = "sequence"
    include_empty_frames: bool = False
    init_schema: bool = True
    build_spatial: bool = True
    write_max_diffs: bool = True

    @field_validator("dataset", "instance", "zone", "vector_field")
    @classmethod
    def validate_optional_tokens(cls, value, info):
        if value is None:
            return None
        return _reject_cli_option_like(value, info.field_name)

    @field_validator("steps", "scalar_fields")
    @classmethod
    def validate_list_tokens(cls, value, info):
        return _reject_cli_option_like_list(value, info.field_name)

    @field_validator("field_mappings")
    @classmethod
    def validate_mapping_tokens(cls, value: Dict[str, str]) -> Dict[str, str]:
        return {
            _reject_cli_option_like(target, "field mapping target"): _reject_cli_option_like(source, "field mapping source")
            for target, source in value.items()
        }


IngestRequest = Union[CfdIngestRequest, H5IngestRequest]


class BenchmarkRequest(StrictModel):
    datasets: List[str] = Field(min_length=1)
    workloads: List[
        Literal["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11"]
    ] = Field(default_factory=lambda: ["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"])
    backends: List[Backend] = Field(default_factory=lambda: ["postgresql"])
    duration_sec: float = Field(default=5.0, gt=0)
    geom_engine: Literal["db", "vtk"] = "db"
    steps: Optional[List[int]] = None
    variables: Optional[List[str]] = None
    zone_fluid: Optional[str] = None
    zone_hull: str = "0_Wall_hull"
    progress: bool = False
    progress_interval_sec: float = Field(default=5.0, gt=0)

    @field_validator("datasets", "variables")
    @classmethod
    def validate_list_tokens(cls, value, info):
        return _reject_cli_option_like_list(value, info.field_name)

    @field_validator("zone_fluid", "zone_hull")
    @classmethod
    def validate_zone_tokens(cls, value, info):
        if value is None:
            return None
        return _reject_cli_option_like(value, info.field_name)


class JobView(StrictModel):
    job_id: str
    type: Literal["ingest", "benchmark"]
    status: JobStatus
    dataset: Optional[str] = None
    upload_id: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    cancel_requested: bool = False
    partial_result_available: bool = False
    error: Optional[str] = None


class JobList(StrictModel):
    jobs: List[JobView]


class CsvResult(StrictModel):
    job_id: str
    canonical: Literal["csv"] = "csv"
    partial: bool
    columns: List[str]
    rows: List[Dict[str, str]]


class InterpolationRequest(StrictModel):
    dataset: str = Field(min_length=1)
    step: int
    points: List[List[float]] = Field(min_length=1)
    variables: Optional[List[str]] = None
    zone: Optional[str] = None
    diagnostics: bool = False

    @field_validator("dataset", "zone")
    @classmethod
    def validate_name_tokens(cls, value, info):
        if value is None:
            return None
        return _reject_cli_option_like(value, info.field_name)

    @field_validator("variables")
    @classmethod
    def validate_variable_tokens(cls, value):
        return _reject_cli_option_like_list(value, "variables")

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: List[List[float]]) -> List[List[float]]:
        if any(len(point) != 3 for point in value):
            raise ValueError("each interpolation point must contain exactly three coordinates")
        return value


class InterpolationPointResult(StrictModel):
    point: List[float]
    values: Dict[str, float]
    validation: Literal["PASS", "FAIL"]
    cell_id: Optional[int] = None
    source_element_id: Optional[int] = None
    cell_node_ids: Optional[List[int]] = None
    support_node_ids: Optional[List[int]] = None
    support_source_node_ids: Optional[List[int]] = None
    weights: Optional[List[float]] = None
    reconstruction_error: Optional[float] = None
    vertex_value_source: Optional[str] = None
    support_vertex_values: Optional[Dict[str, List[float]]] = None


class InterpolationResponse(StrictModel):
    dataset: str
    backend: Literal["iotdb"] = "iotdb"
    step: int
    zone: str
    results: List[InterpolationPointResult]


class CapabilitiesResponse(StrictModel):
    api_version: str
    core_cli: str
    backends: List[Backend]
    workloads: List[str]
    default_workloads: List[str]
    geometry_engines: List[str]
    upload_formats: List[str]
    interpolation: Dict[str, object]
    scheduling: Dict[str, object]


class DatasetEntry(StrictModel):
    dataset: str
    sources: List[str]


class DatasetList(StrictModel):
    datasets: List[DatasetEntry]
    note: str
