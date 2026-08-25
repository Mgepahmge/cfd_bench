# CFD-Bench HTTP API

This directory is a standalone HTTP adapter around the existing `cfd-bench`
image.  It intentionally does **not** reimplement benchmark or ingest logic.

- CFD/H5 ingest jobs are translated to the existing `cfd-bench ingest` and
  `cfd-bench ingest-h5` CLI contracts.
- Benchmark jobs are translated to `cfd-bench run --output .../results.csv`.
  That CSV is the canonical benchmark result; the JSON endpoint is only a view
  of the same file.
- Fluid interpolation directly reuses the existing
  `FluidInterpolationEngine`, matching the current core limitation to IoTDB +
  Tecplot CFD datasets.
- One in-process worker serialises ingest/benchmark jobs. Benchmarks are
  exclusive with interpolation and active upload chunks so those operations do
  not perturb benchmark timing.

The production image must run with one Uvicorn worker because the scheduler and
resource gate are intentionally in-process.  Horizontal/multi-worker execution
would require an external coordinator, which is deliberately out of scope for
this offline deployment.

## Build locally

Build the existing image first (if it is not already present), then build only
the thin API image:

```bash
# Existing image; no changes are required to it.
docker image inspect cfd-bench:v22.4.2

# New image. The base tag can be overridden without changing the Dockerfile.
docker build \
  -f Dockerfile.api \
  --build-arg CFD_BENCH_BASE_IMAGE=cfd-bench:v22.4.2 \
  -t cfd-bench-api:v1 .
```

For an offline target, export only the new API image. Docker includes all
filesystem layers inherited from the base image in the archive:

```bash
docker save -o cfd-bench-api-v1.tar cfd-bench-api:v1
```

On the target machine:

```bash
docker load -i cfd-bench-api-v1.tar
docker compose -f compose.api.yaml up -d
```

The PostgreSQL and IoTDB images/volumes are the same ones used by the existing
CFD-Bench deployment and must already be available on the offline host.

Swagger/OpenAPI is available at `http://HOST:8000/docs`.

## Large-file upload flow

Create a resumable upload session:

```bash
curl -X POST http://HOST:8000/api/v1/uploads \
  -H 'Content-Type: application/json' \
  -d '{
    "format":"cfd-dat",
    "files":[{"name":"200.dat","size_bytes":48318382080}]
  }'
```

The response contains `upload_id`, `file_id`, and a recommended `chunk_size`
(default 64 MiB). Upload raw bytes with the current offset:

```bash
curl -X PATCH \
  http://HOST:8000/api/v1/uploads/UPLOAD_ID/files/FILE_ID \
  -H 'Upload-Offset: 0' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @chunk.bin
```

Recover the offset after a disconnect:

```bash
curl -I http://HOST:8000/api/v1/uploads/UPLOAD_ID/files/FILE_ID
```

After every file reaches its declared size:

```bash
curl -X POST http://HOST:8000/api/v1/uploads/UPLOAD_ID/complete
```

A CFD upload may contain multiple `.dat` files.  They are staged in one
directory, matching the existing `cfd-bench ingest --dat <directory>` behavior.
An H5 upload contains exactly one `.h5`/`.hdf5` file.

## Ingest jobs

CFD:

```json
POST /api/v1/ingests
{
  "format": "cfd-dat",
  "upload_id": "upl_...",
  "dataset": "JBC_615k",
  "backends": ["postgresql", "iotdb", "tiledb", "vtk"],
  "zone_indices": [0, 1]
}
```

H5:

```json
POST /api/v1/ingests
{
  "format": "h5",
  "upload_id": "upl_...",
  "dataset": "beam_modal",
  "backends": ["postgresql"],
  "zone": "0_Fluid",
  "timestep_mode": "sequence"
}
```

Use `POST /api/v1/uploads/{upload_id}/inspect-h5` before H5 ingest to expose the
same source structure that the core `inspect-h5 --json` command reads.

## Benchmark jobs and CSV

```json
POST /api/v1/benchmarks
{
  "datasets": ["JBC_615k"],
  "workloads": ["w1", "w2", "w6"],
  "backends": ["iotdb"],
  "duration_sec": 5,
  "geom_engine": "db"
}
```

The request returns `202` + `job_id`. Query the job with:

```text
GET /api/v1/jobs/{job_id}
GET /api/v1/jobs/{job_id}/logs?stream=stdout
POST /api/v1/jobs/{job_id}/cancel
```

The original CFD-Bench CSV is returned unchanged by:

```text
GET /api/v1/jobs/{job_id}/result.csv
```

A program-friendly view of that same CSV is available at:

```text
GET /api/v1/jobs/{job_id}/result
```

## Interpolation

```json
POST /api/v1/interpolate
{
  "dataset": "JBC_615k",
  "step": 600,
  "points": [
    [1.25, 3.48, -0.72],
    [1.30, 3.50, -0.70]
  ],
  "variables": ["U", "V", "W", "P"],
  "zone": "0_Fluid",
  "diagnostics": true
}
```

The endpoint uses one IoTDB session for the whole point batch. With diagnostics
enabled it returns the containing cell, source element ID, supporting nodes,
barycentric weights, reconstruction error, runtime vertex projection source,
and support values already exposed by the current interpolation engine.

## Scheduling semantics

- HTTP requests remain concurrent.
- Ingest and benchmark subprocesses are queued and executed by one worker.
- A benchmark waits for any active upload chunk/interpolation request to finish,
  then runs exclusively.
- New upload chunks, interpolation calls, and dataset discovery are rejected
  with HTTP `423 Locked` while a benchmark is active.
- Interpolation is also blocked during ingest because both may stress the same
  database backend.
- Job/status/log/CSV endpoints remain available while a benchmark runs.
