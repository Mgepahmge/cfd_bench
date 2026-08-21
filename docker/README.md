# CFD-Bench Docker stack

This stack keeps the benchmark application and stateful database services in
separate containers while presenting them as one reproducible environment.

## What is included

- `cfd-bench:v22.4`: CFD-Bench plus all Python backend dependencies, including
  Apache IoTDB Python client 2.0.10, TileDB-Py, VTK, h5py, psycopg2, SciPy and
  tqdm.
- `postgis/postgis:16-3.5`: PostgreSQL 16 + PostGIS 3.5.
- `apache/iotdb:2.0.10-standalone`: Apache IoTDB standalone server.
- TileDB is embedded/file-backed in this project, so it runs inside the
  `cfd-bench` container rather than as a separate server.

The application and database configuration are matched by `compose.yaml`:

| Setting | Application | Service |
| --- | --- | --- |
| PostgreSQL database | `cae_data` | `POSTGRES_DB=cae_data` |
| PostgreSQL user | `postgres` | `POSTGRES_USER=postgres` |
| PostgreSQL password | `123456` | `POSTGRES_PASSWORD=123456` |
| PostgreSQL host/port | `postgres:5432` | service `postgres`, port 5432 |
| IoTDB host/port | `iotdb:6667` | service `iotdb`, RPC 6667 |
| IoTDB credentials | `root/root` | IoTDB default `root/root` |
| IoTDB root path | `root.simulation_data` | CFD-Bench canonical tree |
| Shared file data | `/app/data` | host `./data` bind mount |

The entrypoint additionally makes a real authenticated connection to both
PostgreSQL and IoTDB before executing the requested command.

## 1. Build

From the project root:

```bash
mkdir -p data datasets output
docker compose build
```

The stack is pinned to `linux/amd64` for reproducibility because the selected PostGIS image is amd64. Performance benchmarking should therefore be run on a native x86-64 Docker host rather than through CPU emulation.

The build requires Internet access to pull the base images and Python wheels.

## 2. Start the databases

```bash
docker compose up -d postgres iotdb
```

Check status:

```bash
docker compose ps
```

## 3. Run CFD-Bench directly

```bash
docker compose run --rm cfd-bench cfd-bench --help
```

Example benchmark:

```bash
docker compose run --rm cfd-bench \
  cfd-bench run \
  --backend postgresql iotdb tiledb vtk \
  --workloads w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 \
  --duration 10 \
  --datasets Kvlcc_351K_Small \
  --output /app/output/results.csv
```

Host-visible CSV files should be written under `/app/output`; that directory is
mapped to `./output` on the host.

## 4. Input datasets

Put source datasets under the host `./datasets` directory. It is mounted
read-only as `/datasets` inside the application container.

For example:

```text
datasets/
  Kvlcc2_351k/
    Postprocessing/
  structure_01.h5
```

## 5. Run the supplied test scripts

CFD ingest:

```bash
docker compose run --rm cfd-bench \
  ./scripts/01_ingest_cfd.sh \
  /datasets/Kvlcc2_351k/Postprocessing \
  Kvlcc_351K_Small
```

H5 ingest:

```bash
docker compose run --rm cfd-bench \
  ./scripts/02_ingest_h5.sh \
  /datasets/structure_01.h5 \
  structure_01
```

Full W1-W11 test:

```bash
docker compose run --rm cfd-bench \
  ./scripts/03_full_test.sh Kvlcc_351K_Small 10 /app/output/full.csv
```

Performance ratio test:

```bash
docker compose run --rm cfd-bench \
  ./scripts/04_performance_test.sh Kvlcc_351K_Small
```

The scripts derive their project root as `/app`, so they set
`CFD_BENCH_DATA_ROOT=/app/data`, exactly matching the Compose bind mount.

## 6. Fluid interpolation

```bash
docker compose run --rm cfd-bench \
  cfd-bench interpolate \
  --datasets Kvlcc_351K_Small \
  --step 200 \
  --point -7.2 0.15 0.04 \
  --variables U V W P
```

## Optional: export all images as one offline bundle

On a machine with Docker and Internet access:

```bash
./docker/build_bundle.sh
```

This creates `cfd-bench-docker-bundle-v22.4.tar.gz` containing:

- `cfd-bench:v22.4`
- `postgis/postgis:16-3.5`
- `apache/iotdb:2.0.10-standalone`

On another machine:

```bash
./docker/load_bundle.sh cfd-bench-docker-bundle-v22.4.tar.gz
docker compose up -d postgres iotdb
```

The Compose file will then use the already-loaded images. Runtime database and
benchmark data are still kept separately in volumes / `./data`; they are not
baked into the image bundle.

## Persistence and reset

Persistent state:

- PostgreSQL: named volume `postgres-data`
- IoTDB: named volumes `iotdb-data` and `iotdb-logs`
- TileDB/VTK/runtime files: host `./data`
- CSV output: host `./output`

Stop without deleting data:

```bash
docker compose down
```

Delete PostgreSQL/IoTDB volumes as well:

```bash
docker compose down -v
rm -rf data/* output/*
```

## Host port conflicts

Container-to-container settings never change, but host port exposure can be
changed through `.env`:

```bash
cp .env.example .env
```

Then edit, for example:

```text
CFD_BENCH_PG_HOST_PORT=15432
CFD_BENCH_IOTDB_HOST_PORT=16667
```

CFD-Bench will still connect internally to `postgres:5432` and `iotdb:6667`.
