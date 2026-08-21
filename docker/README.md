# CFD-Bench Docker stack

This stack keeps the benchmark application and stateful database services in
separate containers while presenting them as one reproducible environment.

## What is included

- `cfd-bench:v22.4.2`: CFD-Bench plus all Python backend dependencies, including
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

This creates `cfd-bench-docker-bundle-v22.4.2.tar.gz` containing:

- `cfd-bench:v22.4.2`
- `postgis/postgis:16-3.5`
- `apache/iotdb:2.0.10-standalone`

On another machine:

```bash
./docker/load_bundle.sh cfd-bench-docker-bundle-v22.4.2.tar.gz
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

## Database networking

PostgreSQL and IoTDB are intentionally **not published on host ports**.
CFD-Bench connects through the private Compose network using `postgres:5432`
and `iotdb:6667`. This avoids conflicts with PostgreSQL or IoTDB instances
already running on the host.

If you need host-side database access for debugging, add a temporary Compose
override that publishes different host ports; the application container should
still keep using `postgres:5432` and `iotdb:6667`.

## IoTDB bootstrap validation and reset

The IoTDB service is considered healthy only after `SHOW DATANODES` reports at
least one DataNode in `Running` state. A listening RPC port alone is not enough:
during a failed/partial standalone bootstrap the RPC endpoint can accept a
session while the ConfigNode still has no registered DataNode, which makes
writes fail with errors such as:

```text
DataNode is not enough, please register more. Current DataNodes: []
```

If this happens during the **first Docker setup**, reset only Docker-managed
IoTDB state and start it again:

```bash
./docker/reset_iotdb.sh
docker compose up -d iotdb
docker compose ps
```

This reset does not touch PostgreSQL and does not touch host `./data`.

Verify the registered DataNode directly:

```bash
docker compose exec iotdb \
  /iotdb/sbin/start-cli.sh -h iotdb -p 6667 -u root -pw root \
  -e "show datanodes"
```

A healthy standalone instance must contain a row whose status is `Running`.
Only after that should ingest be started.

### `ReadOnly(DiskFull)` during startup

If `SHOW DATANODES` reports `ReadOnly(DiskFull)`, resetting the IoTDB volume is
usually **not** the fix. IoTDB deliberately switches a DataNode to read-only
when the filesystem free-space ratio drops below `disk_space_warning_threshold`
(upstream default: `0.05`, i.e. 5%). Check the filesystem visible to the
container and Docker's disk usage first:

```bash
docker compose exec iotdb df -h /iotdb/data /iotdb/logs
docker system df
```

The benchmark Compose stack keeps the protection enabled but defaults the
threshold to 1%, which is often more practical on large shared benchmark
servers. Override it in `.env` when needed:

```bash
CFD_BENCH_IOTDB_DISK_SPACE_WARNING_THRESHOLD=0.01
```

Then recreate IoTDB so the restart-required setting is applied:

```bash
docker compose up -d --force-recreate iotdb
docker compose exec iotdb \
  /iotdb/sbin/start-cli.sh -h iotdb -p 6667 -u root -pw root \
  -e "show datanodes"
```

Do not reduce the threshold merely to hide a genuinely full disk. If the
filesystem has little absolute free space, reclaim Docker images/build cache or
move Docker's data root/IoTDB volumes to a larger filesystem instead.

The Compose configuration also pins both replication factors to `1`, matching
the intended one-ConfigNode/one-DataNode standalone deployment.
