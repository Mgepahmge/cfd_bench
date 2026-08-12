# CFD-Bench

CFD-Bench: A CFD-driven benchmark for scientific data processing using database approaches (VLDB 2026).

## Quick start

```bash
pip install -e ".[all]"

# One-shot ingest (PG + IoTDB + TileDB) — pass a .dat file or Postprocessing directory
cfd-bench ingest --dat /path/to/JBC_615k/Postprocessing --ship JBC_615k

# One-shot run all workloads W1–W8
cfd-bench run --ships JBC_615k --geom-engine db
```

Development mode without install:

```bash
PYTHONPATH=src python -m cfd_bench.cli ingest --dat /path/to/Postprocessing --ship JBC_615k
PYTHONPATH=src python -m cfd_bench.cli run --ships JBC_615k --geom-engine db
```

## Architecture

```
User CLI (cfd-bench ingest | run)
    → ingest/orchestrator.py  |  workloads/runner.py
        → ingest/{iotdb,tiledb,postgresql,vtk}/
        → workloads/w1..w8 + geom_resolver
            → API/*MeshClient
                → infra/* repository + mesh_runtime
```

- **Ingest**: unified modern stack; `--dat` accepts a single `.dat` or a Postprocessing directory (multi-timestep).
- **Run**: decouples scalar backend (`--backend`) from geometry engine (`--geom-engine db|vtk`).
- **Default path**: `geom_engine=db` with backends `postgresql iotdb tiledb` — no VTK required.

## Package layout

```
src/cfd_bench/
  cli/               User entry: cfd-bench ingest / run
  core/              DatasetKey, paths, shared types
  storage/schema/    Data domains (mesh_static, post_processing, derived)
  mesh_ops/          Shared geometry algorithms (W1–W8)
  infra/             Per-backend repository + mesh runtime
    postgresql/      PostGIS spatial layer, Q-criterion ops (W7)
  API/               Unified MeshClient facade (IoTDB, TileDB, PostgreSQL, VTK)
  ingest/            DAT → database ETL
    orchestrator.py  Multi-backend ingest orchestration
    iotdb/           IoTDB topology + cell_vars
    tiledb/          TileDB topology + cell_vars
    postgresql/      PG DDL, topology/cell-vars, PostGIS ETL
    vtk/             DAT → VTK baseline export (optional)
  workloads/         W1–W8 benchmark runners
    runner.py        Multi-workload orchestration
    common/          config, backends, geom_resolver, shared CLI
```

Requires Python **3.8+**. Optional extras: `iotdb`, `postgresql`, `tiledb`, `vtk`.

## Data root

Runtime paths resolve under a shared data root (VTK files, TileDB arrays, Max_Range, etc.):

```bash
export CFD_BENCH_DATA_ROOT=/path/to/your/data   # default: ~/data or /data if present
```

| Path (under data root) | Purpose |
|------------------------|---------|
| `vtk_dir/` | VTK fluid-zone files (for `--geom-engine vtk`) |
| `vtk_hull_dir/` | VTK hull-zone files |
| `TileDB_Instances/` | TileDB array root |
| `Max_Range/` | Precomputed variable min/max |

## Data loading (`cfd-bench ingest`)

Modern ingest writes three logical domains:

| Domain | Path pattern | Used by |
|--------|--------------|---------|
| `mesh_static/{zone}/` | topology (nodes, cells, adjacency, …) | W1, W3, W6, W7 |
| `post_processing/step_{t}/` | cell_vars, node_vars | W1–W5, W8 |
| `derived/step_{t}/` | cell_qcriterion, cell_gradient | W7, W8 |

### Unified ingest

```bash
# Full postprocessing directory (recommended — loads all timesteps)
cfd-bench ingest --dat /path/to/JBC_615k/Postprocessing --ship JBC_615k

# Selected backends only
cfd-bench ingest --dat /path/to/Postprocessing --ship JBC_615k --backends tiledb iotdb

# Skip PG DDL on re-run
cfd-bench ingest --dat /path/to/Postprocessing --ship JBC_615k --backends postgresql --no-init-pg-schema

# Optional VTK baseline export (--geom-engine vtk)
cfd-bench ingest --dat /path/to/Postprocessing --ship JBC_615k --include-vtk
```

**PostgreSQL**: DDL → topology → cell_vars (all `.dat` in directory) → PostGIS layers. Requires **PostgreSQL + PostGIS** and `scipy`.

**IoTDB**: `mesh_static` topology + `post_processing_management.{ship}.step_{t}.cell_vars` / `cell_vars_hull`.

**TileDB**: `mesh_static` topology + `post_processing/step_{t}/cell_vars` for every `.dat` file.

**VTK** (optional): exports `{ship}_GEO_{step}.vtk` and `{ship}_hull_{step}.vtk` under `vtk_dir/` / `vtk_hull_dir/`.

Connection defaults: `cfd_bench.infra.postgresql.config.PostgreSQLConfig` (`cae_data` @ `localhost:5432`).

### Low-level ingest scripts (maintainer / debug)

```bash
PYTHONPATH=src python -m cfd_bench.ingest.iotdb.load_cell_vars --dat_dir /path/to/Postprocessing --ship_type JBC --scale 615k
PYTHONPATH=src python -m cfd_bench.ingest.postgresql.load_topology --dat /path/to/200.dat
PYTHONPATH=src python -m cfd_bench.ingest.tiledb.load_cell_vars --dat_dir /path/to/Postprocessing
```

## Running workloads (`cfd-bench run`)

| Workload | Description |
|----------|-------------|
| W1 | Point / line / plane intersection + scalar query |
| W2 | Coordinate range query + multi-timestep point query + aggregation |
| W3 | Variable-range submesh + isosurface extraction |
| W4 | Multi-timestep particle advection (Δt = 0.01) |
| W5 | Single-timestep streamline integration (Δt = 1.0) |
| W6 | Hull surface pressure integration (normals + scalar query) |
| W7 | ROI Q-criterion computation |
| W8 | Variable range query (vortex / threshold cell selection) |

### Geometry engine (`--geom-engine`)

| Mode | Flag | Behavior |
|------|------|----------|
| `db` (default) | `--geom-engine db` | Geometry via data backend MeshClient. No VTK required. |
| `vtk` | `--geom-engine vtk` | Geometry via VTK files; baseline comparison. |

```bash
cfd-bench run --ships JBC_615k --geom-engine db
cfd-bench run --workloads w1 w2 --ships JBC_615k --backend iotdb --duration 10
cfd-bench run --ships JBC_615k --backend vtk --geom-engine vtk
```

### Single workload (developer path)

```bash
PYTHONPATH=src python -m cfd_bench.workloads.w1.run \
  --ships JBC_615k --backend postgresql iotdb --geom-engine db
```

## API usage

```python
from cfd_bench.API.iotdb_api import IoTDBMeshClient
from cfd_bench.API.postgresql_api import PostgreSQLMeshClient

client = IoTDBMeshClient()
client.connect("JBC_615k", step=200, zone="0_Fluid")
cells = client.point_intersection(points)
vals = client.point_query(cells, "P")
client.close()
```

## Known limitations

- **IoTDB Q-criterion (W7)**: `IoTDBMeshClient.compute_qcriterion_roi` is not yet implemented; use TileDB/PostgreSQL or pre-materialized derived data.
- **VTK Q-criterion (W7)**: `VTKMeshClient.compute_qcriterion_roi` is not yet implemented.

## Raw data

Download CFD Lifecycle Dataset (.dat files in `postprocessing/`) before ingest:

https://www.scidb.cn/en/detail?dataSetId=3553563d222d41998d7ccdd2ceff1bf9

## ODB-like HDF5 result ingest (PostgreSQL)

The PostgreSQL path can also ingest ODB-like `.h5` result files whose mesh is stored under `Parts` and whose field output is stored under `Steps/<step>/Frames/<frame>`.

```bash
# Inspect only; no PostgreSQL dependency is required.
cfd-bench inspect-h5 --h5 /path/to/result.h5

# Parse the file and show the canonical mapping without writing the database.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --dataset beam_static --dry-run

# Load one instance into PostgreSQL.
cfd-bench ingest-h5 \
  --h5 /path/to/result.h5 \
  --dataset beam_static \
  --instance PART-1-1 \
  --zone 0_Fluid
```

The loader assigns dense zero-based benchmark node/cell IDs, while preserving source FE labels and Step/Frame metadata in `h5_node_source`, `h5_cell_source`, and `h5_frame_metadata`. Nodal fields are averaged to cells; element/integration-point fields are reduced to one value per cell. A three-component source field (default `U`) is mapped to benchmark `U/V/W`. Exact scalar fields `P/K/E` are loaded only when present; missing physical quantities are **not fabricated**.

Explicit mappings are supported when the source uses different field names/components:

```bash
cfd-bench ingest-h5 \
  --h5 /path/to/result.h5 \
  --dataset beam_static \
  --map P=S.S11 \
  --map E=E.E11
```

HDF5 frames are mapped to integer benchmark timesteps (`sequence` by default); the original Step/Frame, mode/increment, and time/frequency are preserved in `h5_frame_metadata`. The loader also writes W3 `*_max_diffs.csv` files unless `--no-max-diffs` is used.

For HDF5 datasets, workloads can override the legacy CFD steps and variable list:

```bash
cfd-bench run \
  --ships beam_static \
  --backend postgresql \
  --steps 0 \
  --variables U V W E \
  --max-range-dir ~/data/Max_Range
```

Install HDF5 support with `pip install 'cfd_bench[h5]'`; PostgreSQL loading additionally requires `pip install 'cfd_bench[postgresql]'`.
