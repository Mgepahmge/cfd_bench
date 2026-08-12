# CFD-Bench

CFD-Bench: A CFD-driven benchmark for scientific data processing using database approaches (VLDB 2026).

## Quick start

```bash
pip install -e ".[all]"

# One-shot ingest (PG + IoTDB + TileDB) — pass a .dat file or Postprocessing directory
cfd-bench ingest --dat /path/to/JBC_615k/Postprocessing --datasets JBC_615k

# One-shot run legacy workloads W1–W8
cfd-bench run --datasets JBC_615k --geom-engine db
```

Development mode without install:

```bash
PYTHONPATH=src python -m cfd_bench.cli ingest --dat /path/to/Postprocessing --datasets JBC_615k
PYTHONPATH=src python -m cfd_bench.cli run --datasets JBC_615k --geom-engine db
```

## Architecture

```
User CLI (cfd-bench ingest | run)
    → ingest/orchestrator.py  |  workloads/runner.py
        → ingest/{iotdb,tiledb,postgresql,vtk}/
        → workloads/w1..w11 + geom_resolver
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
  mesh_ops/          Shared geometry algorithms (W1–W11)
  infra/             Per-backend repository + mesh runtime
    postgresql/      PostGIS spatial layer, Q-criterion ops (W7)
  API/               Unified MeshClient facade (IoTDB, TileDB, PostgreSQL, VTK)
  ingest/            DAT → database ETL
    orchestrator.py  Multi-backend ingest orchestration
    iotdb/           IoTDB topology + cell_vars
    tiledb/          TileDB topology + cell_vars
    postgresql/      PG DDL, topology/cell-vars, PostGIS ETL
    vtk/             DAT → VTK baseline export (optional)
  workloads/         W1–W11 benchmark runners
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
cfd-bench ingest --dat /path/to/JBC_615k/Postprocessing --datasets JBC_615k

# Selected backends only
cfd-bench ingest --dat /path/to/Postprocessing --datasets JBC_615k --backends tiledb iotdb

# Skip PG DDL on re-run
cfd-bench ingest --dat /path/to/Postprocessing --datasets JBC_615k --backends postgresql --no-init-pg-schema

# Optional VTK baseline export (--geom-engine vtk)
cfd-bench ingest --dat /path/to/Postprocessing --datasets JBC_615k --include-vtk
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
| W9 | H5 element centroid coordinate range → source element IDs (PostgreSQL) |
| W10 | H5 Frame statistics: count/min/max/mean/stddev for mapped fields (PostgreSQL) |
| W11 | H5 source point IDs → per-point min/max across all Frames for a nodal field (PostgreSQL) |

### Geometry engine (`--geom-engine`)

| Mode | Flag | Behavior |
|------|------|----------|
| `db` (default) | `--geom-engine db` | Geometry via data backend MeshClient. No VTK required. |
| `vtk` | `--geom-engine vtk` | Geometry via VTK files; baseline comparison. |

```bash
cfd-bench run --datasets JBC_615k --geom-engine db
cfd-bench run --workloads w1 w2 --datasets JBC_615k --backend iotdb --duration 10
cfd-bench run --datasets JBC_615k --backend vtk --geom-engine vtk
```

### Single workload (developer path)

```bash
PYTHONPATH=src python -m cfd_bench.workloads.w1.run \
  --datasets JBC_615k --backend postgresql iotdb --geom-engine db
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

The PostgreSQL HDF5 path is metadata-driven. For the normal single-instance case, the minimal workflow is:

```bash
# Optional inspection only.
cfd-bench inspect-h5 --h5 /path/to/result.h5

# Dataset identity is explicit; H5 layout details are inferred when unambiguous.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static

# PostgreSQL is the default backend. Dataset identity stays explicit, while zone,
# timesteps and common variables are discovered directly from PostgreSQL.
cfd-bench run --datasets beam_static
```

The dataset key is explicit and required via `--datasets`. `--instance`, `--steps`, `--vector-field`, `--scalar-fields`, `--map`, `--zone-fluid`, `--variables`, etc. remain available as **overrides** when the source is genuinely ambiguous; they are not required for ordinary unambiguous files. Explicit `--map` entries augment the inferred mapping instead of replacing it, so for example `--map P=S.S11` keeps automatically discovered U/V/W/E.

The loader assigns dense zero-based benchmark node/cell IDs while preserving source FE labels and Step/Frame metadata in `h5_node_source`, `h5_cell_source`, and `h5_frame_metadata`. Nodal fields are averaged to cells for the original benchmark operations **and their genuine source nodal values are also preserved in `node_scalar`** for W11; element/integration-point fields are reduced to one value per cell. A recognizable three-component displacement/velocity field is mapped to benchmark U/V/W. Exact P/K/E field names are loaded when present; unknown physical quantities are not silently invented.

### H5-only workloads W9–W11

W9–W11 currently target PostgreSQL datasets created by `ingest-h5`. W9 selects source H5 element labels by element-centroid coordinate box. W10 computes count/min/max/mean/population-stddev for every mapped physical quantity in a selected Frame, using genuine nodal values when the source field is nodal and cell values otherwise. W11 samples source H5 node labels and computes per-node min/max for one directly nodal physical quantity across all ingested Frames. Because v3 did not persist genuine nodal values, datasets ingested with v3 should be re-ingested before running W11.

```bash
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_modal
cfd-bench run --workloads w9 w10 w11 --datasets beam_modal --duration 10
```

### W3 max-diff metadata

New HDF5 ingests materialize W3 search widths in PostgreSQL table `benchmark_max_diff`. PostgreSQL W3 reads this table directly and, for databases ingested by an older version, can compute the values from `cell_scalar` + `cell_adjacency` on demand. Therefore PostgreSQL W3 no longer depends on `~/data/Max_Range` or any other machine-specific sidecar directory. CSV max-diff files are still exported by default only for backward compatibility with the non-PostgreSQL backends.

### PostgreSQL connection settings

The same environment variables are used by both ingest and run:

```bash
export CFD_BENCH_PG_DB_NAME=cae_data
export CFD_BENCH_PG_USER=postgres
export CFD_BENCH_PG_PASSWORD=...
export CFD_BENCH_PG_HOST=localhost
export CFD_BENCH_PG_PORT=5432
```

The legacy `--db-*` options on `ingest-h5` remain available as one-command overrides.

Install HDF5 support with `pip install 'cfd_bench[h5]'`; PostgreSQL loading additionally requires `pip install 'cfd_bench[postgresql]'`.
