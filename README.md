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
- **Default run path**: `geom_engine=db` with PostgreSQL; IoTDB/TileDB remain explicit via `--backend`.

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
| W9 | H5 element centroid coordinate range → source element IDs (PostgreSQL / IoTDB / TileDB) |
| W10 | H5 Frame statistics: count/min/max/mean/stddev for mapped fields (PostgreSQL / IoTDB / TileDB) |
| W11 | H5 source point IDs → per-point min/max across all Frames for a nodal field (PostgreSQL / IoTDB / TileDB) |

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

- **W6 is data-dependent**: it requires a pressure-like `P` field and a surface/hull zone suitable for force integration. Structural H5 files that do not contain those semantics are not made artificial just to run W6.
- **VTK Q-criterion (W7)**: `VTKMeshClient.compute_qcriterion_roi` is not yet implemented.

## Raw data

Download CFD Lifecycle Dataset (.dat files in `postprocessing/`) before ingest:

https://www.scidb.cn/en/detail?dataSetId=3553563d222d41998d7ccdd2ceff1bf9

## ODB-like HDF5 result ingest (PostgreSQL / IoTDB / TileDB)

The PostgreSQL HDF5 path is metadata-driven. For the normal single-instance case, the minimal workflow is:

```bash
# Optional inspection only.
cfd-bench inspect-h5 --h5 /path/to/result.h5

# Dataset identity is explicit; H5 layout details are inferred when unambiguous.
# PostgreSQL remains the default target, preserving the v5 command.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static

# IoTDB only.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static --backends iotdb

# TileDB only.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static --backends tiledb

# Or load the same canonical H5 data to multiple backends.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static --backends postgresql iotdb tiledb

# Runtime metadata is auto-discovered from the selected backend.
cfd-bench run --datasets beam_static
cfd-bench run --datasets beam_static --backend iotdb
cfd-bench run --datasets beam_static --backend tiledb
```

The dataset key is explicit and required via `--datasets`. `--instance`, `--steps`, `--vector-field`, `--scalar-fields`, `--map`, `--zone-fluid`, `--variables`, etc. remain available as **overrides** when the source is genuinely ambiguous; they are not required for ordinary unambiguous files. Explicit `--map` entries augment the inferred mapping instead of replacing it, so for example `--map P=S.S11` keeps automatically discovered U/V/W/E.

The loader assigns dense zero-based benchmark node/cell IDs while preserving source FE labels and Step/Frame metadata in `h5_node_source`, `h5_cell_source`, and `h5_frame_metadata`. Nodal fields are averaged to cells for the original benchmark operations **and their genuine source nodal values are also preserved in `node_scalar`** for W11; element/integration-point fields are reduced to one value per cell. A recognizable three-component displacement/velocity field is mapped to benchmark U/V/W. Exact P/K/E field names are loaded when present; unknown physical quantities are not silently invented.

C3D10 quadratic tetrahedra are handled explicitly: adjacency is built from complete six-node triangular faces instead of the generic shared-node heuristic, so sharing only a quadratic edge is not treated as a neighbor. PostgreSQL spatial shells use the four C3D10 corner nodes directly and are streamed to PostGIS in batches, avoiding one ConvexHull call and a large in-memory WKT buffer per element. Existing B33 and legacy CFD ingest paths keep their previous behavior.

### H5 → IoTDB layout

The H5 IoTDB adapter reuses the existing tree-model convention that the IoTDB `Time` value is the dense benchmark entity ID, not physical simulation time. Frame identity remains in the path (`step_<n>`) and in H5 metadata.

| IoTDB device | `Time` meaning | Stored values | Workloads |
|---|---|---|---|
| `mesh_static.<dataset>.<zone>.nodes` | dense node ID | x/y/z | W1/W3/W7 |
| `...cells` | dense element ID | centroid + bbox + numeric cell type | W1/W2/W4/W5/W7/W9 |
| `...cell_nodes` | dense element ID | dynamic `node_id_*` columns | W1/W3/W6 |
| `...cell_adjacency` | dense element ID | dynamic `neighbor_id_*` columns | W3/W7 |
| `...node_source` | dense node ID | original H5 node label | W11 |
| `...cell_source` | dense element ID | original H5 element label + element type | W9 |
| `post_processing_management.<dataset>.step_<n>.cell_vars` | dense element ID | mapped U/V/W/P/K/E/... | W1–W8/W10 |
| `...step_<n>.node_vars` | dense node ID | genuine nodal mapped fields | W10/W11 |
| `h5_metadata.<dataset>.dataset_meta` | 0 | zone, Part/Instance, variables, element types, counts | discovery/W9–W11 |
| `h5_metadata.<dataset>.frames` | frame/timestep ID | Step/Frame/mode/time-frequency metadata | discovery/W11 |
| `derived.<dataset>.step_<n>.max_diff` | 0 | per-variable max neighbor difference | W3 |

### H5 → TileDB layout

The H5 TileDB adapter reuses the legacy dense-array layout. Dense benchmark IDs remain the array dimensions, while source FE labels and H5 metadata are stored separately so W1–W8 keep using the original TileDB runtime.

| TileDB array | Dimension | Stored values | Workloads |
|---|---|---|---|
| `mesh_static/<zone>/nodes.tdb` | dense node ID | x/y/z | W1/W3/W7 |
| `mesh_static/<zone>/cells.tdb` | dense element ID | centroid + bbox + numeric cell type | W1/W2/W4/W5/W7/W9 |
| `mesh_static/<zone>/cell_nodes.tdb` | dense element ID | 16 fixed node slots | W1/W3/W6 |
| `mesh_static/<zone>/cell_adjacency.tdb` | dense element ID | 16 fixed neighbor slots | W3/W7 |
| `mesh_static/<zone>/node_source.tdb` | dense node ID | original H5 node label | W11 |
| `mesh_static/<zone>/cell_source.tdb` | dense element ID | original H5 element label + element type code | W9 |
| `post_processing/step_<n>/cell_vars.tdb` | dense element ID | mapped cell fields | W1–W8/W10 |
| `post_processing/step_<n>/node_vars.tdb` | dense node ID | genuine nodal mapped fields | W10/W11 |
| `h5_metadata/dataset_meta.tdb` | 0 | discovery metadata and Frame list | discovery/W9–W11 |
| `derived/step_<n>/max_diff.tdb` | 0 | per-variable max neighbor difference | W3 |

B33 and C3D10 both fit the current 16-slot topology arrays (2 and 10 nodes per element respectively; C3D10 has at most 4 face-neighbors).

### H5-only workloads W9–W11

W9–W11 support PostgreSQL, IoTDB, and TileDB H5 ingests. W9 selects source H5 element labels by element-centroid coordinate box. W10 computes count/min/max/mean/population-stddev for mapped physical quantities in a selected Frame, using genuine nodal values when the source field is nodal and cell values otherwise. W11 samples source H5 node labels and computes per-node min/max for one directly nodal physical quantity across all ingested Frames.

```bash
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_modal --backends tiledb
cfd-bench run --workloads w9 w10 w11 --datasets beam_modal --backend tiledb --duration 10
```

### W3 max-diff metadata

New HDF5 ingests materialize W3 search widths inside the selected backend: PostgreSQL uses `benchmark_max_diff`; IoTDB uses `derived.{dataset}.step_<n>.max_diff`; TileDB uses `derived/step_<n>/max_diff.tdb`. PostgreSQL retains its old-database recomputation fallback, while legacy CFD IoTDB/TileDB can still use Max_Range sidecars. New H5 IoTDB/TileDB workloads do not depend on a machine-specific Max_Range directory.

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

IoTDB ingest/run uses the same environment variables by default:

```bash
export CFD_BENCH_IOTDB_HOST=127.0.0.1
export CFD_BENCH_IOTDB_PORT=6667
export CFD_BENCH_IOTDB_USER=root
export CFD_BENCH_IOTDB_PASSWORD=root
export CFD_BENCH_IOTDB_ROOT_PATH=root.simulation_data
```

`ingest-h5` also exposes `--iotdb-*` one-command overrides. PostgreSQL defaults and commands are unchanged when `--backends` is omitted.

Install HDF5 support with `pip install 'cfd_bench[h5]'`; backend loading additionally requires the corresponding extra: `cfd_bench[postgresql]`, `cfd_bench[iotdb]`, or `cfd_bench[tiledb]`.
