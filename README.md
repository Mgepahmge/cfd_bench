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
    vtk/             CFD/H5 → first-class VTK backend
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
| `vtk_dir/` | VTK backend datasets (`<dataset>/manifest.json` + per-zone/frame `.vtu`) |
| `vtk_hull_dir/` | Read-only compatibility path for pre-v16 flat VTK hull files |
| `TileDB_Instances/` | TileDB array root |
| `Max_Range/` | Compatibility sidecars for pre-v14 legacy CFD databases |

## Legacy CFD DAT path (v14 canonical ingest)

The legacy Tecplot CFD path was rebuilt in v14 around a backend-neutral canonical mesh contract. The decoder accepts the NaViiX/Tecplot `FEPolyhedron` layout used by the benchmark (`N/E`, verbose `Nodes/Faces/Elements`, `DATAPACKING=BLOCK`, cell-centred U/V/W/P/K/E, face-node lists and left/right element ownership), including multiple zones such as a fluid volume plus a wall/hull surface. Cell centroids, adjacency and boundary ownership are derived once from the same topology and then written consistently to PostgreSQL, IoTDB and TileDB.

**CFD datasets ingested by v13 or earlier should be re-ingested with v14.** The canonical CFD path adds/rebuilds static runtime metadata, PostgreSQL cell bounds/point buckets, backend-local W3 max-diff metadata, and dynamic-width IoTDB/TileDB connectivity. Re-ingest is deterministic and replaces stale static rows instead of preserving old `ON CONFLICT DO NOTHING` topology.

**v14.1 PostgreSQL hotfix:** the CFD-only `cell_bounds` table uses `min_x/max_x/min_y/max_y/min_z/max_z` rather than PostgreSQL's reserved MVCC system-column names `xmin/xmax`. CFD topology parsing now shows a passive progress bar by default during `cfd-bench ingest`, and the canonical exporter reuses one node-coordinate matrix while computing all cell AABBs instead of rebuilding it once per cell. H5/structural ingest is unchanged.

**H5/structural datasets do not need to be re-ingested for the three database backends.** Their PostgreSQL/IoTDB/TileDB storage contracts remain frozen. v15 extends W9-W11 to canonical CFD data at runtime; v16 adds a separate VTK storage backend without changing those frozen database ingests.

```bash
# Rebuild one CFD dataset with the canonical DAT path.
cfd-bench ingest --dat /path/to/Kvlcc_351k_Small/Postprocessing --datasets Kvlcc_351k_Small

# Then run the legacy CFD workloads on any selected backend.
cfd-bench run --datasets Kvlcc_351k_Small --backend postgresql --workloads w1 w2 w3 w4 w5 w6 w7 w8
cfd-bench run --datasets Kvlcc_351k_Small --backend iotdb      --workloads w1 w2 w3 w4 w5 w6 w7 w8
cfd-bench run --datasets Kvlcc_351k_Small --backend tiledb     --workloads w1 w2 w3 w4 w5 w6 w7 w8
```

For canonical CFD data, empty random point/ROI samples are bounded per transaction so W1/W2/W4/W7 always return control to the historical outer duration loop. This does **not** add a hard benchmark deadline or change the validated H5 retry behaviour. PostgreSQL point buckets cover cell AABBs rather than only cell centroids; IoTDB/TileDB store connectivity with the width required by the actual FE topology instead of truncating at 16 entries. W3 max-diff metadata is stored inside each backend and no longer depends on another backend generating a `Max_Range` sidecar.

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

# VTK is a normal ingest backend (v16+)
cfd-bench ingest --dat /path/to/Postprocessing --datasets JBC_615k --backends vtk

# VTK can be selected together with database backends
cfd-bench ingest --dat /path/to/Postprocessing --datasets JBC_615k --backends postgresql iotdb tiledb vtk
```

**PostgreSQL**: DDL → topology → cell_vars (all `.dat` in directory) → PostGIS layers. Requires **PostgreSQL + PostGIS** and `scipy`.

**IoTDB**: `mesh_static` topology + `post_processing_management.{ship}.step_{t}.cell_vars` / `cell_vars_hull`.

**TileDB**: `mesh_static` topology + `post_processing/step_{t}/cell_vars` for every `.dat` file.

**VTK**: first-class backend for both canonical CFD and H5 data. New ingests write `<vtk_dir>/<dataset>/manifest.json` plus `zones/<zone>/step_<frame>.vtu`. The manifest carries dataset type, zones, steps, variables, H5 frame metadata, source-ID semantics and W3 max-diff metadata. Pre-v16 flat `.vtk` files remain read-only compatible.

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
| W5 | Single-timestep streamline integration (Δt = 1.0; exits on mesh miss/invalid velocity/deadline) |
| W6 | Hull surface pressure integration (normals + scalar query) |
| W7 | ROI Q-criterion computation |
| W8 | Variable range query (vortex / threshold cell selection) |
| W9 | Element centroid coordinate range → H5 source IDs / CFD implicit element IDs |
| W10 | Per-frame count/min/max/mean/stddev for available physical quantities |
| W11 | Point IDs → min/max across frames (genuine H5 nodal fields; runtime cell→node projection for CFD) |

### W5 point-location semantics

`point_intersection()` now has one no-hit convention across PostgreSQL, IoTDB, TileDB, and VTK: points that do not hit a cell are omitted from the returned ID array. W5 also enforces the benchmark deadline inside each streamline, stops on non-finite/zero velocity, and caps the number of integration steps so a single transaction cannot run forever. Existing ingested data does not need to be rebuilt for this change.

### Geometry engine (`--geom-engine`)

| Mode | Flag | Behavior |
|------|------|----------|
| `db` (default) | `--geom-engine db` | Geometry via data backend MeshClient. No VTK required. |
| `vtk` | `--geom-engine vtk` | Geometry via VTK files; baseline comparison. |

```bash
cfd-bench run --datasets JBC_615k --geom-engine db
cfd-bench run --workloads w1 w2 --datasets JBC_615k --backend iotdb --duration 10
cfd-bench run --datasets JBC_615k --backend vtk
```

### CSV benchmark output

Console output remains unchanged. Pass `--output` to mirror each completed benchmark section to a CSV file **after** its timed loop finishes:

```bash
cfd-bench run \
  --backend postgresql iotdb tiledb vtk \
  --workloads w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 \
  --datasets Kvlcc_351K_Small \
  --duration 10 \
  --output results.csv
```

The CSV columns are `run_id,timestamp_utc,dataset,workload,backend,operation,step,transactions,duration_sec,txns_per_sec,details`. W1 emits separate point/line/plane rows; W6 records zone/scalar details when available; W11 records batch and variable information. Without `--output`, no result file is opened and console behavior remains unchanged. The requested CSV path is created/overwritten once per `cfd-bench run` invocation. CSV rows are written only after a benchmark section has left its timing loop, so result persistence does not consume transaction-window time.

When IoTDB and TileDB are selected in the same process, CFD-Bench also isolates the Apache IoTDB driver's process-wide `DeprecationWarning` filter change around IoTDB discovery/connect. This keeps later TileDB warning behavior consistent with a standalone TileDB run without globally suppressing warnings.

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

## VTK backend (v16+)

VTK is a normal data backend rather than an ingest side effect. It implements W1–W11 for both canonical CFD and H5 datasets, auto-discovers steps/variables/zones from its manifest, caches loaded frames and spatial locators, uses NumPy-backed field access, and derives expensive point `Velocity` only for the W7 ROI instead of persisting a whole-mesh duplicate. The historical `--geom-engine vtk` mode is retained for comparison with database data backends and for read-only compatibility with old flat VTK files.

## Raw data

Download CFD Lifecycle Dataset (.dat files in `postprocessing/`) before ingest:

https://www.scidb.cn/en/detail?dataSetId=3553563d222d41998d7ccdd2ceff1bf9

## ODB-like HDF5 result ingest (PostgreSQL / IoTDB / TileDB / VTK)

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

# VTK only.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static --backends vtk

# Or load the same canonical H5 data to multiple backends.
cfd-bench ingest-h5 --h5 /path/to/result.h5 --datasets beam_static --backends postgresql iotdb tiledb vtk

# Runtime metadata is auto-discovered from the selected backend.
cfd-bench run --datasets beam_static
cfd-bench run --datasets beam_static --backend iotdb
cfd-bench run --datasets beam_static --backend tiledb
cfd-bench run --datasets beam_static --backend vtk
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

New HDF5 ingests materialize W3 search widths inside the selected backend: PostgreSQL uses `benchmark_max_diff`; IoTDB uses `derived.{dataset}.step_<n>.max_diff`; TileDB uses `derived/step_<n>/max_diff.tdb`. PostgreSQL retains its old-database recomputation fallback, while pre-v14 legacy CFD IoTDB/TileDB databases can still use Max_Range sidecars as a compatibility fallback. Canonical v14 CFD ingests and H5 ingests keep W3 max-diff metadata inside the selected backend and do not depend on a machine-specific Max_Range directory.

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

## Large-mesh runtime performance and observability (v12-v13)

W1-W8 keep the historical benchmark-duration semantics: `--duration` is still the target time window for each benchmark loop, not a hard per-transaction deadline. v12-v13 improve large-mesh scaling without changing the persisted PostgreSQL, IoTDB, or TileDB schemas, so existing ingests can be reused.

The runtime treats mesh topology as immutable benchmark state. IoTDB and TileDB share a small process-local static-mesh cache across workload/timestep clients, hot geometry paths operate on contiguous NumPy centroid/AABB arrays rather than rebuilding Python dictionaries, and coordinate/line/plane tests are vectorized with bounded-memory chunking for very large meshes. PostgreSQL point location keeps candidate ranking in SQL instead of downloading all centroids; PostgreSQL plane AABBs and W7 static centroid/adjacency state are reused across timesteps.

v13 further replaces the resident IoTDB/TileDB per-cell Python `dict[int, tuple]` representation with a NumPy-backed read-only Mapping facade. The public runtime mapping behavior is retained, but large meshes no longer require one permanent Python tuple per cell in addition to the NumPy geometry arrays. The uniform-grid point locator also uses compact sorted key/offset/cell-id arrays rather than a Python dictionary of bucket lists. Global AABB bounds, centers, and extents are computed once and reused by point/line/plane queries.

TileDB scalar/velocity point reads fetch requested rows/ranges instead of `A[:]`, W4/W5 fetch U/V/W in one backend operation, W8 caches per-frame variable ranges, W6 caches static surface normals and reduces pressure/normal arrays with NumPy matrix operations, W7 solves the three velocity-gradient right-hand sides in one least-squares call, and W3 performs topology/node/scalar reads on the selected submesh when possible instead of repeatedly materializing the whole mesh. These are runtime-only changes: no re-ingest is required.

### Runtime stage/progress diagnostics

Low-frequency `[stage]` messages are printed around setup and benchmark boundaries (backend connection, cold static-mesh loading/indexing, bounds resolution, benchmark-loop start/end, and total workload wall time). These markers are outside transaction hot loops and are intended to show whether a long wall-clock delay is setup, geometry, scalar I/O, or computation.

For finer diagnostics, opt in to the passive progress reporter:

```bash
cfd-bench run \
  --backend tiledb \
  --datasets structure_01 \
  --workloads w1 w2 w3 w4 w5 w6 w7 w8 \
  --progress \
  --progress-interval 5
```

`--progress` reports elapsed time, completed transactions, and the current operation phase (for example `line intersection`, `scalar query U`, `extract submesh`, or `Q-criterion`). On an interactive terminal it renders a compact progress bar; redirected/non-interactive output uses ordinary heartbeat lines. The reporter is observational only: it does not create a new deadline, cancel a transaction, alter random inputs, or change the workload stop conditions. Leave it disabled for the cleanest throughput measurement.

## IoTDB fluid linear interpolation mapping

The standalone `interpolate` command maps legacy Tecplot CFD fields from
Apache IoTDB to one or more target coordinates.  It is intentionally separate
from W1-W11 and does not change the frozen benchmark/ingest contracts.

```bash
cfd-bench interpolate \
  --datasets Kvlcc_351K_Small \
  --step 200 \
  --point -7.2 0.15 0.04 \
  --variables U V W P
```

Repeat `--point X Y Z` to map several coordinates in one IoTDB session.  Omit
`--variables` to use every CFD variable recorded in dataset metadata, and use
`--zone` only when the default result zone is not appropriate.

Legacy CFD DAT files store X/Y/Z at nodes but U/V/W/P/K/E at cell centers.  No
schema change is required: the interpolation feature derives each containing
cell vertex value at query time as the mean of all incident cell-centered
values (the same runtime projection already used by CFD W11), then evaluates a
piecewise-linear barycentric interpolation inside the cell.  For arbitrary
convex FE cells, the implementation selects a numerically stable containing
four-vertex simplex, so it does not depend on Tecplot local node ordering.
Nothing is written back to IoTDB.

The console result includes the containing dense cell / one-based Tecplot
element ID, supporting dense / one-based node IDs, barycentric weights,
coordinate reconstruction error, projected support values, interpolated
physical quantities, and a final `validation=PASS|FAIL` geometry/result check.
