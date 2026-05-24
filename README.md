# CFD-Bench

CFD-Bench: A CFD-driven benchmark for scientific data processing using database approaches (VLDB 2026).

## Package layout

```
src/cfd_bench/
  core/              DatasetKey, paths, shared types
  storage/schema/    Modern data domains (mesh_static, post_processing, derived)
  mesh_ops/          Shared geometry algorithms (W1–W8)
  infra/             Per-backend repository + mesh runtime
  API/               Unified MeshClient facade (IoTDB, TileDB, PostgreSQL, VTK)
  ingest/            DAT → database ETL (topology + cell vars + legacy dataloaders)
  workloads/         W1–W8 benchmark runners
  demo/              load_data.py, run_workloads.py
```

Install as editable package:

```bash
pip install -e ".[all]"
```

Requires Python **3.8+**.

## Data loading (ingest)

Modern ingest writes three logical domains:

| Domain | Path pattern | Used by |
|--------|--------------|---------|
| `mesh_static/{zone}/` | topology (nodes, cells, adjacency, …) | W1, W3, W6, W7 |
| `post_processing/step_{t}/` | cell_vars, node_vars | W1–W5, W8 |
| `derived/step_{t}/` | cell_qcriterion, cell_gradient | W7, W8 |

Unified pipeline:

```bash
PYTHONPATH=src python -m cfd_bench.ingest.pipeline \
  --backend tiledb --dat /path/to/200.dat --with-topology
```

Legacy bulk loader (variables only, all backends):

```bash
PYTHONPATH=src python -m cfd_bench.demo.load_data
```

## Running workloads

```bash
# 单个 workload
PYTHONPATH=src python -m cfd_bench.workloads.w1.run --ships JBC_615k

# 全部 W1–W8
PYTHONPATH=src python -m cfd_bench.demo.run_workloads --ships JBC_615k
```

兼容旧入口（转发至 cfd_bench）：

```bash
PYTHONPATH=src python -m demo.workload
PYTHONPATH=src python -m demo.LoadDataTo_DB
```

## API usage

```python
from cfd_bench.API.iotdb_api import IoTDBMeshClient

client = IoTDBMeshClient()
ctx = client.connect("JBC_615k", step=200, zone="0_Fluid")
cells = client.point_intersection(points)
vals = client.point_query(cells, "P")
client.close()
```

## 独立压测

`iotdb_test/`、`tiledb_test/` 为独立压测脚本，已改用 `cfd_bench.API.*`，不属于正式 benchmark 主路径。

## Raw data

Download CFD Lifecycle Dataset (.dat files in `postprocessing/`) before ingest:
https://www.scidb.cn/en/detail?dataSetId=3553563d222d41998d7ccdd2ceff1bf9
