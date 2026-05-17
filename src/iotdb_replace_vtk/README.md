# IoTDB Replace VTK

`iotdb_replace_vtk` 是 IoTDB 原生接口层，使用 `iotdb_*` API 提供与 `VTK_Interface` 等价的核心能力。

## 目录结构

- `iotdb_mesh_interface.py`: 主入口 `IoTDBMeshInterface`
- `repository.py`: IoTDB 查询与路径封装
- `mesh_runtime.py`: 静态网格缓存层
- `geometry_ops.py`: 点/线/面相交
- `isosurface_ops.py`: 等值面提取（轻量实现）
- `submesh_ops.py`: 子网格提取
- `normals.py`: 法向计算
- `types.py`: `MeshContext` / `LiteMesh` / `LitePolyData`

## 依赖数据域

静态几何域（必需）：

- `root.simulation_data.mesh_static.<dataset_key>.<zone>.nodes`
- `root.simulation_data.mesh_static.<dataset_key>.<zone>.cells`
- `root.simulation_data.mesh_static.<dataset_key>.<zone>.cell_nodes`
- `root.simulation_data.mesh_static.<dataset_key>.<zone>.cell_adjacency`
- `root.simulation_data.mesh_static.<dataset_key>.<zone>.face_planes`

时序变量域（必需）：

- `root.simulation_data.post_processing_management.<dataset_key>.step_<t>.cell_vars`

## 快速示例

```python
import numpy as np
from iotdb_replace_vtk import IoTDBMeshInterface

with IoTDBMeshInterface() as api:
    ctx = api.iotdb_connect(dataset_key="JBC_615k", step=0, zone="0_Fluid")
    ids = api.iotdb_line_intersection([0, 0, 0], [1, 1, 1])
    vals = api.iotdb_point_query(ids[:100], "P")
    print(ctx.available_caps, ids.shape, vals.shape)
```

## 拓扑导入脚本

在 `Benchmark/Dat_Data_Decoder` 中新增：

- `load_mesh_topology_to_iotdb.py`
- `build_node_vars_to_iotdb.py`
- `build_boundary_faces_to_iotdb.py`
- `materialize_qcriterion_to_iotdb.py`

这些脚本用于补齐 W1-W8 所需静态拓扑和派生层数据。
