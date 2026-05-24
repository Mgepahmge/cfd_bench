## import DB_Interface_PG as PG_API  # IoTDB-only mode
from cfd_bench.infra.iotdb import legacy_variables_client as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import vtk
import random
import os
from vtk import vtkUnstructuredGrid
import time
import re
import psutil  # 用于监测资源

# --- 综合资源监测类 ---
class UnifiedResourceMonitor:
    def __init__(self, label, target_process_names=None):
        self.label = label
        self.target_names = target_process_names
        self.main_process = psutil.Process(os.getpid())
        
    def _get_total_stats(self):
        # 统计当前 Python 进程消耗
        total_mem = self.main_process.memory_info().rss
        total_read = self.main_process.io_counters().read_bytes
        
        # 累加指定数据库后端服务进程的消耗
        if self.target_names:
            for proc in psutil.process_iter(['name', 'pid']):
                try:
                    if any(name in proc.info['name'].lower() for name in self.target_names):
                        p = psutil.Process(proc.info['pid'])
                        total_mem += p.memory_info().rss
                        total_read += p.io_counters().read_bytes
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return total_mem / 1024 / 1024, total_read / 1024 / 1024

    def __enter__(self):
        self.start_time = time.time()
        self.start_mem, self.start_read = self._get_total_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_mem, end_read = self._get_total_stats()
        print(f"\n>>>> [综合性能报告 - {self.label}] <<<<")
        print(f"  耗时: {end_time - self.start_time:.2f} s")
        print(f"  内存变化: {end_mem - self.start_mem:+.2f} MB (当前总占用: {end_mem:.2f} MB)")
        print(f"  磁盘读取: {max(0, end_read - self.start_read):.2f} MB")
        print("-" * 55)

# --- 辅助函数 ---
def random_range(vtk_mesh: vtkUnstructuredGrid) -> NDArray[np.float64]:
    bounds = vtk_mesh.GetBounds()
    x1, x2 = sorted([random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])])
    y1, y2 = sorted([random.uniform(bounds[2], bounds[3]), random.uniform(bounds[2], bounds[3])])
    z1, z2 = sorted([random.uniform(bounds[4], bounds[5]), random.uniform(bounds[4], bounds[5])])
    return np.array([[x1, y1, z1], [x2, y2, z2]])

def aggregation(vals: NDArray[np.float64]):
    if len(vals) == 0: return 0, 0, 0
    return np.mean(vals), np.max(vals), np.min(vals)

def main():
    vtk.vtkObject.GlobalWarningDisplayOff()
    ''' Pre-processing '''
    iotdb_api = IoTDB_API.Iotdb_Interface(os.getenv("SHIP_TO_RUN", "JBC_615k"), os.getenv("STEP_TO_RUN", "200"))
    iotdb_entity = iotdb_api.iotdb_connect()
    vtk_api = VTK_API.VTK_Interface()

    # 配合自动化脚本，优先从环境变量读取船型
    env_ship = os.getenv("SHIP_TO_RUN")
    SHIP_TYPE_LIST = [env_ship] if env_ship else ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]

    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VARIABLE_LIST = ["U", "V", "W", "P", "K", "E"]
    VTK_MESH_DIR = "../../vtk_dir/"
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    TileDB_DIR = "../../TileDB_Instances/"

    for ship_type in SHIP_TYPE_LIST:
        if ship_type in ["JBC_3843k", "Kvlcc2_3709k"]:
            valid_time_steps = [ts for ts in TIME_STEP_LIST if ts != "2000"]
        else:
            valid_time_steps = TIME_STEP_LIST.copy()

        vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith("_200.vtk")]
        vtk_mesh = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        # IoTDB-only mode: skip PostgreSQL/TileDB/VTK sections below.
        with UnifiedResourceMonitor(f"IoTDB_{ship_type}", target_process_names=["iotdb", "java"]):
            iotdb_api.set_ship_type(ship_type)
            iotdb_transaction = 0
            skip_current_dataset = False
            start = time.time()
            while time.time() - start < 60:
                attribute_name = random.choice(VARIABLE_LIST)
                while True:
                    lower_bound, upper_bound = random_range(vtk_mesh)
                    cell_indexes = iotdb_api.range_query_coord(iotdb_entity, lower_bound, upper_bound)
                    if cell_indexes.size > 0: break

                iotdb_result = []
                for time_step in valid_time_steps:
                    iotdb_api.set_time_step(time_step)
                    iotdb_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
                    if iotdb_vals is None or len(iotdb_vals) == 0:
                        print(f"SKIP_DATASET_EMPTY: Ship={ship_type} Stage=PointQuery Step={time_step}")
                        skip_current_dataset = True
                        break
                    iotdb_result.extend(iotdb_vals)
                if skip_current_dataset:
                    break
                aggregation(iotdb_result)
                iotdb_transaction += 1
            if skip_current_dataset:
                continue
            print(f"Total IoTDB Transactions ({ship_type}): {iotdb_transaction}")
        continue

        ''' W 2.3  Testing TileDB '''
        with UnifiedResourceMonitor(f"TileDB_{ship_type}"):
            ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
            tiledb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, "200fluid.tdb"))

            tiledb_transaction = 0
            start = time.time()
            while time.time() - start < 60:
                attribute_name = random.choice(VARIABLE_LIST)
                while True:
                    lower_bound, upper_bound = random_range(vtk_mesh)
                    cell_indexes = tdb_api.Spatial_Range_Query_TileDB(tiledb_entity, lower_bound, upper_bound)
                    if cell_indexes.size > 0: break

                tiledb_result = []
                for time_step in valid_time_steps:
                    tiledb_file, = [f for f in os.listdir(ship_tiledb_dir) if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)]
                    tdb_ent = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))
                    tiledb_vals = tdb_api.Point_Query_Attribute_TileDB(tdb_ent, attribute_name, cell_indexes)
                    tiledb_result.extend(tiledb_vals)
                aggregation(tiledb_result)
                tiledb_transaction += 1
            print(f"Total TileDB Transactions ({ship_type}): {tiledb_transaction}")

        ''' W 2.4  Testing VTK '''
        with UnifiedResourceMonitor(f"VTK_{ship_type}"):
            vtk_transaction = 0
            start = time.time()
            while time.time() - start < 60:
                attribute_name = random.choice(VARIABLE_LIST)
                while True:
                    lower_bound, upper_bound = random_range(vtk_mesh)
                    cell_indexes = vtk_api.range_query_coord(vtk_mesh, lower_bound, upper_bound)
                    if cell_indexes.size > 0: break
                
                vtk_result = []
                for time_step in valid_time_steps:
                    vtk_query_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
                    vtk_ent = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))
                    vtk_vals = vtk_api.point_query(vtk_ent, cell_indexes, attribute_name)
                    vtk_result.extend(vtk_vals)
                aggregation(vtk_result)
                vtk_transaction += 1
            print(f"Total VTK Transactions ({ship_type}): {vtk_transaction}")

if __name__ == "__main__":
    main()