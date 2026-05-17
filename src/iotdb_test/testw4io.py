## import DB_Interface_PG as PG_API  # IoTDB-only mode
import DB_Interface_IoTDB as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import os
import random
import time
import re
import vtk
import psutil  # 用于监测资源

# --- 综合资源监测类 ---
class UnifiedResourceMonitor:
    def __init__(self, label, target_process_names=None):
        self.label = label
        self.target_names = target_process_names
        self.main_process = psutil.Process(os.getpid())
        
    def _get_total_stats(self):
        # 统计当前 Python 进程
        total_mem = self.main_process.memory_info().rss
        total_read = self.main_process.io_counters().read_bytes
        
        # 累加指定数据库进程的消耗
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
        print(f"  内存变化: {end_mem - self.start_mem:+.2f} MB (全系统相关进程总占用: {end_mem:.2f} MB)")
        print(f"  磁盘读取: {max(0, end_read - self.start_read):.2f} MB")
        print("-" * 55)

# --- 辅助函数 ---
def random_start(vtk_mesh: VTK_API.vtkUnstructuredGrid):
    vtk_itf = VTK_API.VTK_Interface()
    while True:
        bounds = vtk_mesh.GetBounds()
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[2], bounds[3])
        z = random.uniform(bounds[4], bounds[5])
        cell_id = vtk_itf.vtk_point_intersection(vtk_mesh, np.array([[x, y, z]], dtype=np.float64))
        if cell_id.size == 1:
            break
    return cell_id, np.array([x, y, z], dtype=np.float64)

def cal_next_point(current_point: NDArray[np.float64], velocity: NDArray[np.float64], delta_t: float = 0.01) -> NDArray[np.float64]:
    return current_point + velocity * delta_t

def main():
    vtk.vtkObject.GlobalWarningDisplayOff()
    ''' Pre-processing '''
    iotdb_api = IoTDB_API.Iotdb_Interface(os.getenv("SHIP_TO_RUN", "JBC_615k"), os.getenv("STEP_TO_RUN", "200"))
    iotdb_entity = iotdb_api.iotdb_connect()
    vtk_api = VTK_API.VTK_Interface()

    # 配合自动化脚本，优先从环境变量读取船型
    env_ship = os.getenv("SHIP_TO_RUN")
    if env_ship:
        SHIP_TYPE_LIST = [env_ship]
    else:
        SHIP_TYPE_LIST = ["JBC_615k"]

    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VTK_MESH_DIR = "../../vtk_dir/"
    TileDB_DIR = "../../TileDB_Instances/"
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]

    for ship_type in SHIP_TYPE_LIST:
        # 移除了 ship_type 的普通打印，改为报告内体现

        if ship_type in ["JBC_3843k", "Kvlcc2_3709k"]:
            valid_time_steps = [ts for ts in TIME_STEP_LIST if ts != "2000"]
        else:
            valid_time_steps = TIME_STEP_LIST.copy()

        vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith("_200.vtk")]
        vtk_mesh = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        # IoTDB-only mode: skip PostgreSQL/TileDB/VTK sections below.
        with UnifiedResourceMonitor("IoTDB", target_process_names=["iotdb", "java"]): 
            iotdb_api.set_ship_type(ship_type)
            iotdb_transaction = 0
            skip_current_dataset = False
            start_time = time.time()
            while time.time() - start_time < 60: 
                current_cell_indexe, current_coordinate = random_start(vtk_mesh)
                for time_step in valid_time_steps:
                    iotdb_api.set_time_step(time_step)
                    u_vals = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "U")
                    v_vals = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "V")
                    w_vals = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "W")
                    if any(v is None or len(v) == 0 for v in [u_vals, v_vals, w_vals]):
                        print(f"SKIP_DATASET_EMPTY: Ship={ship_type} Stage=Trajectory Step={time_step}")
                        skip_current_dataset = True
                        break
                    u = u_vals[0]
                    v = v_vals[0]
                    w = w_vals[0]

                    velocity = np.array([u, v, w], dtype=np.float64)
                    next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)
                    next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                    
                    if next_cell_indexe.size == 0:
                        break
                    current_cell_indexe = next_cell_indexe
                    current_coordinate = next_coordinate
                if skip_current_dataset:
                    break
                iotdb_transaction += 1
            if skip_current_dataset:
                continue
            # print(f"Total IoTDB Transactions ({ship_type}): {iotdb_transaction}")
        continue

        ''' W 4.3  Testing TileDB '''
        with UnifiedResourceMonitor("TileDB", target_process_names=None):
            ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
            tiledb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                current_cell_indexe, current_coordinate = random_start(vtk_mesh)
                for time_step in valid_time_steps:
                    tiledb_file, = [f for f in os.listdir(ship_tiledb_dir) if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)]
                    tdb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))
                    
                    u, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, "U", current_cell_indexe)
                    v, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, "V", current_cell_indexe)
                    w, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, "W", current_cell_indexe)

                    velocity = np.array([u, v, w], dtype=np.float64)
                    next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)
                    next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                    
                    if next_cell_indexe.size == 0:
                        break
                    current_cell_indexe = next_cell_indexe
                    current_coordinate = next_coordinate
                tiledb_transaction += 1
            # print(f"Total TileDB Transactions ({ship_type}): {tiledb_transaction}")

        ''' W 4.4  Testing VTK '''        
        with UnifiedResourceMonitor("VTK", target_process_names=None):
            vtk_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                current_cell_indexe, current_coordinate = random_start(vtk_mesh)
                for time_step in valid_time_steps:
                    vtk_query_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
                    vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))

                    u, = vtk_api.point_query(vtk_entity, current_cell_indexe, "U")
                    v, = vtk_api.point_query(vtk_entity, current_cell_indexe, "V")
                    w, = vtk_api.point_query(vtk_entity, current_cell_indexe, "W")

                    velocity = np.array([u, v, w], dtype=np.float64)
                    next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)
                    next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                    
                    if next_cell_indexe.size == 0:
                        break
                    current_cell_indexe = next_cell_indexe
                    current_coordinate = next_coordinate
                vtk_transaction += 1
            # print(f"Total VTK Transactions ({ship_type}): {vtk_transaction}")

if __name__ == "__main__":
    main()