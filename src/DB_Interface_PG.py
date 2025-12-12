import numpy as np
from numpy.typing import NDArray
import psycopg2
from psycopg2 import sql
import pandas as pd

class PG_Interface:
    def __init__(self, ship_type: str = None, ship_scale: str = None, zone_type: str = None, time_step: str = None):
        """
        初始化 PostgreSQL 接口
        Args:
            ship_type: 船舶类型（如 JBC、Kvlcc2）
            ship_scale: 船舶规模（如 615k、3709k）
            zone_type: 区域类型（如 fluid、hull）
            time_step: 时间步（如 2、4）
        """
        self.ship_type = ship_type
        self.ship_scale = ship_scale
        self.zone_type = zone_type
        self.time_step = time_step
        self.db_entity = None

    @staticmethod
    def pg_connect(db_name="cae_data2", db_user="postgres", db_password="123456", db_host="localhost", db_port="5432"):
        """
        连接到 PostgreSQL 数据库
        """
        try:
            db_entity = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
            )
            # print("PostgreSQL 连接成功")
            return db_entity
        except psycopg2.Error as e:
            print(f"PostgreSQL 连接失败: {e}")
            return None
        
    def set_ship_type(self, ship_type: str):
        """设置船舶类型"""
        self.ship_type = ship_type

    def set_ship_scale(self, ship_scale: str):
        """设置船舶规模"""
        self.ship_scale = ship_scale

    def set_zone_type(self, zone_type: str):
        """设置区域类型"""
        self.zone_type = zone_type
    
    def set_time_step(self, time_step: str):
        """设置时间步"""
        self.time_step = time_step
    
    def point_query(self, db_entity, cell_indexes: np.array, attribute_name: str) -> NDArray[np.float64]:
        """
        点查询：根据单元索引查询单个属性的值
        注意：PostgreSQL中cell_indexes 使用 1-based 索引（从1开始）
        
        Args:
            db_entity: 数据库连接
            cell_indexes: 单元索引数组（1-based，从1开始）
            attribute_name: 要查询的属性名（如 'p', 'u' 等）
        
        Returns:
            属性值的 NDArray
        """
        if not db_entity:
            raise ValueError("数据库连接无效")
        
        if not self.ship_type or not self.ship_scale or not self.zone_type or not self.time_step:
            raise ValueError("请先设置 ship_type、ship_scale、zone_type 和 time_step")
        
        cursor = db_entity.cursor()
        
        try:
            # 查询该属性的数据数组
            sql = """
                SELECT data
                FROM cae_simulation_data 
                WHERE ship_type = %s 
                AND scale = %s
                AND zone_type = %s
                AND timestep = %s 
                AND variable = %s
                AND is_element = TRUE
            """
            
            cursor.execute(sql, (self.ship_type, self.ship_scale, self.zone_type, int(self.time_step), attribute_name))
            row = cursor.fetchone()
            
            if row and row[0]:  # 有数据
                data_array = row[0]  # PostgreSQL 数组
                
                # 提取指定索引的值
                values = []
                for idx in cell_indexes:
                    if 0 <= int(idx) < len(data_array):
                        values.append(float(data_array[int(idx)]))
                    else:
                        values.append(float('nan'))
                
                result = np.array(values, dtype=np.float64)
                # print(f"点查询完成: {attribute_name}[{cell_indexes}] -> {len(values)} 个值")
                return result
            else:
                print(f"警告: 未找到属性 '{attribute_name}' 的数据")
                return np.full(len(cell_indexes), float('nan'), dtype=np.float64)
                
        except Exception as e:
            print(f"点查询失败: {e}")
            return np.array([], dtype=np.float64)
        finally:
            cursor.close()
    
    def range_query_var(self, db_entity, lower_bound: float, upper_bound: float, attribute_name: str) -> np.array:
        """
        变量范围查询：基于变量值的范围查询
        
        Args:
            db_entity: 数据库连接
            lower_bound: 下限值
            upper_bound: 上限值
            attribute_name: 变量名
        
        Returns:
            符合条件的单元索引数组
        """
        if not db_entity:
            raise ValueError("数据库连接无效")
        
        if not self.ship_type or not self.ship_scale or not self.zone_type or not self.time_step:
            raise ValueError("请先设置 ship_type、ship_scale、zone_type 和 time_step")
        
        cursor = db_entity.cursor()
        
        try:
            # 查询该属性的数据数组
            sql = """
                SELECT data
                FROM cae_simulation_data 
                WHERE ship_type = %s 
                AND scale = %s
                AND zone_type = %s
                AND timestep = %s 
                AND variable = %s
                AND is_element = TRUE
            """
            
            cursor.execute(sql, (self.ship_type, self.ship_scale, self.zone_type, int(self.time_step), attribute_name))
            row = cursor.fetchone()
            
            if row and row[0]:
                data_array = row[0]
                
                # 找出值在范围内的索引
                cell_indexes = []
                for idx, value in enumerate(data_array):
                    if lower_bound <= float(value) <= upper_bound:
                        cell_indexes.append(idx)
                
                result = np.array(cell_indexes, dtype=np.int32)
                # print(f"变量范围查询: {attribute_name} ∈ [{lower_bound}, {upper_bound}] 找到 {len(result)} 个单元")
                return result
            else:
                print(f"警告: 未找到属性 '{attribute_name}' 的数据")
                return np.array([], dtype=np.int32)
                
        except Exception as e:
            print(f"变量范围查询失败: {e}")
            return np.array([], dtype=np.int32)
        finally:
            cursor.close()    

    def range_query_coord(self, db_entity, lower_bound: NDArray[np.float64], upper_bound: NDArray[np.float64]) -> np.array:
        """
        坐标范围查询：基于三维坐标的范围查询
        
        Args:
            db_entity: 数据库连接
            lower_bound: 三维坐标下限 [x_min, y_min, z_min]
            upper_bound: 三维坐标上限 [x_max, y_max, z_max]
        
        Returns:
            符合条件的单元索引数组
        """
        if not db_entity:
            raise ValueError("数据库连接无效")
        
        if len(lower_bound) != 3 or len(upper_bound) != 3:
            raise ValueError("坐标范围必须是三维数组")
        
        if not self.ship_type  or not self.ship_scale or not self.zone_type:
            raise ValueError("请先设置 ship_type、ship_scale 和 zone_type")
        
        cursor = db_entity.cursor()
        
        try:
            # 分别查询 X, Y, Z 坐标的数据
            coords_data = {}
            for coord_name in ['X', 'Y', 'Z']:
                sql = """
                    SELECT data
                    FROM cae_simulation_data 
                    WHERE ship_type = %s 
                    AND scale = %s
                    AND zone_type = %s
                    AND variable = %s
                    AND is_element = TRUE
                    limit 1
                """
                
                cursor.execute(sql, (self.ship_type, self.ship_scale, self.zone_type, coord_name))
                row = cursor.fetchone()
                if row and row[0]:
                    coords_data[coord_name] = row[0]
                else:
                    print(f"警告: 未找到坐标 '{coord_name}' 的数据")
                    return np.array([], dtype=np.int32)
            
            # X BETWEEN ... AND Y BETWEEN ... AND Z BETWEEN ...
            x_lower, y_lower, z_lower = lower_bound
            x_upper, y_upper, z_upper = upper_bound
            
            x_data = coords_data['X']
            y_data = coords_data['Y']
            z_data = coords_data['Z']
            
            # 找出同时满足三个坐标范围的索引
            cell_indexes = []
            for idx in range(len(x_data)):
                x_val = float(x_data[idx])
                y_val = float(y_data[idx])
                z_val = float(z_data[idx])
                
                if (x_lower <= x_val <= x_upper and 
                    y_lower <= y_val <= y_upper and 
                    z_lower <= z_val <= z_upper):
                    cell_indexes.append(idx)  
            
            result = np.array(cell_indexes, dtype=np.int32)
            # print(f"坐标范围查询: X∈[{x_lower}, {x_upper}], Y∈[{y_lower}, {y_upper}], Z∈[{z_lower}, {z_upper}] 找到 {len(result)} 个单元")
            return result
            
        except Exception as e:
            print(f"坐标范围查询失败: {e}")
            return np.array([], dtype=np.int32)
        finally:
            cursor.close()    