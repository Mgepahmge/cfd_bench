from operator import index
import os
from tkinter import Variable
from xml.dom.minidom import Element
from numpy import var
from Dat_Data_Decoder import CAE_Decoder
import psycopg2
from psycopg2 import OperationalError
import Zone
import pandas as pd
import numpy as np
from Zone import Zone_3D
from Dat_Data_Decoder import print_array
from pathlib import Path
from collections import defaultdict


def create_postgres_connection(db_name="cae_data", db_user="postgres", db_password="123456", db_host="localhost",
                               db_port="5432"):
    """
    Create a connection to the PostgreSQL database.
    """
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def create_table(connection):
    """
    创建表
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS cae_simulation_data (
        ship_type VARCHAR(50) NOT NULL,
        scale VARCHAR(50) NOT NULL,
        zone_type VARCHAR(50) NOT NULL,
        timestep INTEGER NOT NULL,
        variable VARCHAR(50) NOT NULL,
        is_element BOOLEAN NOT NULL,
        data DOUBLE PRECISION[] NOT NULL,
        PRIMARY KEY (ship_type, scale, timestep, variable, is_element, zone_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ship_timestep ON cae_simulation_data(ship_type, timestep);
    CREATE INDEX IF NOT EXISTS idx_variable_search ON cae_simulation_data(variable, is_element);
    """
    
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_sql)
        connection.commit()
        print("表结构创建成功")
    except Exception as e:
        print(f"创建表结构时出错: {e}")
        connection.rollback()
    finally:
        if cursor:
            cursor.close()

def insert_cae_simulation_data(connection, df, table_name="cae_simulation_data"):
    """
    插入CAE数据
    """
    if not connection:
        print("No valid database connection")
        return

    columns = ["ship_type", "scale", "zone_type", "timestep", "variable", "is_element", "data"]
    df_reordered = df[columns]
    cols_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(columns))
    
    sql = f"""
    INSERT INTO {table_name} ({cols_str}) 
    VALUES ({placeholders}) 
    ON CONFLICT (ship_type, scale, timestep, variable, is_element, zone_type) 
    DO UPDATE SET 
        data = EXCLUDED.data;
    """

    try:
        cursor = connection.cursor()
        
        data = []
        for row in df_reordered.values:
            row_converted = []

            for row_idx, row in enumerate(df_reordered.values):
                row_converted = []
                for col_idx, val in enumerate(row):
                    col_name = columns[col_idx]
                    
                    if col_name == "data":
                        # 处理数据数组，保持高精度
                        if isinstance(val, (list, np.ndarray)):
                            if isinstance(val, np.ndarray):
                                # 使用高精度转换
                                if val.dtype == np.float32:
                                    # 将float32转换为高精度的Python float
                                    row_converted.append([float(x) for x in val])
                                else:
                                    row_converted.append([float(f"{x:.15g}") for x in val])
                            else:
                                # 对于Python列表，直接转换
                                row_converted.append([float(f"{x:.15g}") if isinstance(x, (int, float)) else x for x in val])
                        elif pd.isna(val) or val is None:
                            row_converted.append(None)
                        else:
                            print(f"警告: 数据列类型异常: {type(val)}, 值: {val[:10] if isinstance(val, (list, np.ndarray)) else val}")
                            row_converted.append([])
                    
                    elif col_name == "timestep":
                        # 确保timestep是整数
                        row_converted.append(int(val) if not pd.isna(val) else 0)
                    
                    elif col_name == "is_element":
                        # 确保布尔值
                        row_converted.append(bool(val) if not pd.isna(val) else False)
                    
                    else:
                        # 其他列保持原样
                        if pd.isna(val) or val is None:
                            row_converted.append(None)
                        elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                            row_converted.append(int(val))
                        elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                            # 使用高精度字符串表示
                            row_converted.append(float(f"{val:.15g}"))
                        elif isinstance(val, (int, float, str, bool)):
                            row_converted.append(val)
                        else:
                            print(f"警告: 遇到未知数据类型 {type(val)}，值：{val}")
                            row_converted.append(str(val))
                
                data.append(tuple(row_converted))

            # 批量插入
            cursor.executemany(sql, data)
            connection.commit()
            # print(f"成功插入/更新 {cursor.rowcount} 行数据到 {table_name}")
            
    except Exception as e:
        print(f"插入数据时出错: {e}")
        connection.rollback()
        import traceback
        traceback.print_exc()
    finally:
        if cursor:
            cursor.close()

        # # 调试信息：显示第一条数据
        # if data:
        #     print(f"第一条插入数据预览:")
        #     for i, col in enumerate(columns):
        #         if col == "data":
        #             print(f"  {col}: 数组长度={len(data[0][i])}, 前5个值={data[0][i][:5]}")
        #         else:
        #             print(f"  {col}: {data[0][i]}")

def close_postgres_connection(connection):
    """
    Close the PostgreSQL database connection.
    """
    if connection:
        connection.close()
        print("PostgreSQL connection closed")

def extract_ship_info_from_path(file_path):
    """
    从文件路径中提取船舶类型和尺度信息
    路径示例: data/gpudb/downloaded/JBC_615k/Postprocessing/200.dat
    """
    path_parts = Path(file_path).parts
    
    # 定义已知的船舶类型
    known_ship_types = ['JBC', 'Kvlcc2', 'Suboff']
    
    for part in path_parts:
        if '_' in part:
            for ship_type in known_ship_types:
                if ship_type in part:
                    try:
                        _, scale_part = part.split('_')
                        # 验证尺度格式 (数字+k)
                        if scale_part[:-1].isdigit() and scale_part.endswith('k'):
                            return ship_type, scale_part
                    except ValueError:
                        continue
    # 如果无法解析，返回默认值
    print(f"警告: 无法从路径解析船舶信息: {file_path}")
    return "Unknown", "000k"

def find_dat_files(base_path):
    """
    递归查找所有.dat文件，并提取船舶信息
    """
    dat_files_with_info = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.dat'):
                file_path = os.path.join(root, file)
                
                # 从路径中提取船舶类型和尺度
                ship_type, scale = extract_ship_info_from_path(file_path)
                
                # 从文件名提取时间步 (去掉.dat后缀)
                timestep_str = os.path.splitext(file)[0]
                try:
                    timestep = int(timestep_str)
                except ValueError:
                    timestep = 0
                    print(f"警告: 无法从文件名解析时间步: {file}")
                
                dat_files_with_info.append({
                    'file_path': file_path,
                    'ship_type': ship_type,
                    'scale': scale,
                    'timestep': timestep
                })
    
    return dat_files_with_info


def main(input_path=None):
    # 配置
    CONFIG = {
        "txt_output_dir": "Decoded_Data/",
        "postgres": {
            "HOST": "localhost",
            "PORT": "5432",
            "USER": "postgres",
            "PASSWORD": "123456",
            "DATABASE": "cae_data"
        }
    }

    # 检查dat目录
    if input_path is not None:
        base_path = input_path
    else:
        base_path = '/data/gpudb/downloaded/'  # 根据结构调整
    
    print(f"开始在基础路径下搜索.dat文件: {base_path}")

    if not os.path.exists(base_path):
        print(f"错误: 路径不存在: {base_path}")
        return

    # 查找所有dat文件
    dat_files_info = find_dat_files(base_path)
    
    if not dat_files_info:
        print(f"在 {base_path} 下未找到.dat文件")
        return

    # print(f"找到 {len(dat_files_info)} 个.dat文件")
    
    # # 按船舶类型和尺度分组显示
    # ship_groups = defaultdict(list)
    # for info in dat_files_info:
    #     key = f"{info['ship_type']}_{info['scale']}"
    #     ship_groups[key].append(info['timestep'])
    
    # for ship_key, timesteps in ship_groups.items():
    #     print(f"  {ship_key}: {len(timesteps)}个时间步, 时间步范围: {min(timesteps)}-{max(timesteps)}")

    # 创建数据库连接
    connection = create_postgres_connection(
        db_name=CONFIG["postgres"]["DATABASE"],
        db_user=CONFIG["postgres"]["USER"],
        db_password=CONFIG["postgres"]["PASSWORD"],
        db_host=CONFIG["postgres"]["HOST"],
        db_port=CONFIG["postgres"]["PORT"]
    )

    if not connection:
        print("无法连接到数据库")
        return
    
    processed_count = 0
    zone_count = 0

    try:
        # 创建表结构
        create_table(connection)
        
        # processed_count = 0
        for file_info in dat_files_info:
            file_path = file_info['file_path']
            ship_type = file_info['ship_type']
            scale = file_info['scale']
            timestep = file_info['timestep']
            
            print(f"\n处理文件: {file_path}")
            print(f"  船舶类型: {ship_type}, 尺度: {scale}, 时间步: {timestep}")

            try:
                # 解析dat文件
                data = CAE_Decoder(3)
                data.Decode_dat_file(file_path)

                if not data.Zones:
                    print(f"  警告: 文件 {file_path} 没有解析到区域数据")
                    continue

                # fluid_zone = data.Zones[0]

                # # 使用从路径提取的信息创建DataFrame
                # dataframes = fluid_zone.to_dataframes(
                #     ship_type=ship_type, 
                #     scale=scale
                # )

                # 处理所有区域，但只导入fluid和hull区域
                for zone_idx, zone in enumerate(data.Zones):
                    zone_name = zone.Zone_name
                    zone_type = zone.Zone_type
                    
                    # 判断是否为fluid或hull区域
                    zone_name_lower = zone_name.lower()
                    zone_type_lower = zone_type.lower()
                    
                    is_fluid_zone = 'fluid' in zone_name_lower or 'fluid' in zone_type_lower
                    is_hull_zone = 'hull' in zone_name_lower or 'hull' in zone_type_lower
                    
                    if not (is_fluid_zone or is_hull_zone):
                        print(f"  跳过区域 [{zone_idx}]: {zone_name} ({zone_type}) - 不是fluid或hull区域")
                        continue
                    
                    # 确定最终的zone_type值
                    if is_fluid_zone:
                        final_zone_type = 'fluid'
                    elif is_hull_zone:
                        final_zone_type = 'hull'
                    else:
                        final_zone_type = zone_type

                    # print(f"  导入区域 [{zone_idx}]: {zone_name} ({zone_type})")
                    # print(f"    单元数: {zone.Element_count}, 节点数: {zone.Node_count}")
                    
                    try:
                        df = zone.to_postgresql_dataframe(
                            ship_type=ship_type,
                            scale=scale,
                            timestep=timestep,
                            zone_type=final_zone_type
                        )
                        
                        if df.empty:
                            print(f"    警告: 区域数据为空")
                            continue
                        
                        # 验证数据长度
                        for _, row in df.iterrows():
                            if row['is_element']:
                                expected_len = zone.Element_count
                            else:
                                expected_len = zone.Node_count
                            
                            actual_len = len(row['data'])
                            if actual_len != expected_len:
                                print(f"    警告: {row['variable']} 数据长度不匹配: 期望{expected_len}, 实际{actual_len}")
                        
                        # print(f"    准备插入 {len(df)} 条记录到PostgreSQL")
                        
                        # 插入到数据库
                        insert_cae_simulation_data(connection, df, "cae_simulation_data")
                        
                        zone_count += 1
                        
                    except Exception as e:
                        print(f"    处理区域时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                processed_count += 1

            except Exception as e:
                print(f"  处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # print(f"\n处理完成!")
        # print(f"  成功处理 {processed_count}/{len(dat_files_info)} 个文件")
        # print(f"  成功导入 {zone_count} 个区域 (只包含fluid和hull区域)")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_postgres_connection(connection)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, 
                       default='/data/gpudb/downloaded/',
                       help='Input base path for .dat files')
    args = parser.parse_args()
    main(args.input_path)