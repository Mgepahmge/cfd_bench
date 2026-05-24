from genericpath import isdir
from . import Zone
import numpy as np
from tqdm import tqdm, tqdm_gui
import os
import pandas as pd
#from scipy.sparse import coo_matrix


class CAE_Decoder:
    #Func: Decode_dat_file
    #Inputs:
    #   path: the path of the dat file.
    #   return: the decoded CAE data structure.
    #   description: Decode a given CAE data. The file structure of the .dat file is default.
    Title = ""
    Variables = []
    Var_count = -1
    Zones = []
    N_DIM = -1
    def __init__(self, DIM):
        # Re-initialize the object everytime to avoid left over problems in previous iterations.
        self.Title = ""
        self.Variables = []
        self.Var_count = -1
        self.Zones = []
        self.N_DIM = DIM
        return

    # 文件读取
    def Decode_dat_file(self, path):
        file_object = open(path, 'r', encoding="UTF-8")
        raw_content = file_object.read()
        file_object.close()
        
        paragraphs = raw_content.split("ZONE  T=")
        
        # Processing header first, extracting title and variables:
        header_lines = paragraphs[0].split("\n")
        for line in header_lines:
            line = line.strip()
            # Extracting title
            if line.startswith('TITLE'):
                tokens = line.split('=')
                self.Title = tokens[1].replace('"', '')
                print(f"Title: {self.Title}")
            
            # Extracting variables
            if line.startswith('VARIABLES'):
                tokens = line.split('=')
                _vars = tokens[1].replace('"', '').split(' ')
                for var in _vars:
                    self.Variables.append(var)
                self.Var_count = len(self.Variables)
                print(f"Variables: {self.Variables}, Count: {self.Var_count}")
        
        # Each of the remaining paragraphs represent a ZONE, process them using the same logic:
        for i in range(1, len(paragraphs)):
            if i > 2:
                # Thus far we only decode the first Zone, i.e. fluid zone. The rest of the zones have problem understanding their organization
                # consider the rest as future works.
                break
            #Decoding Each Zone
            paragraph = paragraphs[i]
            zone = Zone.Zone_3D(paragraph, self.Var_count, self.N_DIM, self.Variables)
            self.Zones.append(zone)

        return
    
    

def print_array(arr, N, name):
    with open(name + ".txt", "w") as f:
        f.write(f"{N}\n")
        for item in arr:
            f.write(f"{item}\n")

def main(input_path = None):
    PRINT_DAT = True
    if input_path is not None:
        path = input_path
    else:
        path = 'tecplot/'
    print("Decoding Post-processing data")
    if not os.path.exists(path):
        print(f"Error, path does not exist: {path}")
        return
    elif not os.path.isdir(path):
        print(f"Error, please input the directory to which the .dat files are located: {path}")

    #处理每个.dat文件
        
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):  # Process Files only

            # Check if the filename ends with .dat
            if filepath.lower().endswith('.dat'):
                print(f"Processing file: {filepath}")
                data = CAE_Decoder(3)
                data.Decode_dat_file(filepath)
                # Thus far, only the fluid zone is our interest   

                


                fluid_zone:Zone.Zone_3D = data.Zones[0]   #Zones[0]待修改
                hull_zone:Zone.Zone_3D = data.Zones[1]   
                #print(f"Fluid Zone: Nodes: {fluid_zone.Node_count}, Elements: {fluid_zone.Element_count}")

        
                '''
                if PRINT_DAT:
                    decoded_path = 'Decoded_Data/'
                    print('Outputting decoded .dat file')
                    if os.path.exists(decoded_path) == False:
                        os.mkdir(decoded_path)

                    
                    file_name = os.path.basename(filepath)
                    path_written_to = os.path.join(decoded_path, file_name)
                    #path_written_to = decoded_path + filepath +'/'


                    if os.path.exists(path_written_to):
                        print('WARNING, data corresponding to timestep ' + filepath + ' has been already generated. Shutting down...')
                        return
                    else:
                        os.mkdir(path_written_to)
                    for i in range(0,3): # 3 dimensions
                        print_array(fluid_zone.Element_Coordinates[i], fluid_zone.Element_count, path_written_to + 'Element_'+ data.Variables[i])
                        print_array(fluid_zone.Node_Coordinates[i], fluid_zone.Node_count, path_written_to + 'Node_'+ data.Variables[i])
                    for i in range(3, data.Var_count):
                        print_array(fluid_zone.Element_Variables[i - 3], fluid_zone.Element_count, path_written_to + 'Element_'+ data.Variables[i])
            
                    print("DONE!!!")
                    
                '''


                if PRINT_DAT:
                    decoded_path = 'Decoded_Data/'
                    print('Outputting decoded .dat file')
                    
                    # 确保根目录存在
                    if not os.path.exists(decoded_path):
                        os.mkdir(decoded_path)

                    # 获取文件名（不含路径），并去除后缀作为子文件夹名
                    file_name = os.path.basename(filepath)  # 例如 "400.dat"
                    folder_name = os.path.splitext(file_name)[0]  # 例如 "400"（去除 .dat 后缀）
                    
                    # 子文件夹完整路径（Decoded_Data/400/）
                    subfolder_path = os.path.join(decoded_path, folder_name)

                    # 检查子文件夹是否已存在（避免重复生成）
                    if os.path.exists(subfolder_path):
                        print(f'WARNING: 数据文件夹 {subfolder_path} 已存在，已终止操作')
                        return
                    else:
                        os.mkdir(subfolder_path)  # 创建子文件夹

                    # 写入元素坐标（3个维度）
                    for i in range(0, 3):
                        # 正确拼接子文件夹路径 + 文件名（例如 Decoded_Data/400/Element_X）
                        element_file = os.path.join(subfolder_path, f'Element_{data.Variables[i]}')
                        print_array(fluid_zone.Element_Coordinates[i], fluid_zone.Element_count, element_file)
                        
                        node_file = os.path.join(subfolder_path, f'Node_{data.Variables[i]}')
                        print_array(fluid_zone.Node_Coordinates[i], fluid_zone.Node_count, node_file)

                    # 写入其他变量
                    for i in range(3, data.Var_count):
                        var_file = os.path.join(subfolder_path, f'Element_{data.Variables[i]}')
                        print_array(fluid_zone.Element_Variables[i - 3], fluid_zone.Element_count, var_file)

                    print(f"DONE!!! 数据已保存至 {subfolder_path}")

            
        
    else:
        print(f"Warning: '{filepath}' is not a .dat file, ignoring...")
    

if __name__ == "__main__":
    main()
