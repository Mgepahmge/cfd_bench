import re
from tkinter import OFF
from xml.sax.handler import DTDHandler
from xmlrpc.client import boolean
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
# Class Zone_3D:
# A data structure used for storing a zone for a 3D model
# The main components are as follows
# Elements: #Elements (#: Number of)
# Faces: #Faces
# Nodes: #Nodes
# ZoneType: Self-explainatory
# Parameters:
#   X, Y, Z: float[Nodes] arrays
#   U, V, W, P, K, E: float[Elements] arrays
"""

class Zone_3D:
    


    def generateMesh(self):
        
        return 

    def __init__(self, raw_content, var_count, dim, variables):
        self.Zone_name = ''
        self.Zone_type = ''
        self.Element_count = ''
        self.Face_count = ''
        self.Node_count = ''
        self.Variables = ''
        self.Node_Coordinates = []
        self.Element_Variables = []
        self.Element_Coordinates = []
        self.NCPF = []
        self.FN = []
        self.LE = []
        self.RE = []
        self.EN = []
        self.DIMENSION = -1
        self.EF = []
        self.WRITE_CONNECTIVITY = False

        self.DIMENSION = dim

        self.Variables = variables
        # Construting the line of DT=(...)
        splitter_DT = 'DT=('
        for i in range(0, var_count):
            splitter_DT += 'DOUBLE'
            if i < var_count - 1:
                splitter_DT += ' '
        splitter_DT += ')'
        
        sections = raw_content.split(splitter_DT)
        _vars = sections[1]
        
        # Extracting header
        _header = sections[0]
        # Extracting Zone name
        lines = _header.split('\n')
        self.Zone_name = lines[0].replace('"', '')
        # Extracting Node, Element, Face counts & Zonetype
        for line in lines:
            if line.strip().startswith('Nodes'):
                line = line.replace(' ', '')
                parts = line.split(',')
                for part in parts:
                    pairs = part.split('=')
                    if pairs[0] == 'Nodes':
                        self.Node_count = int(pairs[1])
                    elif pairs[0] == 'Elements':
                        self.Element_count = int(pairs[1])
                    elif pairs[0] == 'Faces':
                        self.Face_count = int(pairs[1])
                    else:
                        self.Zone_type = pairs[1]
                        
        # Extracting parameter values
        
        vals_components = _vars.split("#")
          
        DT = vals_components[0]
        
        node_count_per_face = []
        face_nodes = []
        left_elements = []
        right_elements = []

        for i in range(1, len(vals_components)):
            component = vals_components[i]
            if component.startswith(' node count per face'):
                node_count_per_face = component
            elif component.startswith(' face nodes'):
                face_nodes = component
            elif component.startswith(' left elements'):
                left_elements = component
            elif component.startswith(' right elements'):
                right_elements = component


        '''
        Decoding DT:
        '''
        # Processing DT
        print("Decoding DT:")
        DT_array = DT.replace("\n","").replace("   ","  ").replace("  "," ").strip().split(" ")
        
        N = self.Node_count
        E = self.Element_count
        
        if len(DT_array) != 3*N + (var_count - 3)*E:
            print("WARINING, The length of DT could be wrong.")

        N = self.Node_count
        E = self.Element_count
        F = self.Face_count
        
        visited_element = 0
        for i in tqdm(range(0, var_count)):
            if i < 3:
                tmp_var_txt = DT_array[visited_element: visited_element + N]
                visited_element += N
                tmp_var_double = np.zeros(N, dtype=np.float64)
                for i in range(0, N):
                    tmp_var_double[i] = np.float64(tmp_var_txt[i])
                self.Node_Coordinates.append(tmp_var_double)
            else:
                tmp_var_txt = DT_array[visited_element: visited_element + E]
                visited_element += E
                tmp_var_double = np.zeros(E, dtype=np.float64)
                for i in range(0, E):
                    tmp_var_double[i] = np.float64(tmp_var_txt[i])
                self.Element_Variables.append(tmp_var_double)
                

            
        '''
        Decoding node_count_per_face:
        '''
        print("Decoding NCPF:")
        if len(node_count_per_face) > 0:
            self.NCPF = np.array(self.decode_regular_part(node_count_per_face, self.Face_count, 0))
        else:
            # If node_count_per_face is not specified, then we are probably dealing with a polygon, e.g. 2-D mesh, where faces become lines and elements become faces. 
            # Therefore, the ncpf is bound to be 2, representing the two ends of a line
            self.NCPF = np.full(self.Face_count, 2)
        '''
        Decoding face_nodes:
        '''
        print("Decoding FN:")
        face_nodes_array = np.array(self.decode_regular_part(face_nodes, -1, -1))
        #self.FN = self.decode_face_nodes(face_nodes)
        self.FN = self.decode_face_node_array(face_nodes_array, self.Face_count)
        '''
        Decoding left_elements:
        '''
        print("Decoding LE:")
        self.LE = np.array(self.decode_regular_part(left_elements, self.Face_count, -1))
        
        '''
        Decoding right_elements:
        '''
        print("Decoding RE:")
        self.RE = np.array(self.decode_regular_part(right_elements, self.Face_count, -1))

        
        '''
        Constructing element_nodes:
        '''
        print("Constructing Element_Nodes")
        [self.EN, self.EF] = self.construct_element_face_and_nodes()

        '''
        Computing element centroids:
        '''
        self.decode_element_centroids_using_element_nodes()

        '''
        Computing element connectivity:
        '''
        if self.WRITE_CONNECTIVITY:
            self.EC = self.construct_element_adjacency()
            with open('connectivity.pkl', 'wb') as f:
                pickle.dump(self.EC, f)
                
        
        REQUIRES_CHECKING = False
        if REQUIRES_CHECKING:
            ''' Checking ...'''
            # # Create histogram
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            # Plot histograms
            ax1.hist(self.Element_Coordinates[0], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.hist(self.Element_Coordinates[1], bins=30, color='salmon', edgecolor='black', alpha=0.7)
            ax3.hist(self.Element_Coordinates[2], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)

            # Customize each subplot
            ax1.set_title('Normal Distribution', fontsize=14)
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, linestyle='--', alpha=0.6)

            ax2.set_title('Exponential Distribution', fontsize=14)
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, linestyle='--', alpha=0.6)

            ax3.set_title('Uniform Distribution', fontsize=14)
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, linestyle='--', alpha=0.6)

            # Add overall title and adjust layout
            plt.suptitle('Distribution Comparison of Three Arrays', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show(block = True)
        

            # Checking EN & EF
            good_match_count = 0
            bad_match_count = 0
            for e in tqdm(range(0,self.Element_count)):

                nodes_by_EN = set(self.EN[e])
            
                nodes_by_EF = set()
                for f in self.EF[e]:
                    for n in self.FN[f]:
                        nodes_by_EF.add(n)
                if nodes_by_EF == nodes_by_EN:
                    # print(f"GOOD: EN and EF of Element {e} matches...")
                    good_match_count += 1
                else:
                    print(f"ERROR: EN and EF of Element {e} does not match...")
                    bad_match_count += 1
            
            print(f"\nGood match count = {good_match_count}.")
            print(f"\nBad match count = {bad_match_count}.")
            

    def decode_regular_part(self, array_in_text, N, offset):
        lines = array_in_text.split("\n")[1:]
        values_in_txt = []
        for line in tqdm(lines):
            if len(line) == 0:
                continue
            tokens = line.strip().replace("   ","  ").replace("  "," ").split(" ")
            for token in tokens:
                values_in_txt.append(token)
        if N != -1:
            if len(values_in_txt) != N:
                print("Error, decoding regular array wrong.")
        
        result = np.zeros(len(values_in_txt), dtype = np.int64)
        for i in range(0, len(values_in_txt)):
            result[i] = np.int64(values_in_txt[i]) + offset
            
        return result
    
    def decode_face_nodes(self, array_in_text):
        lines = array_in_text.split("\n")[1:]
        result = []
        for line in tqdm(lines):
            if len(line) == 0:
                continue
            processed_line = []
            elements = line.strip().split(' ')
            for element in elements:
                processed_line.append(np.int64(element))
            result.append(processed_line)
            
        return result
    
    def decode_face_node_array(self, face_node_array, N):
        # N: face count
        result = []
        offset = 0
        for i in tqdm(range(0, N)):
            tmp_face_node_count = self.NCPF[i]
            result.append(face_node_array[offset: offset + tmp_face_node_count])
            offset += tmp_face_node_count
            
        return result
    
    def to_dataframe(self):
        node_indexes = [i for i in range(0, self.Node_count)]
        
        df_node_coordinates:pd.DataFrame = pd.DataFrame(index = node_indexes, columns = [])
        for i in range(0, len(self.DIMENSION)): 
            tmp_series = pd.Series(self.Node_Coordinates[i], index = node_indexes, dtype = np.float64, name = self.DIMENSION[i])
            df_node_coordinates = pd.concat([df_node_coordinates, tmp_series], axis = 1)
            
        element_indexes = [i for i in range(0, self.Element_count)]
        df_element_coordinates:pd.DataFrame = pd.DataFrame(index = element_indexes, columns = [])
        for i in range(0, len(self.DIMENSIONS)): 
            tmp_series = pd.Series(self.Element_Coordinates[i], index = element_indexes, dtype = np.float64, name = self.DIMENSIONS[i])
            df_element_coordinates = pd.concat([df_element_coordinates, tmp_series], axis = 1)
            
        df_element_variables:pd.DataFrame = pd.DataFrame(index = element_indexes, columns = [])
        
        
        df_element_variables:pd.DataFrame = pd.DataFrame(index = element_indexes, columns = [])
        variables = self.Element_Variables
        for i in range(0, len(variables)):
            tmp_series = pd.Series(self.Element_Variables[i], index = element_indexes, dtype = np.float64, name = variables[i])
            df_element_variables = pd.concat([df_element_variables, tmp_series], axis = 1)
        
        
        return [df_node_coordinates, df_element_coordinates, df_element_variables]
        
    def to_postgresql_dataframe(self, ship_type, scale, timestep, zone_type=None):
        """
        生成用于PostgreSQL插入的DataFrame，保持元素索引顺序和精度
        """       

        final_zone_type = zone_type if zone_type is not None else self.Zone_type

        # 确保元素索引顺序正确
        element_indices = sorted(self.EN.keys())
        
        data_records = []
        
        dimensions = ['X', 'Y', 'Z']

        # 处理节点坐标数据 (X, Y, Z) - is_element=False
        for dim_idx, dim_name in enumerate(dimensions):
            if dim_idx < 3:  # X, Y, Z
                # 节点数据
                node_data = {
                    'ship_type': ship_type,
                    'scale': scale,
                    'zone_type': final_zone_type,
                    'timestep': timestep,
                    'variable': dim_name,
                    'is_element': False,
                    'data': self.Node_Coordinates[dim_idx].tolist(),
                    'max_range': None
                }
                data_records.append(node_data)
        
        # 处理单元坐标数据 (X, Y, Z) - is_element=True
        for dim_idx, dim_name in enumerate(dimensions):
            if dim_idx < 3:  # X, Y, Z
                # 确保元素坐标数组按照正确的元素索引顺序
                element_coords = []
                for elem_idx in element_indices:
                    if dim_idx < len(self.Element_Coordinates):
                        element_coords.append(float(self.Element_Coordinates[dim_idx][elem_idx]))
                    else:
                        element_coords.append(0.0)
                
                element_data = {
                    'ship_type': ship_type,
                    'scale': scale,
                    'zone_type': self.Zone_type,
                    'timestep': timestep,
                    'variable': dim_name,
                    'is_element': True,
                    'data': element_coords,
                    'max_range': None
                }
                data_records.append(element_data)
        
        # 处理其他变量 (U, V, W, P, K, E) - is_element=True
        # 确保变量顺序与Variables列表匹配
        for var_idx, var_name in enumerate(self.Variables[3:], start=3):  # 从第4个变量开始
            if var_idx - 3 < len(self.Element_Variables):
                # 确保按照正确的元素索引顺序
                var_values = []
                for elem_idx in element_indices:
                    if elem_idx < len(self.Element_Variables[var_idx - 3]):
                        # 使用高精度转换
                        val = self.Element_Variables[var_idx - 3][elem_idx]
                        if isinstance(val, np.float32):
                            var_values.append(float(val))
                        else:
                            var_values.append(val)
                    else:
                        var_values.append(0.0)
                
                variable_data = {
                    'ship_type': ship_type,
                    'scale': scale,
                    'zone_type': self.Zone_type,
                    'timestep': timestep,
                    'variable': var_name,
                    'is_element': True,
                    'data': var_values,
                    'max_range': None
                }
                data_records.append(variable_data)
        
        # 创建DataFrame并确保顺序
        df = pd.DataFrame(data_records)
        
        # 排序以确保一致性
        df = df.sort_values(['variable', 'is_element', 'zone_type'])
        
        return df

    # def construct_element_nodes(self):
    #     element_nodes_dict = defaultdict(set)
        
    #     for f in tqdm(range(0, self.Face_count)):
    #         face_nodes = self.FN[f]
    #         # Processing left elements
    #         tmp_element_id = self.LE[f]
            
    #         if tmp_element_id == -1:
    #             # element '-1' represents the boundary element, omit it during computation
    #             continue
            

    #         if tmp_element_id in element_nodes_dict:
    #             tmp_element_nodes = element_nodes_dict[tmp_element_id]
    #         else:
    #             tmp_element_nodes = set()
                
    #         for p in face_nodes:
    #             tmp_element_nodes.add(p)
    #         element_nodes_dict[tmp_element_id] = tmp_element_nodes
                

    #         # Processing right elements
    #         tmp_element_id = self.RE[f]

    #         if tmp_element_id == -1:
    #             # element '-1' represents the boundary element, omit it during computation
    #             continue
   
    #         if tmp_element_id in element_nodes_dict:
    #             tmp_element_nodes = element_nodes_dict[tmp_element_id]
    #         else:
    #             tmp_element_nodes = set()
                
    #         for p in face_nodes:
    #             tmp_element_nodes.add(p)
    #         element_nodes_dict[tmp_element_id] = tmp_element_nodes
            

    def construct_element_face_and_nodes(self):
        element_nodes_dict = defaultdict(set)
        element_faces_dict = defaultdict(set)
        
        for f in tqdm(range(0, self.Face_count)):
            face_nodes = self.FN[f]
            # Processing left elements
            tmp_element_id = self.LE[f]
            
            # Filtering unwanted element ids
            if tmp_element_id == -1:
                # element '-1' represents the boundary element, omit it during computation
                continue
            
            # Checking if element_id has been processed
            if tmp_element_id in element_nodes_dict:
                tmp_element_nodes = element_nodes_dict[tmp_element_id]
                tmp_element_faces = element_faces_dict[tmp_element_id]
            else:
                tmp_element_nodes = set()
                tmp_element_faces = set()
                
            # Adding phase
            tmp_element_faces.add(f)
            for p in face_nodes:
                tmp_element_nodes.add(p)
                
            element_faces_dict[tmp_element_id] = tmp_element_faces
            element_nodes_dict[tmp_element_id] = tmp_element_nodes
                

            # Processing right elements
            tmp_element_id = self.RE[f]

            # Checking if element_id has been processed
            if tmp_element_id in element_nodes_dict:
                tmp_element_nodes = element_nodes_dict[tmp_element_id]
                tmp_element_faces = element_faces_dict[tmp_element_id]
            else:
                tmp_element_nodes = set()
                tmp_element_faces = set()
                
            # Adding phase
            tmp_element_faces.add(f)
            for p in face_nodes:
                tmp_element_nodes.add(p)
                
            element_faces_dict[tmp_element_id] = tmp_element_faces
            element_nodes_dict[tmp_element_id] = tmp_element_nodes
            
        return [element_nodes_dict, element_faces_dict]


    def decode_element_centroids_using_element_nodes(self):
        
        X = self.Node_Coordinates[0]
        Y = self.Node_Coordinates[1]
        Z = self.Node_Coordinates[2]
        
        print("Loading X, Y, Z:")
        Element_X = np.zeros(self.Element_count)
        Element_Y = np.zeros(self.Element_count)
        Element_Z = np.zeros(self.Element_count)
        for i in tqdm(range(0,self.Element_count)):
            tmp_element_nodes = self.EN[i]
            centroid = np.zeros(3)
            for n in tmp_element_nodes:
                centroid[0] += X[n]
                centroid[1] += Y[n]
                centroid[2] += Z[n]
                
            centroid = centroid/8
            # for j in [0,1,2]:
            #     centroid[j] = centroid[j]/len(tmp_element_nodes)
            Element_X[i] = centroid[0]
            Element_Y[i] = centroid[1]
            Element_Z[i] = centroid[2]
        
        self.Element_Coordinates.append(Element_X)
        self.Element_Coordinates.append(Element_Y)
        self.Element_Coordinates.append(Element_Z)
        

    def construct_element_adjacency(self):
        dict_connectivity = {}
        print("Build connectivity dictionary:")
        for i in tqdm(range(0, self.Face_count)):
            tmp_LE = self.LE[i]
            tmp_RE = self.RE[i]
            if tmp_LE in dict_connectivity:
                dict_connectivity[tmp_LE].append(tmp_RE)
            else:
                dict_connectivity[tmp_LE] = [tmp_RE]
                
            if tmp_RE in dict_connectivity:
                dict_connectivity[tmp_RE].append(tmp_LE)
            else:
                dict_connectivity[tmp_RE] = [tmp_LE]
         
        connectivity_list = []
        print("Sorting connectivity list:")
        for i in tqdm(range(0, self.Element_count)):
            connectivity_list.append(dict_connectivity[i])
            
        return connectivity_list
        

    # def decode_element_faces(self):
    #     print("Loading mesh cells:")
    #     # # The first step is to find the faces of each cell using LE (left element) and RE (right element)
    #     # min_LE = np.min(self.LE)
    #     # max_LE = np.max(self.LE)
    #     # min_RE = np.min(self.RE)
    #     # max_RE = np.max(self.RE)
    #     # if (max_RE - min_RE) != (max_LE - min_LE):
    #     #     print("ERROR, Left Element & Right Element do not match.")
    #     #     exit()
            
    #     # # WARNING: For unknown reasons some of the LE/RE values starts with 1 instead of 0. 
    #     # # Therefore, we need to check LE/RE values and adjust to 0-initiated arrays.
    #     # unified_LE = np.subtract(self.LE, min_LE)
    #     # unified_RE = np.subtract(self.RE, min_RE)

    #     element_faces = defaultdict(set)
        
    #     print("Processing left elements:")
    #     for i in tqdm(range(0, len(self.LE))):
    #         e = self.LE[i]
    #         # element '0' represents the boundary element, omit it during computation
    #         if e == -1: 
    #             continue
            
    #         # The effective numbering of elements starts with 0
    #         if e in element_faces:
    #             # since e is a left element of face i, i naturally becomes a face of e
    #             element_faces[e].add(i)
    #         else:
    #             # if i is not in the dictionary, create a new face
    #             element_faces[e] = {i}
                
    #     # The same process repeats for the right elements
    #     print("Processing right elements:")
    #     for i in tqdm(range(0, len(self.RE))):
    #         e = self.RE[i]
    #         # element '0' represents the boundary element, omit it during computation
    #         if e == -1: 
    #             continue
    #         # The effective numbering of elements starts with 0
    #         if e in element_faces:
    #             #since e is a right element of face i, i naturally becomes a face of e
    #             element_faces[e].add(i)
    #         else:
    #             element_faces[e] = {i}
        
    #     self.EF = element_faces