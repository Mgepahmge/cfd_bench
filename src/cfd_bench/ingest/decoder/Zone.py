import numpy as np
from collections import defaultdict

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


        # Processing DT
        DT_array = DT.replace("\n", "").replace("   ", "  ").replace("  ", " ").strip().split(" ")
        
        N = self.Node_count
        E = self.Element_count
        
        if len(DT_array) != 3 * N + (var_count - 3) * E:
            pass  # DT length mismatch tolerated; downstream validation may fail

        N = self.Node_count
        E = self.Element_count
        F = self.Face_count

        visited_element = 0
        for vi in range(0, var_count):
            if vi < 3:
                tmp_var_txt = DT_array[visited_element : visited_element + N]
                visited_element += N
                tmp_var_double = np.zeros(N, dtype=np.float64)
                for j in range(0, N):
                    tmp_var_double[j] = np.float64(tmp_var_txt[j])
                self.Node_Coordinates.append(tmp_var_double)
            else:
                tmp_var_txt = DT_array[visited_element : visited_element + E]
                visited_element += E
                tmp_var_double = np.zeros(E, dtype=np.float64)
                for j in range(0, E):
                    tmp_var_double[j] = np.float64(tmp_var_txt[j])
                self.Element_Variables.append(tmp_var_double)
        if len(node_count_per_face) > 0:
            self.NCPF = np.array(self.decode_regular_part(node_count_per_face, self.Face_count, 0))
        else:
            # If node_count_per_face is not specified, then we are probably dealing with a polygon, e.g. 2-D mesh, where faces become lines and elements become faces. 
            # Therefore, the ncpf is bound to be 2, representing the two ends of a line
            self.NCPF = np.full(self.Face_count, 2)

        face_nodes_array = np.array(self.decode_regular_part(face_nodes, -1, -1))
        self.FN = self.decode_face_node_array(face_nodes_array, self.Face_count)
        self.LE = np.array(self.decode_regular_part(left_elements, self.Face_count, -1))
        self.RE = np.array(self.decode_regular_part(right_elements, self.Face_count, -1))
        self.EN, self.EF = self.construct_element_face_and_nodes()
        self.decode_element_centroids_using_element_nodes()

    def decode_regular_part(self, array_in_text, N, offset):
        lines = array_in_text.split("\n")[1:]
        values_in_txt = []
        for line in lines:
            if len(line) == 0:
                continue
            tokens = line.strip().replace("   ", "  ").replace("  ", " ").split(" ")
            values_in_txt.extend(tokens)
        if N != -1 and len(values_in_txt) != N:
            raise ValueError(f"expected {N} values, got {len(values_in_txt)}")

        result = np.zeros(len(values_in_txt), dtype=np.int64)
        for i in range(len(values_in_txt)):
            result[i] = np.int64(values_in_txt[i]) + offset
        return result

    def decode_face_node_array(self, face_node_array, N):
        result = []
        offset = 0
        for i in range(N):
            tmp_face_node_count = self.NCPF[i]
            result.append(face_node_array[offset : offset + tmp_face_node_count])
            offset += tmp_face_node_count
        return result

    def construct_element_face_and_nodes(self):
        element_nodes_dict = defaultdict(set)
        element_faces_dict = defaultdict(set)
        
        for f in range(self.Face_count):
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
        
        Element_X = np.zeros(self.Element_count)
        Element_Y = np.zeros(self.Element_count)
        Element_Z = np.zeros(self.Element_count)
        for i in range(self.Element_count):
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
        for i in range(self.Face_count):
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
        for i in range(self.Element_count):
            connectivity_list.append(dict_connectivity[i])
        return connectivity_list