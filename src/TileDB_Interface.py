import os
import time
import numpy as np
from numpy.typing import NDArray
import tiledb
import Dat_Data_Decoder as Decoder
import Zone 
import numpy as np
from pathlib import Path

class TileDB_Interface:

    # These parameters should be initialized afterwards
    dimension = None
    epsilon = 1e-3 # A constant for various purposes
    tiledb_context = None

    def __init__(self, DIMENSIONALITY = 3):

        self.dimension = DIMENSIONALITY
        self.tiledb_context = tiledb.Ctx()

        return

    '''
    Desription:
        Load_dat_data() access a folder full of .dat files, and converts each dat file into a tiledb entity, write it to TileDB_path.

    Input:
        dat_path: The folder that holds the .dat files.
        Tiledb_path: The path to where the tiledb entities are written.

    Output:
        none
    '''

    def Load_dat_data(self, dat_path, SHIP_TYPE, Tiledb_path = "TileDB_Instances/"):
        os.makedirs(Tiledb_path, exist_ok=True)
        for file in os.listdir(dat_path):
            filepath = os.path.join(dat_path, file)
            if os.path.isfile(filepath):  # Process Files only

                # Check if the filename ends with .dat
                if filepath.lower().endswith('.dat'):
                    dataset_name = Path(filepath).stem

                    print(f"Processing file: {filepath}")
                    data = Decoder.CAE_Decoder(3)
                    data.Decode_dat_file(dat_path + '/' + file)
                    # Thus far, only the fluid zone is our interest
                    fluid_zone:Zone.Zone_3D = data.Zones[0]
                    hull_zone:Zone.Zone_3D = data.Zones[1]

                    # The domain defines the coordinate space and data types for dimensions.
                    dom_fluid = self.initialize_domain_for_zone(fluid_zone.Element_Coordinates)
                    dom_hull = self.initialize_domain_for_zone(hull_zone.Element_Coordinates)

                    # Define the attributes. In our case, attributes_fluid and attributes_hull should be identical
                    attributes_fluid = self.initialize_attributes_for_zones(fluid_zone.Variables)
                    attributes_hull = self.initialize_attributes_for_zones(hull_zone.Variables)
                    
                    # --- Declare tiledb schemas --- #
                    schema_fluid = tiledb.ArraySchema(
                        domain = dom_fluid,
                        attrs = attributes_fluid,
                        sparse = True,  
                        capacity = fluid_zone.Element_count,
                        cell_order = "row-major",
                        tile_order = "row-major",
                        ctx = self.tiledb_context
                    )

                    tiledb_dataset_name_fluid = Tiledb_path + SHIP_TYPE + '_' + dataset_name + "fluid.tdb"
                    tiledb.SparseArray.create(tiledb_dataset_name_fluid, schema_fluid)
                    
                    # tiledb.SparseArray.create('TileDB_Instances\\1000fluid.tdb', schema_fluid)
                    schema_hull = tiledb.ArraySchema(
                        domain = dom_hull,
                        attrs = attributes_hull,
                        sparse = True,  
                        capacity = hull_zone.Element_count,
                        cell_order = "row-major",
                        tile_order = "row-major",
                        ctx = self.tiledb_context
                    )

                    tiledb_dataset_name_hull = Tiledb_path + SHIP_TYPE + '_' + dataset_name + "hull.tdb"
                    tiledb.SparseArray.create(tiledb_dataset_name_hull, schema_hull)

                    # --- Inject data --- #

                    # create write stream (basically just wrap data into dictionary)
                    write_stream_fluid = self.initialize_attribute_write_stream(fluid_zone)
                    write_stream_hull = self.initialize_attribute_write_stream(hull_zone)

                    # open the array and write data
                    with tiledb.SparseArray(tiledb_dataset_name_fluid, mode = 'w', ctx = self.tiledb_context) as tiledb_fluid:
                        # Data is written by providing coordinates and a dictionary of attributes.
                        tiledb_fluid[fluid_zone.Element_Coordinates[0], fluid_zone.Element_Coordinates[1], fluid_zone.Element_Coordinates[2]] = write_stream_fluid

                    with tiledb.SparseArray(tiledb_dataset_name_hull, mode = 'w', ctx = self.tiledb_context) as tiledb_hull:
                        # Data is written by providing coordinates and a dictionary of attributes.
                        tiledb_hull[hull_zone.Element_Coordinates[0], hull_zone.Element_Coordinates[1], hull_zone.Element_Coordinates[2]] = write_stream_hull

                    

        return

    '''
    Desription:
        Load_TileDB_File() loads a tiledb file/folder into a db entity. The subsequent queries are executed depending on this entity

    Input:
        tiledb_path: The path to the tiledb file/folder.

    Output:
        db_entity: A tiledb database entity.
    '''

    def Load_TileDB_File(self, tiledb_path):
        db_entity = tiledb.open(tiledb_path, mode='r', ctx = self.tiledb_context)
        return db_entity


    '''
    Desription:
        Spatial_Range_Query_TileDB() takes as input spatial query conditions, and output indexes of points that satisfies the conditions
        Spatial_Range_Query_TileDB() Performs range query specifically on coordinates, since tiledb accelerates coordinate-dependent queries.

    Input:
        db_entity: A tile-db entity that the query is executed on.
        x_lower, x_upper, y_lower, y_upper, z_lower, z_upper: The lower/upper bounds on the x,y,z dimensions.
        print_result: Whether the user what to print the query result.

    Output:
        result_indexes: Indexes of points that satisfies the conditions.
    '''
    def Spatial_Range_Query_TileDB(self, db_entity, lower_bound:NDArray[np.float64], upper_bound:NDArray[np.float64], print_result = False):
        x_lower, y_lower, z_lower = lower_bound
        x_upper, y_upper, z_upper = upper_bound

        # 防止 VTK 的边界比 TileDB Schema 定义的边界大，导致 "out of domain bounds" 错误
        domain = db_entity.schema.domain
        
        # 获取 X, Y, Z 维度的定义域 (假设维度名称是 "X", "Y", "Z")
        dx_min, dx_max = domain.dim("X").domain
        dy_min, dy_max = domain.dim("Y").domain
        dz_min, dz_max = domain.dim("Z").domain

        # 裁剪 (Clamp) 操作：如果查询下界小于 Domain下界，取 Domain下界；如果上界大于 Domain上界，取 Domain上界
        x_lower = max(x_lower, dx_min); x_upper = min(x_upper, dx_max)
        y_lower = max(y_lower, dy_min); y_upper = min(y_upper, dy_max)
        z_lower = max(z_lower, dz_min); z_upper = min(z_upper, dz_max)

        # 如果任何一个维度的下界 > 上界，说明查询范围完全在 Domain 之外，交集为空
        if x_lower > x_upper or y_lower > y_upper or z_lower > z_upper:
            if print_result:
                print("Query range is out of domain bounds, returning empty.")
            return np.array([], dtype=np.int32)

        temp_result = db_entity.multi_index[x_lower: x_upper, y_lower: y_upper, z_lower: z_upper]

        result_indexes = temp_result['Index']
        if print_result:
            print(result_indexes)

        return np.array(result_indexes, dtype=np.int32)


    '''
    Desription:
        Attribute_Range_Query_TileDB() takes as input range query conditions on an attribute, and output indexes of points that satisfies the conditions.
        This query performs range query on non-spatial attributes. Queries like this will degrade to simple sequential scan.

    Input:
        db_entity: A tile-db entity that the query is executed on.
        attribute_name: The name of the attribute to be queried on.
        min_val, max_val: the lower/upper bound of the query range.
        print_result: Whether the user what to print the query result.

    Output:
        result_indexes: Indexes of points that satisfies the conditions.
    '''
    def Attribute_Range_Query_TileDB(self, db_entity, attribute_name:str, lower:float, upper:float, print_result = False):

        result = []

        dataset = db_entity[:]

        index_arr = dataset['Index']
        attribute_arr = dataset[attribute_name]

        for i in range(0, len(attribute_arr)):
            if attribute_arr[i]>=lower and attribute_arr[i] <= upper:
                result.append(index_arr[i])
            
        result = np.asarray(result)
        if print_result == True:
            print(result)

        return np.array(result, dtype=np.int32)


    '''
    Desription:
        Point_Query_Attribute_TileDB() takes as input a list of point indexes, and returns the attributes of the points.

    Input:
        db_entity: A tile-db entity that the query is executed on.
        attribute_names: A list of attribute names to be fetched
        indexes: the A list of point indexes
        print_result: Whether the user what to print the query result.

    Output:
        result: A list of sub-lists. Each sublist corresponds to a single attribute.
        E.g., If the attribute names = ['P','K'], then the result would be like [[p1,p2,p3,...,pn],[k1,k2,k3,...,kn]]
    '''
    def Point_Query_Attribute_TileDB(self, db_entity, attribute_name:str, indexes:np.array, print_result = False):
        result = []

        dataset = db_entity[:]

        tmp_attribute_array = dataset[attribute_name]
        for index in indexes:
                result.append(tmp_attribute_array[index])
            
        if print_result == True:
            print(result)

        return np.array(result, dtype=np.float64)




    ''' Private functions (You won't call these functions directly) '''

    def initialize_domain_for_zone(self, coordinates: np.array):
        dom = tiledb.Domain(
            tiledb.Dim(name="X", domain=(np.min(coordinates[0]) - self.epsilon, np.max(coordinates[0]) + self.epsilon), tile=0.05, dtype=np.float32, ctx=self.tiledb_context),
            tiledb.Dim(name="Y", domain=(np.min(coordinates[1]) - self.epsilon, np.max(coordinates[1]) + self.epsilon), tile=0.05, dtype=np.float32, ctx=self.tiledb_context),
            tiledb.Dim(name="Z", domain=(np.min(coordinates[2]) - self.epsilon, np.max(coordinates[2]) + self.epsilon), tile=0.05, dtype=np.float32, ctx=self.tiledb_context),
            ctx = self.tiledb_context
        ) 
        return dom

    def initialize_attributes_for_zones(self, Variables):
        attributes = []
        # Index is a default attribute that everybody needs
        attributes.append(tiledb.Attr(name = 'Index', dtype = np.int32, ctx = self.tiledb_context))
        for i in range(3, len(Variables)):
            attributes.append(tiledb.Attr(name=Variables[i], dtype = np.float32, ctx = self.tiledb_context))

        return attributes

    def initialize_attribute_write_stream(self, zone:Zone.Zone_3D):
        # initialization
        tiledb_write_stream = {}
        tiledb_write_stream['Index'] = np.arange(0, zone.Element_count)
        
        for i in range(3, len(zone.Variables)):
            attribute = zone.Variables[i]
            source_data = zone.Element_Variables[i - 3]
            tiledb_write_stream[attribute] = source_data
            print(f"Added {attribute} to write batch")

        return tiledb_write_stream

    
''' main script '''
def main():
    
    TDB = TileDB_Interface()

    # TDB.Load_dat_data("/home/lzhang/data/gpudb/downloaded/JBC_615k/Postprocessing", "JBC_615k", "TileDB_Instances/")

    tdb_entity = TDB.Load_TileDB_File("TileDB_Instances/JBC_615k_1000fluid.tdb")
    
    TileDB_Interface.Point_Query_Attribute_TileDB(tdb_entity, "P" , [1,10,100,1000], True)

    TileDB_Interface.Spatial_Range_Query_TileDB(tdb_entity, [0,0,0],[20,5,7],True)

    TileDB_Interface.Attribute_Range_Query_TileDB(tdb_entity, 'P', 0.95, 1.0, True)





if __name__ == "__main__":
    main()