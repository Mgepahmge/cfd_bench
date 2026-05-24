import os
import time
import numpy as np
from numpy.typing import NDArray
import tiledb
from . import Dat_Data_Decoder as Decoder
from . import Zone 
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
                    data.Decode_dat_file(filepath)
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

    project_root = Path(__file__).resolve().parents[4]

    input_path = project_root / "data" / "Kvlcc2_351k" / "Postprocessing"
    output_path = project_root / "TileDB_Instances"

    TDB.Load_dat_data(input_path, "Kvlcc2_351k", output_path)

    





if __name__ == "__main__":
    main()