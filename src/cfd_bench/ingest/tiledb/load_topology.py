"""Load mesh topology from DAT into TileDB mesh_static domain."""

from __future__ import annotations

import argparse
import os

import numpy as np

from cfd_bench.core.context import DatasetKey
from cfd_bench.ingest.common.topology_export import export_zone_topology
from cfd_bench.ingest.decoder import CAE_Decoder
from cfd_bench.ingest.tiledb import io as tiledb_io


def export_zone_to_tiledb(root: str, dataset_key: str, zone_name: str, topo: dict, overwrite: bool = True):
    zone_key = zone_name.strip().replace(" ", "_") or "Zone_0"
    meta = {
        "node_count": topo["node_count"],
        "cell_count": topo["cell_count"],
        "face_count": topo["face_count"],
        **topo["bbox"],
    }
    tiledb_io.write_mesh_meta(
        tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "mesh_meta"), meta, overwrite=overwrite
    )
    tiledb_io.write_nodes(
        tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "nodes"),
        topo["nodes"]["x"], topo["nodes"]["y"], topo["nodes"]["z"], overwrite=overwrite,
    )
    cell_count = topo["cell_count"]
    cells = np.array(topo["cells"], dtype=np.float64)
    tiledb_io.write_cells(
        tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "cells"),
        {
            "cx": cells[:, 0], "cy": cells[:, 1], "cz": cells[:, 2],
            "xmin": cells[:, 3], "xmax": cells[:, 4], "ymin": cells[:, 5], "ymax": cells[:, 6],
            "zmin": cells[:, 7], "zmax": cells[:, 8], "cell_type": cells[:, 9],
        },
        cell_count, overwrite=overwrite,
    )
    cn = np.array(topo["cell_nodes"], dtype=np.float32)
    tiledb_io.write_cell_nodes(
        tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "cell_nodes"), cn, cell_count, overwrite=overwrite
    )
    adj = np.array(topo["adjacency"], dtype=np.float32)
    tiledb_io.write_cell_adjacency(
        tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "cell_adjacency"), adj, cell_count, overwrite=overwrite
    )
    if topo["face_planes"]:
        fp = np.array(topo["face_planes"])
        tiledb_io.write_face_planes(
            tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "face_planes"),
            {
                "cell_id": fp[:, 0].astype(np.int32), "neighbor_id": fp[:, 1].astype(np.int32),
                "nx": fp[:, 2], "ny": fp[:, 3], "nz": fp[:, 4], "d": fp[:, 5],
                "face_area": fp[:, 6], "face_cx": fp[:, 7], "face_cy": fp[:, 8], "face_cz": fp[:, 9],
            },
            len(fp), overwrite=overwrite,
        )
    if topo["boundary_faces"]:
        bf = np.array(topo["boundary_faces"])
        tiledb_io.write_boundary_faces(
            tiledb_io.mesh_static_uri(root, dataset_key, zone_key, "boundary_faces"),
            {
                "cell_id": bf[:, 0].astype(np.int32), "patch_code": bf[:, 1],
                "nx": bf[:, 2], "ny": bf[:, 3], "nz": bf[:, 4], "area": bf[:, 5],
                "cx": bf[:, 6], "cy": bf[:, 7], "cz": bf[:, 8],
            },
            len(bf), overwrite=overwrite,
        )


def load_topology(dat_path: str, ship: str, scale: str, root: str = "TileDB_Instances", zone_indices=None, overwrite=True):
    key = DatasetKey(ship=ship, scale=scale)
    zone_indices = zone_indices or [0]
    data = CAE_Decoder(3)
    data.Decode_dat_file(dat_path)
    for zi in zone_indices:
        if zi >= len(data.Zones):
            continue
        zone = data.Zones[zi]
        topo = export_zone_topology(zone)
        export_zone_to_tiledb(root, key.dataset_key, zone.Zone_name, topo, overwrite=overwrite)


def main():
    ap = argparse.ArgumentParser(description="Load mesh topology from DAT into TileDB mesh_static")
    ap.add_argument("--dat", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--root", default="TileDB_Instances")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0])
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()
    if not os.path.isfile(args.dat):
        raise FileNotFoundError(args.dat)
    load_topology(args.dat, args.ship_type, args.scale, args.root, args.zone_indices, args.overwrite)


if __name__ == "__main__":
    main()
