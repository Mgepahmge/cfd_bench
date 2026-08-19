"""Load canonical legacy-CFD topology into TileDB."""

from __future__ import annotations

import argparse
import os
from typing import Mapping

import numpy as np

from cfd_bench.core.context import DatasetKey
from cfd_bench.ingest.cfd.canonical import load_cfd_topology
from cfd_bench.ingest.tiledb import io as tiledb_io


def export_topology_payload_to_tiledb(
    root: str, dataset_key: str, topo: Mapping[str, object], *, overwrite: bool = True
):
    zone = str(topo["zone_name"])
    meta = {"node_count": topo["node_count"], "cell_count": topo["cell_count"], "face_count": topo["face_count"], **topo["bbox"]}
    tiledb_io.write_mesh_meta(tiledb_io.mesh_static_uri(root, dataset_key, zone, "mesh_meta"), meta, overwrite=overwrite)
    nodes = topo["nodes"]
    tiledb_io.write_nodes(
        tiledb_io.mesh_static_uri(root, dataset_key, zone, "nodes"),
        nodes["x"], nodes["y"], nodes["z"], overwrite=overwrite,
    )
    cells = np.asarray(topo["cells"], dtype=np.float64)
    tiledb_io.write_cells(
        tiledb_io.mesh_static_uri(root, dataset_key, zone, "cells"),
        {
            "cx": cells[:, 0], "cy": cells[:, 1], "cz": cells[:, 2],
            "xmin": cells[:, 3], "xmax": cells[:, 4], "ymin": cells[:, 5], "ymax": cells[:, 6],
            "zmin": cells[:, 7], "zmax": cells[:, 8], "cell_type": cells[:, 9].astype(np.int32),
        },
        int(topo["cell_count"]), overwrite=overwrite,
    )
    tiledb_io.write_cell_nodes_dynamic(
        tiledb_io.mesh_static_uri(root, dataset_key, zone, "cell_nodes"),
        topo["cell_nodes"], int(topo["cell_count"]), overwrite=overwrite,
    )
    tiledb_io.write_cell_adjacency_dynamic(
        tiledb_io.mesh_static_uri(root, dataset_key, zone, "cell_adjacency"),
        topo["adjacency"], int(topo["cell_count"]), overwrite=overwrite,
    )
    bf = list(topo.get("boundary_faces", ()))
    if bf:
        arr = np.asarray(bf, dtype=np.float64)
        tiledb_io.write_boundary_faces(
            tiledb_io.mesh_static_uri(root, dataset_key, zone, "boundary_faces"),
            {
                "cell_id": arr[:, 0].astype(np.int32), "patch_code": arr[:, 1],
                "nx": arr[:, 2], "ny": arr[:, 3], "nz": arr[:, 4], "area": arr[:, 5],
                "cx": arr[:, 6], "cy": arr[:, 7], "cz": arr[:, 8],
            },
            len(arr), overwrite=overwrite,
        )
    print(
        f"TileDB topology: {dataset_key}/{zone} nodes={topo['node_count']} cells={topo['cell_count']} "
        f"max_nodes={topo['max_nodes_per_cell']} max_neighbors={topo['max_neighbors_per_cell']}"
    )


def load_topology(dat_path: str, ship: str, scale: str, root: str = "TileDB_Instances", zone_indices=None, overwrite=True, *, topology=None):
    key = DatasetKey(ship=ship, scale=scale)
    payloads = topology if topology is not None else load_cfd_topology(dat_path, list(zone_indices or [0, 1]))
    for topo in payloads.values():
        export_topology_payload_to_tiledb(root, key.dataset_key, topo, overwrite=overwrite)


def main():
    ap = argparse.ArgumentParser(description="Load CFD mesh topology into TileDB")
    ap.add_argument("--dat", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--root", default="TileDB_Instances")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()
    if not os.path.isfile(args.dat):
        raise FileNotFoundError(args.dat)
    load_topology(args.dat, args.ship_type, args.scale, args.root, args.zone_indices, args.overwrite)


if __name__ == "__main__":
    main()
