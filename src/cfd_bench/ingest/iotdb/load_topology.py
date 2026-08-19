"""Load canonical legacy-CFD topology into IoTDB."""

from __future__ import annotations

import argparse
import os
from typing import Mapping, Sequence

import numpy as np
from iotdb.Session import Session
from iotdb.utils.Field import TSDataType as T

from cfd_bench.core.context import DatasetKey
from cfd_bench.core.paths import iotdb_root
from cfd_bench.ingest.cfd.canonical import load_cfd_topology
from cfd_bench.ingest.iotdb.io import (
    delete_timeseries_prefix,
    insert_numpy_columns,
    insert_ragged_int_rows,
    insert_tablet_chunked,
)


def export_topology_payload_to_iotdb(session: Session, dataset_key: str, topo: Mapping[str, object]):
    zone = str(topo["zone_name"])
    base = f"{iotdb_root()}.mesh_static.{dataset_key}.{zone}"
    delete_timeseries_prefix(session, base)

    node_count = int(topo["node_count"])
    cell_count = int(topo["cell_count"])
    max_nodes = int(max(1, topo.get("max_nodes_per_cell", 1)))
    max_neighbors = int(max(1, topo.get("max_neighbors_per_cell", 1)))

    nodes = topo["nodes"]
    insert_numpy_columns(
        f"{base}.nodes",
        session,
        np.arange(node_count, dtype=np.int64),
        {"x": nodes["x"], "y": nodes["y"], "z": nodes["z"]},
        [T.DOUBLE, T.DOUBLE, T.DOUBLE],
    )

    cell_rows = topo["cells"]
    insert_tablet_chunked(
        f"{base}.cells",
        session,
        list(range(cell_count)),
        ["cx", "cy", "cz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cell_type"],
        [T.DOUBLE] * 9 + [T.INT32],
        [
            [float(x) for x in row[:9]] + [int(row[9])]
            for row in cell_rows
        ],
    )

    insert_ragged_int_rows(
        f"{base}.cell_nodes", session, list(range(cell_count)), topo["cell_nodes"], "node_id", max_nodes
    )
    insert_ragged_int_rows(
        f"{base}.cell_adjacency", session, list(range(cell_count)), topo["adjacency"], "neighbor_id", max_neighbors
    )

    bf = list(topo.get("boundary_faces", ()))
    if bf:
        insert_tablet_chunked(
            f"{base}.boundary_faces",
            session,
            list(range(len(bf))),
            ["cell_id", "patch_code", "nx", "ny", "nz", "area", "cx", "cy", "cz"],
            [T.INT64, T.DOUBLE, T.DOUBLE, T.DOUBLE, T.DOUBLE, T.DOUBLE, T.DOUBLE, T.DOUBLE, T.DOUBLE],
            [
                [int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7]), float(r[8])]
                for r in bf
            ],
        )

    meta = topo["bbox"]
    insert_tablet_chunked(
        f"{base}.mesh_meta",
        session,
        [0],
        [
            "node_count", "cell_count", "face_count",
            "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y", "bbox_min_z", "bbox_max_z",
            "max_nodes_per_cell", "max_neighbors_per_cell",
        ],
        [T.INT64, T.INT64, T.INT64] + [T.DOUBLE] * 6 + [T.INT32, T.INT32],
        [[
            node_count, cell_count, int(topo["face_count"]),
            float(meta["bbox_min_x"]), float(meta["bbox_max_x"]),
            float(meta["bbox_min_y"]), float(meta["bbox_max_y"]),
            float(meta["bbox_min_z"]), float(meta["bbox_max_z"]),
            max_nodes, max_neighbors,
        ]],
    )


def load_topology(dat_path: str, ship: str, scale: str, zone_indices=None, *, topology=None, **session_kw):
    key = DatasetKey(ship=ship, scale=scale)
    zone_indices = list(zone_indices or [0, 1])
    payloads = topology if topology is not None else load_cfd_topology(dat_path, zone_indices)
    session = Session(
        session_kw.get("host", "127.0.0.1"),
        session_kw.get("port", "6667"),
        session_kw.get("user", "root"),
        session_kw.get("password", "root"),
    )
    session.open()
    try:
        for topo in payloads.values():
            export_topology_payload_to_iotdb(session, key.dataset_key, topo)
            print(
                f"IoTDB topology: {key.dataset_key}/{topo['zone_name']} "
                f"nodes={topo['node_count']} cells={topo['cell_count']} "
                f"max_nodes={topo['max_nodes_per_cell']} max_neighbors={topo['max_neighbors_per_cell']}"
            )
    finally:
        session.close()


def main():
    ap = argparse.ArgumentParser(description="Load CFD mesh topology into IoTDB")
    ap.add_argument("--dat", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="6667")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="root")
    args = ap.parse_args()
    if not os.path.isfile(args.dat):
        raise FileNotFoundError(args.dat)
    load_topology(
        args.dat, args.ship_type, args.scale, args.zone_indices,
        host=args.host, port=args.port, user=args.user, password=args.password,
    )


if __name__ == "__main__":
    main()
