"""Load mesh topology from DAT into IoTDB mesh_static domain."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from iotdb.Session import Session

from cfd_bench.core.context import DatasetKey
from cfd_bench.core.paths import iotdb_root
from cfd_bench.ingest.common.topology_export import export_zone_topology
from cfd_bench.ingest.decoder import CAE_Decoder
from cfd_bench.ingest.iotdb.io import load_dataframe_to_iotdb


def export_zone_to_iotdb(session: Session, dataset_key: str, zone_name: str, topo: dict):
    zone_key = zone_name.strip().replace(" ", "_") or "Zone_0"
    base = f"{iotdb_root()}.mesh_static.{dataset_key}.{zone_key}"

    node_ids = list(range(topo["node_count"]))
    df_nodes = pd.DataFrame(
        {"x": topo["nodes"]["x"], "y": topo["nodes"]["y"], "z": topo["nodes"]["z"]},
        index=node_ids,
    )
    load_dataframe_to_iotdb(f"{base}.nodes", session, df_nodes)

    cell_ids = list(range(topo["cell_count"]))
    df_cells = pd.DataFrame(
        topo["cells"],
        index=cell_ids,
        columns=["cx", "cy", "cz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cell_type"],
    )
    load_dataframe_to_iotdb(f"{base}.cells", session, df_cells)

    node_cols = [f"node_id_{i}" for i in range(16)]
    df_cell_nodes = pd.DataFrame(topo["cell_nodes"], index=cell_ids, columns=node_cols)
    load_dataframe_to_iotdb(f"{base}.cell_nodes", session, df_cell_nodes)

    adj_cols = [f"neighbor_id_{i}" for i in range(16)]
    df_adj = pd.DataFrame(topo["adjacency"], index=cell_ids, columns=adj_cols)
    load_dataframe_to_iotdb(f"{base}.cell_adjacency", session, df_adj)

    if topo["face_planes"]:
        df_face = pd.DataFrame(
            topo["face_planes"],
            index=np.arange(len(topo["face_planes"])),
            columns=["cell_id", "neighbor_id", "nx", "ny", "nz", "d", "face_area", "face_cx", "face_cy", "face_cz"],
        )
        load_dataframe_to_iotdb(f"{base}.face_planes", session, df_face)
    if topo["boundary_faces"]:
        df_bf = pd.DataFrame(
            topo["boundary_faces"],
            index=np.arange(len(topo["boundary_faces"])),
            columns=["cell_id", "patch_code", "nx", "ny", "nz", "area", "cx", "cy", "cz"],
        )
        load_dataframe_to_iotdb(f"{base}.boundary_faces", session, df_bf)

    meta = topo["bbox"]
    df_meta = pd.DataFrame(
        [[topo["node_count"], topo["cell_count"], topo["face_count"],
          meta["bbox_min_x"], meta["bbox_max_x"], meta["bbox_min_y"], meta["bbox_max_y"],
          meta["bbox_min_z"], meta["bbox_max_z"]]],
        index=[0],
        columns=["node_count", "cell_count", "face_count", "bbox_min_x", "bbox_max_x",
                 "bbox_min_y", "bbox_max_y", "bbox_min_z", "bbox_max_z"],
    )
    load_dataframe_to_iotdb(f"{base}.mesh_meta", session, df_meta)


def load_topology(dat_path: str, ship: str, scale: str, zone_indices=None, **session_kw):
    key = DatasetKey(ship=ship, scale=scale)
    zone_indices = zone_indices or [0]
    data = CAE_Decoder(3)
    data.Decode_dat_file(dat_path)
    session = Session(
        session_kw.get("host", "127.0.0.1"),
        session_kw.get("port", "6667"),
        session_kw.get("user", "root"),
        session_kw.get("password", "root"),
    )
    session.open()
    try:
        for zi in zone_indices:
            if zi >= len(data.Zones):
                continue
            zone = data.Zones[zi]
            topo = export_zone_topology(zone)
            export_zone_to_iotdb(session, key.dataset_key, zone.Zone_name, topo)
    finally:
        session.close()


def main():
    ap = argparse.ArgumentParser(description="Load mesh topology from DAT into IoTDB mesh_static")
    ap.add_argument("--dat", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0])
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
