"""Load legacy CFD cell fields into IoTDB from the shared canonical parser."""

from __future__ import annotations

import argparse

import numpy as np
from iotdb.Session import Session
from iotdb.utils.Field import TSDataType as T

from cfd_bench.core.context import DatasetKey
from cfd_bench.core.paths import iotdb_root
from cfd_bench.ingest.cfd.canonical import (
    iter_cfd_frames,
    load_cfd_topology,
    max_neighbor_diffs,
    validate_frame_topology,
)
from cfd_bench.ingest.iotdb.io import delete_timeseries_prefix, insert_numpy_columns, insert_tablet_chunked


def _leaf(zone_name: str, zone_index: int) -> str:
    z = str(zone_name).lower()
    return "cell_vars_hull" if ("hull" in z or "wall" in z or int(zone_index) == 1) else "cell_vars"


def load_cell_vars_from_dir(
    input_dir: str,
    ship: str,
    scale: str,
    zone_indices=None,
    *,
    host: str = "127.0.0.1",
    port: str = "6667",
    user: str = "root",
    password: str = "root",
    topology=None,
):
    key = DatasetKey(ship=ship, scale=scale)
    zone_indices = list(zone_indices or [0, 1])
    topology = topology if topology is not None else load_cfd_topology(input_dir, zone_indices)
    session = Session(host, port, user, password)
    session.open()
    try:
        seen_steps = []
        primary_var_sets = []
        primary_zone = next((z for z in topology if "fluid" in z.lower()), next(iter(topology)))
        for frame in iter_cfd_frames(input_dir, zone_indices):
            validate_frame_topology(frame, topology)
            seen_steps.append(int(frame.step))
            max_diff_union = {}
            for zone_frame in frame.zones:
                leaf = _leaf(zone_frame.zone_name, zone_frame.zone_index)
                path = f"{iotdb_root()}.post_processing_management.{key.dataset_key}.step_{frame.step}.{leaf}"
                delete_timeseries_prefix(session, path)
                vars_ = list(zone_frame.variables)
                insert_numpy_columns(
                    path,
                    session,
                    np.arange(zone_frame.cell_count, dtype=np.int64),
                    {v: zone_frame.variables[v] for v in vars_},
                    [T.DOUBLE] * len(vars_),
                )
                # W3 operates on the primary/fluid zone.  Materialise its
                # derived metadata in IoTDB itself; no TileDB sidecar coupling.
                if leaf == "cell_vars":
                    max_diff_union.update(max_neighbor_diffs(topology[zone_frame.zone_name], zone_frame.variables))
                    if zone_frame.zone_name == primary_zone:
                        primary_var_sets.append(set(str(v).upper() for v in vars_))
                print(
                    f"IoTDB cell vars: {key.dataset_key} zone={zone_frame.zone_name} "
                    f"step={frame.step} cells={zone_frame.cell_count} vars={vars_}"
                )
            if max_diff_union:
                dpath = f"{iotdb_root()}.derived.{key.dataset_key}.step_{frame.step}.max_diff"
                delete_timeseries_prefix(session, dpath)
                names = sorted(max_diff_union)
                insert_tablet_chunked(
                    dpath, session, [0], names, [T.DOUBLE] * len(names),
                    [[float(max_diff_union[n]) for n in names]],
                )

        # Canonical legacy-CFD runtime metadata.  Keep this separate from the
        # frozen H5 metadata tree so both ingest families remain independent.
        common_vars = sorted(set.intersection(*primary_var_sets)) if primary_var_sets else []
        meta_path = f"{iotdb_root()}.cfd_metadata.{key.dataset_key}.dataset_meta"
        delete_timeseries_prefix(session, meta_path)
        primary = topology[primary_zone]
        insert_tablet_chunked(
            meta_path,
            session,
            [0],
            [
                "is_cfd", "zone", "zones_csv", "variables_csv",
                "timesteps_csv", "node_count", "cell_count",
            ],
            [T.BOOLEAN, T.TEXT, T.TEXT, T.TEXT, T.TEXT, T.INT64, T.INT64],
            [[
                True,
                str(primary_zone),
                ",".join(topology.keys()),
                ",".join(common_vars),
                ",".join(str(x) for x in sorted(set(seen_steps))),
                int(primary["node_count"]),
                int(primary["cell_count"]),
            ]],
        )
    finally:
        session.close()


def load_cell_vars(dat_path: str, ship: str, scale: str, zone_indices=None, **session_kw):
    # Kept for script/API compatibility.  A single file is also a valid input
    # to the canonical frame iterator.
    return load_cell_vars_from_dir(dat_path, ship, scale, zone_indices, **session_kw)


def main():
    ap = argparse.ArgumentParser(description="Load CFD cell vars into IoTDB")
    ap.add_argument("--dat")
    ap.add_argument("--dat_dir")
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="6667")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="root")
    args = ap.parse_args()
    path = args.dat or args.dat_dir
    if not path:
        raise SystemExit("require --dat or --dat_dir")
    load_cell_vars_from_dir(
        path, args.ship_type, args.scale, args.zone_indices,
        host=args.host, port=args.port, user=args.user, password=args.password,
    )


if __name__ == "__main__":
    main()
