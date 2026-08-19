"""Load legacy CFD cell fields into TileDB from the shared canonical parser."""

from __future__ import annotations

import argparse

import numpy as np

from cfd_bench.core.context import DatasetKey
from cfd_bench.ingest.cfd.canonical import (
    iter_cfd_frames,
    load_cfd_topology,
    max_neighbor_diffs,
    validate_frame_topology,
)
from cfd_bench.ingest.tiledb import io as tiledb_io


def _leaf(zone_name: str, zone_index: int) -> str:
    z = str(zone_name).lower()
    return "cell_vars_hull" if ("hull" in z or "wall" in z or int(zone_index) == 1) else "cell_vars"


def load_cell_vars_from_path(
    input_path: str,
    ship: str,
    scale: str,
    root: str = "TileDB_Instances",
    zone_indices=None,
    overwrite: bool = True,
    topology=None,
):
    key = DatasetKey(ship=ship, scale=scale)
    zone_indices = list(zone_indices or [0, 1])
    topology = topology if topology is not None else load_cfd_topology(input_path, zone_indices)
    primary_zone = next((z for z in topology if "fluid" in z.lower()), next(iter(topology)))
    seen_steps = []
    primary_var_sets = []
    for frame in iter_cfd_frames(input_path, zone_indices):
        seen_steps.append(int(frame.step))
        validate_frame_topology(frame, topology)
        for zone_frame in frame.zones:
            vars_ = [str(v).upper() for v in zone_frame.variables]
            uri = tiledb_io.post_uri(root, key.dataset_key, frame.step, _leaf(zone_frame.zone_name, zone_frame.zone_index))
            tiledb_io.write_cell_vars(
                uri,
                {v: np.asarray(zone_frame.variables[v], dtype=np.float32) for v in vars_},
                zone_frame.cell_count,
                vars_,
                overwrite=overwrite,
            )
            if _leaf(zone_frame.zone_name, zone_frame.zone_index) == "cell_vars":
                if zone_frame.zone_name == primary_zone:
                    primary_var_sets.append(set(vars_))
                diffs = max_neighbor_diffs(topology[zone_frame.zone_name], zone_frame.variables)
                tiledb_io.write_max_diffs(
                    tiledb_io.derived_uri(root, key.dataset_key, frame.step, "max_diff"),
                    diffs,
                    overwrite=overwrite,
                )
            print(
                f"TileDB cell vars: {key.dataset_key} zone={zone_frame.zone_name} "
                f"step={frame.step} cells={zone_frame.cell_count} vars={vars_}"
            )

    common_vars = sorted(set.intersection(*primary_var_sets)) if primary_var_sets else []
    primary = topology[primary_zone]
    tiledb_io.write_cfd_dataset_meta(
        tiledb_io.cfd_metadata_uri(root, key.dataset_key, "dataset_meta"),
        {
            "zone": str(primary_zone),
            "variables_csv": ",".join(common_vars),
            "common_variables_csv": ",".join(common_vars),
            "timesteps_csv": ",".join(str(x) for x in sorted(set(seen_steps))),
            "node_count": int(primary["node_count"]),
            "cell_count": int(primary["cell_count"]),
            "frames": [],
            "zones_csv": ",".join(topology.keys()),
        },
        overwrite=overwrite,
    )


def load_cell_vars(dat_path: str, ship: str, scale: str, root: str = "TileDB_Instances", overwrite: bool = True, zone_indices=None):
    return load_cell_vars_from_path(dat_path, ship, scale, root, zone_indices=zone_indices, overwrite=overwrite)


def main():
    ap = argparse.ArgumentParser(description="Load CFD cell vars into TileDB")
    ap.add_argument("--dat")
    ap.add_argument("--dat_dir")
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--root", default="TileDB_Instances")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()
    path = args.dat or args.dat_dir
    if not path:
        raise SystemExit("require --dat or --dat_dir")
    load_cell_vars_from_path(path, args.ship_type, args.scale, args.root, zone_indices=args.zone_indices, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
