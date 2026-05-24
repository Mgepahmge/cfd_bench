"""Load cell-centered variables from DAT into TileDB post_processing."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from cfd_bench.core.context import DatasetKey
from cfd_bench.ingest.decoder import CAE_Decoder
from cfd_bench.ingest.tiledb import io as tiledb_io


def _step_from_filename(path: str) -> int:
    stem = Path(path).stem
    m = re.match(r"^(\d+)", stem)
    if not m:
        raise ValueError(f"cannot infer step from filename: {path}")
    return int(m.group(1))


def _vars_from_zone(zone):
    names = []
    data = {}
    for i in range(3, len(zone.Variables)):
        name = zone.Variables[i]
        names.append(name)
        data[name] = zone.Element_Variables[i - 3].astype("float32")
    return names, data


def load_cell_vars(dat_path: str, ship: str, scale: str, root: str = "TileDB_Instances", overwrite: bool = True):
    key = DatasetKey(ship=ship, scale=scale)
    step = _step_from_filename(dat_path)
    decoder = CAE_Decoder(3)
    decoder.Decode_dat_file(dat_path)

    if len(decoder.Zones) >= 1:
        fluid = decoder.Zones[0]
        var_names, var_data = _vars_from_zone(fluid)
        uri = tiledb_io.post_uri(root, key.dataset_key, step, "cell_vars.tdb")
        tiledb_io.write_cell_vars(uri, var_data, fluid.Element_count, var_names, overwrite=overwrite)

    if len(decoder.Zones) >= 2:
        hull = decoder.Zones[1]
        var_names, var_data = _vars_from_zone(hull)
        uri = tiledb_io.post_uri(root, key.dataset_key, step, "cell_vars_hull.tdb")
        tiledb_io.write_cell_vars(uri, var_data, hull.Element_count, var_names, overwrite=overwrite)


def main():
    ap = argparse.ArgumentParser(description="Load cell vars from DAT into TileDB")
    ap.add_argument("--dat", help="single .dat file")
    ap.add_argument("--dat_dir", help="directory of .dat files")
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--root", default="TileDB_Instances")
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()
    files = []
    if args.dat:
        files.append(args.dat)
    if args.dat_dir:
        for f in sorted(os.listdir(args.dat_dir)):
            fp = os.path.join(args.dat_dir, f)
            if fp.endswith(".dat"):
                files.append(fp)
    for fp in files:
        load_cell_vars(fp, args.ship_type, args.scale, args.root, args.overwrite)
        print(f"done {fp}")


if __name__ == "__main__":
    main()
