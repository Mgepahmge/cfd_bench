"""Load cell-centered variables from DAT into TileDB post_processing."""

from __future__ import annotations

import argparse
import csv
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


def _compute_max_diffs_from_zone(zone, output_dir: str, dataset_key: str, step: int, overwrite: bool = True):
    """Compute σᵥ = max|Q(c₁,v) - Q(c₂,v)| for all geometrically adjacent cell pairs (c₁↔c₂).

    Derived purely from raw DAT data — no database dependency.
    Writes result to Max_Range/{dataset_key}_{step}_max_diffs.csv for W3 isosurface workload.
    """
    import numpy as np

    # 1. Extract element-centered variables (skip first 3 coordinate vars)
    var_names = zone.Variables[3:]  # e.g. ['U', 'V', 'W', 'P', 'K', 'E']
    # zone.Element_Variables[i] corresponds to zone.Variables[3+i]
    elem_vars = [np.asarray(zone.Element_Variables[i], dtype=np.float64) for i in range(len(var_names))]

    # 2. Build adjacency from face connectivity (pure DAT data)
    adjacency = zone.construct_element_adjacency()  # list[list[int]], length = Element_count

    # 3. Compute σᵥ for each variable
    n_cells = zone.Element_count
    result = {}
    for vi, var_name in enumerate(var_names):
        vals = elem_vars[vi]  # (n_cells,) float64
        max_diff = 0.0
        for c in range(n_cells):
            for nb in adjacency[c]:
                if nb >= 0:
                    diff = abs(float(vals[c]) - float(vals[nb]))
                    if diff > max_diff:
                        max_diff = diff
        result[var_name] = max_diff

    # 4. Write CSV (format expected by W3's read_max_diffs)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset_key}_{step}_max_diffs.csv")
    if os.path.exists(csv_path) and not overwrite:
        return csv_path
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "max_diff"])
        for var_name in var_names:
            writer.writerow([var_name, f"{result[var_name]:.6g}"])
    print(f"Max_diffs saved: {csv_path}")
    return csv_path


def load_cell_vars(dat_path: str, ship: str, scale: str, root: str = "TileDB_Instances", overwrite: bool = True):
    key = DatasetKey(ship=ship, scale=scale)
    step = _step_from_filename(dat_path)
    decoder = CAE_Decoder(3)
    decoder.Decode_dat_file(dat_path)

    if len(decoder.Zones) >= 1:
        fluid = decoder.Zones[0]
        var_names, var_data = _vars_from_zone(fluid)
        uri = tiledb_io.post_uri(root, key.dataset_key, step, "cell_vars")
        tiledb_io.write_cell_vars(uri, var_data, fluid.Element_count, var_names, overwrite=overwrite)

    if len(decoder.Zones) >= 2:
        hull = decoder.Zones[1]
        var_names, var_data = _vars_from_zone(hull)
        uri = tiledb_io.post_uri(root, key.dataset_key, step, "cell_vars_hull")
        tiledb_io.write_cell_vars(uri, var_data, hull.Element_count, var_names, overwrite=overwrite)

    # Compute max_diffs from raw DAT data (fluid zone only, pure dat-derived)
    if len(decoder.Zones) >= 1:
        fluid = decoder.Zones[0]
        output_dir = os.path.join(os.path.dirname(os.path.abspath(root)), "Max_Range")
        _compute_max_diffs_from_zone(fluid, output_dir, key.dataset_key, step, overwrite=overwrite)


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
