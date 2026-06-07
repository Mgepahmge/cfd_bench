"""Load cell-centered variables from DAT into IoTDB post_processing domain."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from iotdb.Session import Session

from cfd_bench.core.context import DatasetKey
from cfd_bench.core.paths import iotdb_root
from cfd_bench.ingest.common.dat_files import dat_dir, iter_dat_files
from cfd_bench.ingest.decoder import CAE_Decoder
from cfd_bench.ingest.iotdb.io import load_dataframe_to_iotdb


def _step_from_filename(path: str) -> int:
    stem = Path(path).stem
    match = re.match(r"^(\d+)", stem)
    if not match:
        raise ValueError(f"cannot infer step from filename: {path}")
    return int(match.group(1))


def _vars_from_zone(zone):
    names = []
    columns = {}
    for i in range(3, len(zone.Variables)):
        name = str(zone.Variables[i]).strip()
        names.append(name)
        columns[name] = zone.Element_Variables[i - 3]
    return names, columns


def _cell_vars_leaf(zone_name: str, zone_index: int) -> str:
    name = zone_name.strip().lower()
    if "hull" in name or "wall" in name or zone_index == 1:
        return "cell_vars_hull"
    return "cell_vars"


def load_cell_vars(
    dat_path: str,
    ship: str,
    scale: str,
    zone_indices=None,
    *,
    host: str = "127.0.0.1",
    port: str = "6667",
    user: str = "root",
    password: str = "root",
):
    """Load scalars for one timestep into step_{t}.cell_vars / cell_vars_hull."""
    key = DatasetKey(ship=ship, scale=scale)
    zone_indices = zone_indices or [0, 1]
    step = _step_from_filename(dat_path)

    decoder = CAE_Decoder(3)
    decoder.Decode_dat_file(dat_path)

    session = Session(host, port, user, password)
    session.open()
    try:
        for zi in zone_indices:
            if zi >= len(decoder.Zones):
                continue
            zone = decoder.Zones[zi]
            zone_name = getattr(zone, "Zone_name", f"Zone_{zi}")
            leaf = _cell_vars_leaf(zone_name, zi)
            base = f"{iotdb_root()}.post_processing_management.{key.dataset_key}.step_{step}.{leaf}"
            _, columns = _vars_from_zone(zone)
            if not columns:
                continue
            cell_ids = list(range(zone.Element_count))
            df = pd.DataFrame(columns, index=cell_ids)
            load_dataframe_to_iotdb(base, session, df)
            print(f"IoTDB cell vars: {key.dataset_key} step={step} leaf={leaf} vars={list(columns)}")
    finally:
        session.close()


def load_cell_vars_from_dir(
    input_dir: str,
    ship: str,
    scale: str,
    zone_indices=None,
    **session_kw,
):
    for dat_path in iter_dat_files(input_dir):
        load_cell_vars(dat_path, ship, scale, zone_indices=zone_indices, **session_kw)


def main():
    ap = argparse.ArgumentParser(description="Load cell vars from DAT into IoTDB")
    ap.add_argument("--dat", help="single .dat file")
    ap.add_argument("--dat_dir", help="directory of .dat files")
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="6667")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="root")
    args = ap.parse_args()

    session_kw = dict(host=args.host, port=args.port, user=args.user, password=args.password)
    if args.dat:
        load_cell_vars(
            args.dat, args.ship_type, args.scale, args.zone_indices, **session_kw
        )
    elif args.dat_dir:
        load_cell_vars_from_dir(
            args.dat_dir, args.ship_type, args.scale, args.zone_indices, **session_kw
        )
    else:
        raise SystemExit("require --dat or --dat_dir")


if __name__ == "__main__":
    main()
