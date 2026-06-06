"""Load mesh topology from DAT into PostgreSQL."""

from __future__ import annotations

import argparse
import os

from cfd_bench.ingest.postgresql.pg_io import load_topology_from_dat, parse_ship_type_and_scale


def main():
    ap = argparse.ArgumentParser(description="Load mesh topology from DAT into PostgreSQL")
    ap.add_argument("--dat", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--no_face_planes", action="store_true")
    ap.add_argument("--no_nodes", action="store_true")
    args = ap.parse_args()
    if not os.path.isfile(args.dat):
        raise FileNotFoundError(args.dat)
    ship, scale = parse_ship_type_and_scale(args.ship_type, args.scale)
    load_topology_from_dat(
        args.dat,
        ship,
        scale,
        zone_indices=args.zone_indices,
        export_face_planes=not args.no_face_planes,
        export_nodes=not args.no_nodes,
    )
    print(f"PG topology loaded: {ship}_{scale}")


if __name__ == "__main__":
    main()
