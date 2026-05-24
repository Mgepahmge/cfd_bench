"""Unified ingest pipeline entry point."""

from __future__ import annotations

import argparse
import os


def run_pipeline(
    backend: str,
    dat_path: str,
    ship: str = "JBC",
    scale: str = "615k",
    with_topology: bool = False,
    **kwargs,
):
    if backend == "iotdb":
        if with_topology:
            from cfd_bench.ingest.iotdb.load_topology import load_topology
            load_topology(dat_path, ship, scale, zone_indices=kwargs.get("zone_indices", [0]), **kwargs)
        else:
            raise NotImplementedError("IoTDB cell_vars-only ingest: use cfd_bench.ingest.dataloaders.LoadDataTo_IoTDB")
    elif backend == "tiledb":
        root = kwargs.get("root", "TileDB_Instances")
        if with_topology:
            from cfd_bench.ingest.tiledb.load_topology import load_topology
            load_topology(dat_path, ship, scale, root, kwargs.get("zone_indices", [0]))
        from cfd_bench.ingest.tiledb.load_cell_vars import load_cell_vars
        load_cell_vars(dat_path, ship, scale, root)
    elif backend == "postgresql":
        raise NotImplementedError("PostgreSQL ingest: use ingest/postgresql/ scripts or legacy LoadDataTo_PG")
    else:
        raise ValueError(f"unknown backend: {backend}")


def main():
    ap = argparse.ArgumentParser(description="CFD-Bench unified ingest pipeline")
    ap.add_argument("--backend", choices=["iotdb", "tiledb", "postgresql"], required=True)
    ap.add_argument("--dat", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--with-topology", action="store_true")
    ap.add_argument("--root", default="TileDB_Instances")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    if not os.path.isfile(args.dat):
        raise FileNotFoundError(args.dat)
    run_pipeline(
        args.backend, args.dat, args.ship_type, args.scale,
        with_topology=args.with_topology, root=args.root, zone_indices=args.zone_indices,
    )


if __name__ == "__main__":
    main()
