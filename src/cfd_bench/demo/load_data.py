"""Load data via unified ingest pipeline."""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="Load CFD data to databases")
    ap.add_argument("--modern", action="store_true", help="Use cfd_bench ingest pipeline (topology + vars)")
    ap.add_argument("--backend", choices=["iotdb", "tiledb", "postgresql"])
    ap.add_argument("--dat", help="DAT file for modern pipeline")
    ap.add_argument("--with-topology", action="store_true")
    args = ap.parse_known_args()[0]

    if args.modern and args.backend and args.dat:
        from cfd_bench.ingest.pipeline import run_pipeline

        run_pipeline(args.backend, args.dat, with_topology=args.with_topology)
        return

    from cfd_bench.ingest.dataloaders.LoadDataTo_PG import main as load_pg
    from cfd_bench.ingest.dataloaders.LoadDataTo_IoTDB import main as load_iotdb
    from cfd_bench.ingest.dataloaders.LoadDataTo_TileDB import main as load_tiledb
    from cfd_bench.ingest.dataloaders.LoadDataTo_VTK import main as load_vtk

    load_pg()
    load_iotdb()
    load_tiledb()
    load_vtk()


if __name__ == "__main__":
    main()
