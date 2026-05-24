"""Run all CFD-Bench workloads W1–W8."""

from __future__ import annotations

import argparse
import importlib


def main():
    ap = argparse.ArgumentParser(description="Run CFD-Bench workloads W1–W8")
    ap.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"])
    ap.add_argument("--ships", nargs="+", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--backend", nargs="+", default=["postgresql", "iotdb", "tiledb", "vtk"])
    ap.add_argument("--vtk-dir", default="../vtk_dir")
    ap.add_argument("--tiledb-root", default="../TileDB_Instances")
    args, extra = ap.parse_known_args()

    for w in args.workloads:
        print(f"\n{'=' * 20} Running {w} {'=' * 20}")
        mod = importlib.import_module(f"cfd_bench.workloads.{w}.run")
        argv = ["run_workloads"]
        if args.ships:
            argv += ["--ships"] + args.ships
        argv += ["--duration", str(args.duration)]
        argv += ["--backend"] + args.backend
        argv += ["--vtk-dir", args.vtk_dir]
        argv += ["--tiledb-root", args.tiledb_root]
        argv += extra
        import sys

        old = sys.argv
        try:
            sys.argv = argv
            mod.main(args.ships)
        finally:
            sys.argv = old


if __name__ == "__main__":
    main()
