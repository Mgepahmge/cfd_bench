"""Load DAT post-processing files into VTK baseline directories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cfd_bench.core.context import DatasetKey
from cfd_bench.core.paths import resolve_vtk_dir, resolve_vtk_hull_dir
from cfd_bench.ingest.common.dat_files import iter_dat_files
from cfd_bench.ingest.vtk.mesh_builder import build_meshes_from_dat, write_vtk_mesh


def load_vtk(dat_path: str, ship: str, scale: str) -> None:
    """Export one .dat file to fluid/hull VTK baseline files."""
    key = DatasetKey(ship=ship, scale=scale)
    step = Path(dat_path).stem

    fluid_dir = resolve_vtk_dir()
    hull_dir = resolve_vtk_hull_dir()
    os.makedirs(fluid_dir, exist_ok=True)
    os.makedirs(hull_dir, exist_ok=True)

    fluid_mesh, hull_mesh = build_meshes_from_dat(dat_path)
    fluid_out = os.path.join(fluid_dir, f"{key.dataset_key}_GEO_{step}")
    hull_out = os.path.join(hull_dir, f"{key.dataset_key}_hull_{step}")
    fluid_file = write_vtk_mesh(fluid_mesh, fluid_out)
    hull_file = write_vtk_mesh(hull_mesh, hull_out)
    print(f"VTK exported: {fluid_file} , {hull_file}")


def load_vtk_from_dir(dat_dir: str, ship: str, scale: str) -> None:
    """Export all .dat files in a directory to VTK baseline directories."""
    for dat_path in iter_dat_files(dat_dir):
        load_vtk(dat_path, ship, scale)


def main():
    ap = argparse.ArgumentParser(description="Load DAT post-processing files into VTK baseline dirs")
    ap.add_argument("--dat", help="single .dat file")
    ap.add_argument("--dat_dir", help="directory of .dat files")
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    args = ap.parse_args()

    if not args.dat and not args.dat_dir:
        ap.error("provide --dat or --dat_dir")

    if args.dat:
        load_vtk(args.dat, args.ship_type, args.scale)
    if args.dat_dir:
        load_vtk_from_dir(args.dat_dir, args.ship_type, args.scale)


if __name__ == "__main__":
    main()
