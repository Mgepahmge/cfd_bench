import argparse
import random
import time
import os
import sys

import numpy as np

# Ensure imports work even when running from outside src/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from iotdb_replace_vtk import IoTDBMeshInterface


def random_points(bounds, n):
    p = np.random.rand(n, 3)
    p[:, 0] = p[:, 0] * (bounds[1] - bounds[0]) + bounds[0]
    p[:, 1] = p[:, 1] * (bounds[3] - bounds[2]) + bounds[2]
    p[:, 2] = p[:, 2] * (bounds[5] - bounds[4]) + bounds[4]
    return p


def random_line(bounds):
    return (
        np.array(
            [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
        ),
        np.array(
            [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
        ),
    )


def random_plane(bounds):
    return (
        np.array(
            [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
        ),
        np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]),
    )


def main():
    ap = argparse.ArgumentParser(description="Benchmark W1 style operations for iotdb_replace_vtk")
    ap.add_argument("--dataset_key", required=True)
    ap.add_argument("--zone", default="0_Fluid")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--vtk_file", default=None, help="optional vtk for correctness comparison")
    ap.add_argument("--ops", type=int, default=100)
    ap.add_argument("--var", default="P")
    args = ap.parse_args()

    vtk_mesh = None
    vtk_api = None
    bounds = (-1, 1, -1, 1, -1, 1)
    if args.vtk_file:
        import VTK_Interface as VTK_API  # lazy import, only for comparison mode

        vtk_api = VTK_API.VTK_Interface()
        vtk_mesh = vtk_api.vtk_connect(args.vtk_file)
        bounds = vtk_mesh.GetBounds()

    with IoTDBMeshInterface() as api:
        ctx = api.iotdb_connect(args.dataset_key, args.step, zone=args.zone, need=("w1",))
        print("caps:", sorted(ctx.available_caps), "missing:", sorted(ctx.missing_caps))

        t0 = time.time()
        for _ in range(args.ops):
            pts = random_points(bounds, 64)
            cids = api.iotdb_point_intersection(pts)
            valid = [int(x) for x in cids if int(x) >= 0]
            if valid:
                api.iotdb_point_query(valid, args.var)
        t_point = time.time() - t0

        t0 = time.time()
        for _ in range(args.ops):
            s, e = random_line(bounds)
            cids = api.iotdb_line_intersection(s, e)
            if cids.size > 0:
                api.iotdb_point_query(cids[:128], args.var)
        t_line = time.time() - t0

        t0 = time.time()
        for _ in range(args.ops):
            o, n = random_plane(bounds)
            cids = api.iotdb_plane_intersection(o, n)
            if cids.size > 0:
                api.iotdb_point_query(cids[:128], args.var)
        t_plane = time.time() - t0

        print(f"IoTDBReplace point={t_point:.3f}s line={t_line:.3f}s plane={t_plane:.3f}s")

        if vtk_mesh is not None:
            s, e = random_line(bounds)
            io_ids = set(api.iotdb_line_intersection(s, e).tolist())
            vt_ids = set(vtk_api.vtk_line_intersection(vtk_mesh, s, e).tolist())
            inter = len(io_ids & vt_ids)
            union = max(1, len(io_ids | vt_ids))
            print(f"Line Jaccard(one-sample)={inter/union:.4f}")


if __name__ == "__main__":
    main()
