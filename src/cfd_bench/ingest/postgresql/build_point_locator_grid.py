"""Build point_locator_grid spatial buckets from cell centroids."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from psycopg2.extras import execute_values

from cfd_bench.ingest.postgresql.pg_io import parse_ship_type_and_scale, pg_connect


def build_point_locator_grid(ship_type: str, scale: str, zone_type: str, nx: int = 80, ny: int = 80, nz: int = 40):
    conn = pg_connect()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT min(x), max(x), min(y), max(y), min(z), max(z)
            FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        bbox = cur.fetchone()
        if not bbox or bbox[0] is None:
            raise RuntimeError("cell_centroid 无数据")
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, bbox)
        dx = max((xmax - xmin) / max(nx, 1), 1e-12)
        dy = max((ymax - ymin) / max(ny, 1), 1e-12)
        dz = max((zmax - zmin) / max(nz, 1), 1e-12)
        cur.execute(
            """
            SELECT cell_id, x, y, z FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for cell_id, x, y, z in cur.fetchall():
            ix = int(min(nx - 1, max(0, math.floor((float(x) - xmin) / dx))))
            iy = int(min(ny - 1, max(0, math.floor((float(y) - ymin) / dy))))
            iz = int(min(nz - 1, max(0, math.floor((float(z) - zmin) / dz))))
            buckets[(ix, iy, iz)].append(int(cell_id))
        cur.execute(
            "DELETE FROM point_locator_grid WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )
        rows = []
        for (ix, iy, iz), cell_ids in buckets.items():
            bxmin, bxmax = xmin + ix * dx, xmin + (ix + 1) * dx
            bymin, bymax = ymin + iy * dy, ymin + (iy + 1) * dy
            bzmin, bzmax = zmin + iz * dz, zmin + (iz + 1) * dz
            bucket_id = int(ix + nx * (iy + ny * iz))
            rows.append(
                (
                    ship_type, scale, zone_type, bucket_id, ix, iy, iz,
                    bxmin, bxmax, bymin, bymax, bzmin, bzmax,
                    f"POLYGON(({bxmin} {bymin},{bxmax} {bymin},{bxmax} {bymax},{bxmin} {bymax},{bxmin} {bymin}))",
                    sorted(set(cell_ids)),
                )
            )
        if rows:
            execute_values(
                cur,
                """
                INSERT INTO point_locator_grid
                (ship_type, scale, zone_type, bucket_id, ix, iy, iz,
                 x_min, x_max, y_min, y_max, z_min, z_max, bbox_xy, cell_ids)
                VALUES %s
                """,
                rows,
                template=(
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    "ST_SetSRID(ST_GeomFromText(%s,0),0),%s)"
                ),
                page_size=1000,
            )
        conn.commit()
        print(f"point_locator_grid: buckets={len(rows)}")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Build point_locator_grid")
    ap.add_argument("--ship_type", default="JBC_615k")
    ap.add_argument("--scale", default=None)
    ap.add_argument("--zone_type", default="0_Fluid")
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nz", type=int, default=40)
    args = ap.parse_args()
    ship, scale = parse_ship_type_and_scale(args.ship_type, args.scale)
    build_point_locator_grid(ship, scale, args.zone_type, args.nx, args.ny, args.nz)


if __name__ == "__main__":
    main()
