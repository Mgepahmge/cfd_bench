"""Build PostgreSQL point-location buckets.

For canonical legacy CFD ingests, ``cell_bounds`` is available and cells are
registered in *every* regular-grid bucket overlapped by their AABB.  This fixes
the historical centroid-bucket mismatch that could make a point inside a cell
return no candidate.  H5 datasets do not populate ``cell_bounds`` and retain
their existing centroid-based build path unchanged.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from psycopg2.extras import execute_values

from cfd_bench.ingest.postgresql.pg_io import parse_ship_type_and_scale, pg_connect


def _idx(value, origin, step, n):
    return int(min(n - 1, max(0, math.floor((float(value) - origin) / step))))


def build_point_locator_grid(
    ship_type: str,
    scale: str,
    zone_type: str,
    nx: int = 80,
    ny: int = 80,
    nz: int = 40,
    *,
    db_name: str = "cae_data",
    db_user: str = "postgres",
    db_password: str = "123456",
    db_host: str = "localhost",
    db_port: str = "5432",
):
    conn = pg_connect(db_name, db_user, db_password, db_host, db_port)
    cur = conn.cursor()
    try:
        cur.execute("SELECT to_regclass('public.cell_bounds')")
        has_table = bool(cur.fetchone()[0])
        has_cfd_bounds = False
        if has_table:
            cur.execute(
                "SELECT 1 FROM cell_bounds WHERE ship_type=%s AND scale=%s AND zone_type=%s LIMIT 1",
                (ship_type, scale, zone_type),
            )
            has_cfd_bounds = cur.fetchone() is not None

        cur.execute(
            "DELETE FROM point_locator_grid WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )

        if has_cfd_bounds:
            # Canonical CFD: build the AABB-covered regular grid entirely in
            # PostgreSQL. Pulling millions of cell_bounds into Python and
            # retaining one list per bucket was both slower and far more memory
            # hungry. generate_series expands only the overlapped bucket span.
            cur.execute(
                """
                SELECT MIN(min_x),MAX(max_x),MIN(min_y),MAX(max_y),MIN(min_z),MAX(max_z)
                FROM cell_bounds WHERE ship_type=%s AND scale=%s AND zone_type=%s
                """,
                (ship_type, scale, zone_type),
            )
            bbox = cur.fetchone()
            if not bbox or bbox[0] is None:
                raise RuntimeError("cell_bounds has no data")
            xmin, xmax, ymin, ymax, zmin, zmax = map(float, bbox)
            dx = max((xmax - xmin) / max(nx, 1), 1e-12)
            dy = max((ymax - ymin) / max(ny, 1), 1e-12)
            dz = max((zmax - zmin) / max(nz, 1), 1e-12)
            cur.execute(
                """
                WITH spans AS (
                    SELECT cell_id,
                           GREATEST(0, LEAST(%s-1, FLOOR((min_x-%s)/%s)::int)) AS ix0,
                           GREATEST(0, LEAST(%s-1, FLOOR((max_x-%s)/%s)::int)) AS ix1,
                           GREATEST(0, LEAST(%s-1, FLOOR((min_y-%s)/%s)::int)) AS iy0,
                           GREATEST(0, LEAST(%s-1, FLOOR((max_y-%s)/%s)::int)) AS iy1,
                           GREATEST(0, LEAST(%s-1, FLOOR((min_z-%s)/%s)::int)) AS iz0,
                           GREATEST(0, LEAST(%s-1, FLOOR((max_z-%s)/%s)::int)) AS iz1
                    FROM cell_bounds
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                ), bucket_cells AS (
                    SELECT gx AS ix, gy AS iy, gz AS iz,
                           array_agg(cell_id ORDER BY cell_id) AS cell_ids
                    FROM spans s
                    CROSS JOIN LATERAL generate_series(s.ix0, s.ix1) AS gx
                    CROSS JOIN LATERAL generate_series(s.iy0, s.iy1) AS gy
                    CROSS JOIN LATERAL generate_series(s.iz0, s.iz1) AS gz
                    GROUP BY gx,gy,gz
                )
                INSERT INTO point_locator_grid
                (ship_type,scale,zone_type,bucket_id,ix,iy,iz,
                 x_min,x_max,y_min,y_max,z_min,z_max,bbox_xy,cell_ids)
                SELECT %s,%s,%s,
                       (ix + %s * (iy + %s * iz))::bigint,
                       ix,iy,iz,
                       %s + ix*%s, %s + (ix+1)*%s,
                       %s + iy*%s, %s + (iy+1)*%s,
                       %s + iz*%s, %s + (iz+1)*%s,
                       ST_SetSRID(ST_MakeEnvelope(
                           %s + ix*%s, %s + iy*%s,
                           %s + (ix+1)*%s, %s + (iy+1)*%s
                       ),0),
                       cell_ids
                FROM bucket_cells
                """,
                (
                    nx, xmin, dx, nx, xmin, dx,
                    ny, ymin, dy, ny, ymin, dy,
                    nz, zmin, dz, nz, zmin, dz,
                    ship_type, scale, zone_type,
                    ship_type, scale, zone_type, nx, ny,
                    xmin, dx, xmin, dx,
                    ymin, dy, ymin, dy,
                    zmin, dz, zmin, dz,
                    xmin, dx, ymin, dy, xmin, dx, ymin, dy,
                ),
            )
            inserted = max(int(cur.rowcount or 0), 0)
            conn.commit()
            print(f"point_locator_grid: buckets={inserted} mode=aabb-sql")
            return

        # Frozen H5 / older-database centroid-bucket build path.
        buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        cur.execute(
            """
            SELECT min(x),max(x),min(y),max(y),min(z),max(z) FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        bbox = cur.fetchone()
        if not bbox or bbox[0] is None:
            raise RuntimeError("cell_centroid has no data")
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, bbox)
        dx = max((xmax - xmin) / max(nx, 1), 1e-12)
        dy = max((ymax - ymin) / max(ny, 1), 1e-12)
        dz = max((zmax - zmin) / max(nz, 1), 1e-12)
        cur.execute(
            """
            SELECT cell_id,x,y,z FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        for cid, x, y, z in cur.fetchall():
            buckets[(_idx(x, xmin, dx, nx), _idx(y, ymin, dy, ny), _idx(z, zmin, dz, nz))].append(int(cid))

        rows = []
        for (ix, iy, iz), cell_ids in buckets.items():
            bxmin, bxmax = xmin + ix * dx, xmin + (ix + 1) * dx
            bymin, bymax = ymin + iy * dy, ymin + (iy + 1) * dy
            bzmin, bzmax = zmin + iz * dz, zmin + (iz + 1) * dz
            bucket_id = int(ix + nx * (iy + ny * iz))
            rows.append((
                ship_type, scale, zone_type, bucket_id, ix, iy, iz,
                bxmin, bxmax, bymin, bymax, bzmin, bzmax,
                f"POLYGON(({bxmin} {bymin},{bxmax} {bymin},{bxmax} {bymax},{bxmin} {bymax},{bxmin} {bymin}))",
                sorted(set(cell_ids)),
            ))
        if rows:
            execute_values(
                cur,
                """
                INSERT INTO point_locator_grid
                (ship_type,scale,zone_type,bucket_id,ix,iy,iz,x_min,x_max,y_min,y_max,z_min,z_max,bbox_xy,cell_ids)
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
        print(f"point_locator_grid: buckets={len(rows)} mode=centroid")
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
