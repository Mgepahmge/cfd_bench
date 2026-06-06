"""Build PostGIS cell_geom_full from cell_nodes + node_coordinates."""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from psycopg2.extras import execute_values

from cfd_bench.ingest.postgresql.pg_io import parse_ship_type_and_scale, pg_connect



def _multipolygon_z_wkt(points: np.ndarray, simplices: np.ndarray) -> str:
    polys = []
    for tri in simplices:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        ring = (
            f"{a[0]:.17g} {a[1]:.17g} {a[2]:.17g}, "
            f"{b[0]:.17g} {b[1]:.17g} {b[2]:.17g}, "
            f"{c[0]:.17g} {c[1]:.17g} {c[2]:.17g}, "
            f"{a[0]:.17g} {a[1]:.17g} {a[2]:.17g}"
        )
        polys.append(f"(({ring}))")
    return "MULTIPOLYGON Z (" + ",".join(polys) + ")"


def build_cell_geom_full(ship_type: str, scale: str, zone_type: str, batch_size: int = 500):
    try:
        from scipy.spatial import ConvexHull
    except ImportError as e:
        raise RuntimeError("需要 scipy: pip install scipy") from e

    conn = pg_connect()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT node_id, x, y, z FROM node_coordinates
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        node_xyz = {int(n): (float(x), float(y), float(z)) for n, x, y, z in cur.fetchall()}
        cur.execute(
            """
            SELECT cn.cell_id, cn.node_ids, cc.x, cc.y, cc.z
            FROM cell_nodes cn
            LEFT JOIN cell_centroid cc
              ON cc.ship_type=cn.ship_type AND cc.scale=cn.scale
             AND cc.zone_type=cn.zone_type AND cc.cell_id=cn.cell_id
            WHERE cn.ship_type=%s AND cn.scale=%s AND cn.zone_type=%s
            ORDER BY cn.cell_id
            """,
            (ship_type, scale, zone_type),
        )
        built = []
        for cell_id, node_ids, cx, cy, cz in cur.fetchall():
            pts = []
            for nid in node_ids or []:
                xyz = node_xyz.get(int(nid))
                if xyz is None:
                    pts = []
                    break
                pts.append(xyz)
            if len(pts) < 4:
                continue
            arr = np.array(pts, dtype=np.float64)
            try:
                hull = ConvexHull(arr)
            except Exception:
                continue
            if hull.simplices.size == 0:
                continue
            geom_wkt = _multipolygon_z_wkt(arr, hull.simplices)
            if cx is None:
                cx, cy, cz = np.mean(arr[:, 0]), np.mean(arr[:, 1]), np.mean(arr[:, 2])
            centroid_wkt = f"POINT Z ({float(cx):.17g} {float(cy):.17g} {float(cz):.17g})"
            built.append((ship_type, scale, zone_type, int(cell_id), centroid_wkt, geom_wkt))
        cur.execute(
            "DELETE FROM cell_geom_full WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )
        if built:
            execute_values(
                cur,
                """
                INSERT INTO cell_geom_full (ship_type, scale, zone_type, cell_id, centroid, geom)
                VALUES %s
                """,
                built,
                template="(%s,%s,%s,%s,ST_SetSRID(ST_GeomFromText(%s,0),0),ST_SetSRID(ST_GeomFromText(%s,0),0))",
                page_size=batch_size,
            )
        conn.commit()
        print(f"cell_geom_full: inserted={len(built)}")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Build cell_geom_full PostGIS layer")
    ap.add_argument("--ship_type", default="JBC_615k")
    ap.add_argument("--scale", default=None)
    ap.add_argument("--zone_type", default="0_Fluid")
    args = ap.parse_args()
    ship, scale = parse_ship_type_and_scale(args.ship_type, args.scale)
    build_cell_geom_full(ship, scale, args.zone_type)


if __name__ == "__main__":
    main()
