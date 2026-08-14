"""Build PostGIS cell_geom_full from cell_nodes + node_coordinates."""

from __future__ import annotations

import argparse
from typing import Mapping, Optional, Tuple

import numpy as np


_C3D10_CORNER_FACE_SIMPLICES = np.asarray(
    [
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ],
    dtype=np.int64,
)


def _is_c3d10(element_type: Optional[str]) -> bool:
    return bool(element_type) and str(element_type).upper().startswith("C3D10")


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


def _surface_geometry(points: np.ndarray, element_type: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return surface points/simplices for one cell.

    C3D10 is a quadratic tetrahedron.  For the benchmark PostGIS shell we use
    its four corner nodes directly, which is equivalent to the old ConvexHull
    result for the usual straight-sided C3D10 geometry but avoids running
    Qhull tens of thousands of times.  Other element families retain the
    previous ConvexHull behavior.
    """
    arr = np.asarray(points, dtype=np.float64)
    if _is_c3d10(element_type):
        if arr.shape[0] < 10:
            raise ValueError(f"{element_type} requires at least 10 nodes, got {arr.shape[0]}")
        return arr[:4], _C3D10_CORNER_FACE_SIMPLICES

    if arr.shape[0] < 4:
        return arr, np.empty((0, 3), dtype=np.int64)
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise RuntimeError("需要 scipy: pip install scipy") from exc
    hull = ConvexHull(arr)
    return arr, np.asarray(hull.simplices, dtype=np.int64)


def _insert_geom_batch(cur, rows, batch_size: int) -> int:
    if not rows:
        return 0
    from psycopg2.extras import execute_values

    execute_values(
        cur,
        """
        INSERT INTO cell_geom_full (ship_type, scale, zone_type, cell_id, centroid, geom)
        VALUES %s
        """,
        rows,
        template="(%s,%s,%s,%s,ST_SetSRID(ST_GeomFromText(%s,0),0),ST_SetSRID(ST_GeomFromText(%s,0),0))",
        page_size=batch_size,
    )
    return len(rows)


def build_cell_geom_full(
    ship_type: str,
    scale: str,
    zone_type: str,
    batch_size: int = 500,
    *,
    element_types_by_cell: Optional[Mapping[int, str]] = None,
    db_name: str = "cae_data",
    db_user: str = "postgres",
    db_password: str = "123456",
    db_host: str = "localhost",
    db_port: str = "5432",
):
    from cfd_bench.ingest.postgresql.pg_io import pg_connect

    conn = pg_connect(db_name, db_user, db_password, db_host, db_port)
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
        cell_rows = cur.fetchall()
        cur.execute(
            "DELETE FROM cell_geom_full WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )

        pending = []
        inserted = 0
        skipped = 0
        c3d10_fast = 0
        total = len(cell_rows)
        progress_every = max(5000, batch_size * 10)

        for processed, (cell_id, node_ids, cx, cy, cz) in enumerate(cell_rows, start=1):
            pts = []
            for nid in node_ids or []:
                xyz = node_xyz.get(int(nid))
                if xyz is None:
                    pts = []
                    break
                pts.append(xyz)
            if len(pts) < 4:
                skipped += 1
                continue

            arr = np.asarray(pts, dtype=np.float64)
            element_type = None
            if element_types_by_cell is not None:
                element_type = element_types_by_cell.get(int(cell_id))
            try:
                geom_points, simplices = _surface_geometry(arr, element_type)
            except Exception:
                skipped += 1
                continue
            if simplices.size == 0:
                skipped += 1
                continue
            if _is_c3d10(element_type):
                c3d10_fast += 1

            geom_wkt = _multipolygon_z_wkt(geom_points, simplices)
            if cx is None:
                cx, cy, cz = np.mean(arr[:, 0]), np.mean(arr[:, 1]), np.mean(arr[:, 2])
            centroid_wkt = f"POINT Z ({float(cx):.17g} {float(cy):.17g} {float(cz):.17g})"
            pending.append((ship_type, scale, zone_type, int(cell_id), centroid_wkt, geom_wkt))

            if len(pending) >= batch_size:
                inserted += _insert_geom_batch(cur, pending, batch_size)
                pending.clear()

            if processed % progress_every == 0:
                print(
                    f"cell_geom_full: processed={processed}/{total} "
                    f"inserted={inserted} skipped={skipped} c3d10_fast={c3d10_fast}"
                )

        inserted += _insert_geom_batch(cur, pending, batch_size)
        conn.commit()
        print(
            f"cell_geom_full: inserted={inserted} skipped={skipped} "
            f"c3d10_fast={c3d10_fast}"
        )
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
    from cfd_bench.ingest.postgresql.pg_io import parse_ship_type_and_scale

    ship, scale = parse_ship_type_and_scale(args.ship_type, args.scale)
    build_cell_geom_full(ship, scale, args.zone_type)


if __name__ == "__main__":
    main()
