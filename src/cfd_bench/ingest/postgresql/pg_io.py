"""PostgreSQL mesh topology write helpers."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import psycopg2
from tqdm import tqdm

from cfd_bench.ingest.decoder import CAE_Decoder, Zone_3D


def pg_connect(
    db_name: str = "cae_data",
    db_user: str = "postgres",
    db_password: str = "123456",
    db_host: str = "localhost",
    db_port: str = "5432",
):
    return psycopg2.connect(
        database=db_name, user=db_user, password=db_password, host=db_host, port=db_port
    )


def parse_ship_type_and_scale(ship_type_arg: str, scale_arg=None) -> Tuple[str, str]:
    if "_" in ship_type_arg:
        st, sc = ship_type_arg.split("_", 1)
    else:
        st, sc = ship_type_arg, scale_arg or "default"
    return st, (scale_arg if scale_arg else sc)


def _batch_insert(cursor, table, columns, rows, batch_size=5000):
    if not rows:
        return
    n = len(columns)
    cols_str = ",".join(columns)
    ph = "(" + ",".join(["%s"] * n) + ")"
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        placeholders = ",".join([ph] * len(chunk))
        sql = f"INSERT INTO {table} ({cols_str}) VALUES {placeholders} ON CONFLICT DO NOTHING"
        flat = [x for r in chunk for x in (r if isinstance(r, (list, tuple)) else (r,))]
        cursor.execute(sql, flat)


def _batch_insert_adjacency(cursor, ship_type, scale, zone_type, cell_ids, neighbor_id_lists, batch_size=2000):
    rows = []
    for cid, nbrs in zip(cell_ids, neighbor_id_lists):
        clean = [int(x) for x in nbrs if x is not None and int(x) >= 0]
        if not clean:
            clean = [0]
        rows.append((ship_type, scale, zone_type, int(cid), clean))
    _batch_insert(
        cursor,
        "cell_adjacency",
        ["ship_type", "scale", "zone_type", "cell_id", "neighbor_ids"],
        rows,
        batch_size=batch_size,
    )


def _face_plane_for_zone(zone: Zone_3D):
    X, Y, Z = zone.Node_Coordinates[0], zone.Node_Coordinates[1], zone.Node_Coordinates[2]
    EcX, EcY, EcZ = zone.Element_Coordinates[0], zone.Element_Coordinates[1], zone.Element_Coordinates[2]
    LE, RE, FN = zone.LE, zone.RE, zone.FN
    out = []
    for f in tqdm(range(zone.Face_count), desc="Face planes"):
        le, re = int(LE[f]), int(RE[f])
        if le < 0 or re < 0:
            continue
        node_ids = FN[f]
        if len(node_ids) < 3:
            continue
        pts = np.array([[float(X[n]), float(Y[n]), float(Z[n])] for n in node_ids])
        face_center = pts.mean(axis=0)
        v0, v1 = pts[1] - pts[0], pts[2] - pts[0]
        n = np.cross(v0, v1)
        nnorm = np.linalg.norm(n)
        if nnorm < 1e-15:
            continue
        n = n / nnorm
        le_centroid = np.array([EcX[le], EcY[le], EcZ[le]])
        re_centroid = np.array([EcX[re], EcY[re], EcZ[re]])
        if np.dot(n, re_centroid - le_centroid) < 0:
            n = -n
        d = -float(np.dot(n, face_center))
        out.append((le, re, float(n[0]), float(n[1]), float(n[2]), d))
        out.append((re, le, float(-n[0]), float(-n[1]), float(-n[2]), -d))
    return out


def _export_boundary_faces(
    cursor,
    zone: Zone_3D,
    ship_type: str,
    scale: str,
    zone_type: str,
):
    """
    Parse boundary faces from zone face connectivity and
    INSERT into boundary_face_geom table.

    A boundary face is one where exactly one of LE/RE is valid.
    Geometry is stored as PostGIS PolygonZ.
    """

    from psycopg2.extras import execute_values

    X = zone.Node_Coordinates[0]
    Y = zone.Node_Coordinates[1]
    Z = zone.Node_Coordinates[2]

    LE = zone.LE
    RE = zone.RE
    FN = zone.FN

    bf_rows = []

    for f in range(zone.Face_count):

        le = int(LE[f])
        re = int(RE[f])

        # Boundary face:
        # one valid adjacent cell and one invalid side
        is_boundary = (le >= 0) != (re >= 0)

        if not is_boundary:
            continue

        node_ids = FN[f]

        if len(node_ids) < 3:
            continue


        pts = np.array(
            [
                [
                    float(X[n]),
                    float(Y[n]),
                    float(Z[n]),
                ]
                for n in node_ids
            ],
            dtype=np.float64,
        )


        # Face center
        face_center = pts.mean(axis=0)


        # Compute normal
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]

        n = np.cross(v0, v1)

        nnorm = np.linalg.norm(n)

        if nnorm < 1e-15:
            continue


        n = n / nnorm


        # Triangle area approximation
        area = float(max(1e-12, nnorm * 0.5))


        # The valid adjacent cell
        cid = max(le, re)


        # IMPORTANT:
        # only pass WKT string here.
        # Do NOT pass:
        # ST_SetSRID(ST_GeomFromText(...))
        #
        # because psycopg2 will quote it as text.
        # Construct PolygonZ WKT
        #
        # PostGIS column:
        # geometry(PolygonZ,0)
        #
        # Therefore geometry must be:
        # POLYGON Z ((x y z, ...))

        coords = []

        for p in pts:
            coords.append(
                f"{float(p[0]):.17g} "
                f"{float(p[1]):.17g} "
                f"{float(p[2]):.17g}"
            )

        # close polygon ring
        coords.append(coords[0])

        geom_wkt = (
                "POLYGON Z (("
                + ", ".join(coords)
                + "))"
        )


        bf_rows.append(
            (
                ship_type,
                scale,
                zone_type,
                int(cid),
                area,
                float(n[0]),
                float(n[1]),
                float(n[2]),
                geom_wkt,
                "default",
            )
        )


    if not bf_rows:
        return


    # Remove previous data
    cursor.execute(
        """
        DELETE FROM boundary_face_geom
        WHERE ship_type=%s
          AND scale=%s
          AND zone_type=%s
        """,
        (
            ship_type,
            scale,
            zone_type,
        ),
    )


    # Correct PostGIS insertion
    execute_values(
        cursor,
        """
        INSERT INTO boundary_face_geom
        (
            ship_type,
            scale,
            zone_type,
            cell_id,
            area,
            nx,
            ny,
            nz,
            geom,
            patch_name
        )
        VALUES %s
        """,
        bf_rows,
        template=(
            "(%s,%s,%s,%s,%s,%s,%s,%s,"
            "ST_SetSRID(ST_GeomFromText(%s,0),0),"
            "%s)"
        ),
    )


def export_zone_to_pg(
    zone: Zone_3D,
    ship_type: str,
    scale: str,
    zone_type: str,
    conn,
    *,
    export_face_planes: bool = True,
    export_nodes: bool = True,
):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM cell_nodes WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )
        cursor.execute(
            """
            INSERT INTO mesh_metadata (ship_type, scale, zone_type, node_count, element_count, face_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ship_type, scale, zone_type) DO UPDATE SET
                node_count = EXCLUDED.node_count,
                element_count = EXCLUDED.element_count,
                face_count = EXCLUDED.face_count
            """,
            (ship_type, scale, zone_type, zone.Node_count, zone.Element_count, zone.Face_count),
        )
        adjacency = zone.construct_element_adjacency()
        cell_ids = list(range(zone.Element_count))
        _batch_insert_adjacency(cursor, ship_type, scale, zone_type, cell_ids, adjacency)
        cx, cy, cz = zone.Element_Coordinates[0], zone.Element_Coordinates[1], zone.Element_Coordinates[2]
        rows = [
            (ship_type, scale, zone_type, int(cid), float(xi), float(yi), float(zi))
            for cid, xi, yi, zi in zip(cell_ids, cx, cy, cz)
        ]
        _batch_insert(
            cursor,
            "cell_centroid",
            ["ship_type", "scale", "zone_type", "cell_id", "x", "y", "z"],
            rows,
        )
        if export_face_planes:
            face_rows = _face_plane_for_zone(zone)
            cols = ["ship_type", "scale", "zone_type", "cell_id", "neighbor_id", "nx", "ny", "nz", "d"]
            rows_fp = [
                (ship_type, scale, zone_type, int(cid), int(nid), float(nx), float(ny), float(nz), float(d))
                for cid, nid, nx, ny, nz, d in face_rows
            ]
            _batch_insert(cursor, "cell_face_plane", cols, rows_fp)
        en = zone.EN
        cn_rows = []
        if isinstance(en, dict):
            items = en.items()
        else:
            items = enumerate(en)
        for cid, node_list in items:
            if node_list is None:
                continue
            try:
                nodes = [int(n) for n in node_list]
            except TypeError:
                nodes = [int(node_list)]
            if nodes:
                cn_rows.append((ship_type, scale, zone_type, int(cid), nodes))
        _batch_insert(
            cursor,
            "cell_nodes",
            ["ship_type", "scale", "zone_type", "cell_id", "node_ids"],
            cn_rows,
        )
        if export_nodes:
            X, Y, Z = zone.Node_Coordinates[0], zone.Node_Coordinates[1], zone.Node_Coordinates[2]
            node_ids = list(range(zone.Node_count))
            rows_n = [
                (ship_type, scale, zone_type, int(nid), float(xi), float(yi), float(zi))
                for nid, xi, yi, zi in zip(node_ids, X, Y, Z)
            ]
            _batch_insert(
                cursor,
                "node_coordinates",
                ["ship_type", "scale", "zone_type", "node_id", "x", "y", "z"],
                rows_n,
            )

        # --- W6: Export boundary faces into boundary_face_geom table ---
        _export_boundary_faces(cursor, zone, ship_type, scale, zone_type)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def load_topology_from_dat(
    dat_path: str,
    ship_type: str,
    scale: str,
    zone_indices: Sequence[int] = (0, 1),
    conn=None,
    export_face_planes: bool = True,
    export_nodes: bool = True,
):
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(dat_path)
    data = CAE_Decoder(3)
    data.Decode_dat_file(dat_path)
    own = conn is None
    if own:
        conn = pg_connect()
    try:
        for zi in zone_indices:
            if zi >= len(data.Zones):
                continue
            zone = data.Zones[zi]
            zone_type = getattr(zone, "Zone_name", f"Zone_{zi}").strip().replace(" ", "_") or f"Zone_{zi}"
            export_zone_to_pg(
                zone, ship_type, scale, zone_type, conn,
                export_face_planes=export_face_planes,
                export_nodes=export_nodes,
            )
    finally:
        if own and conn:
            conn.close()
