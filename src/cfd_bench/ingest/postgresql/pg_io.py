"""PostgreSQL writers for the legacy CFD DAT path.

H5 ingest has its own frozen writer.  This module accepts the canonical CFD
payload produced by :mod:`cfd_bench.ingest.cfd.canonical` so all backends see
identical ids/topology.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values


def pg_connect(
    db_name: str = "cae_data",
    db_user: str = "postgres",
    db_password: str = "123456",
    db_host: str = "localhost",
    db_port: str = "5432",
):
    # Environment variables are intentionally honoured here too so direct
    # ingest scripts and workload clients use the same database by default.
    return psycopg2.connect(
        database=os.environ.get("CFD_BENCH_PG_DATABASE", db_name),
        user=os.environ.get("CFD_BENCH_PG_USER", db_user),
        password=os.environ.get("CFD_BENCH_PG_PASSWORD", db_password),
        host=os.environ.get("CFD_BENCH_PG_HOST", db_host),
        port=os.environ.get("CFD_BENCH_PG_PORT", db_port),
    )


def parse_ship_type_and_scale(ship_type_arg: str, scale_arg=None) -> Tuple[str, str]:
    if "_" in ship_type_arg:
        st, sc = ship_type_arg.split("_", 1)
    else:
        st, sc = ship_type_arg, scale_arg or "default"
    return st, (scale_arg if scale_arg else sc)


def _delete_static_zone(cursor, ship_type: str, scale: str, zone_type: str) -> None:
    # Re-ingest must be deterministic.  The old code used ON CONFLICT DO
    # NOTHING for most topology tables, leaving stale rows after a parser fix.
    for table in (
        "point_locator_grid",
        "cell_geom_full",
        "cell_bounds",
        "boundary_face_geom",
        "cell_face_plane",
        "cell_adjacency",
        "cell_centroid",
        "cell_nodes",
        "node_coordinates",
        "node_cells",
    ):
        cursor.execute(
            f"DELETE FROM {table} WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )


def _execute_indexed_chunks(cursor, sql: str, count: int, row_fn, *, chunk_size: int = 10000, page_size: int = 5000, template=None):
    """Feed ``execute_values`` bounded chunks instead of duplicating a huge mesh in Python lists."""
    for start in range(0, int(count), int(chunk_size)):
        end = min(start + int(chunk_size), int(count))
        rows = [row_fn(i) for i in range(start, end)]
        if rows:
            execute_values(cursor, sql, rows, page_size=page_size, template=template)


def _polygon_wkt(node_ids, nodes) -> str:
    x = nodes["x"]
    y = nodes["y"]
    z = nodes["z"]
    coords = [f"{float(x[n]):.17g} {float(y[n]):.17g} {float(z[n]):.17g}" for n in node_ids]
    if coords and coords[-1] != coords[0]:
        coords.append(coords[0])
    return "POLYGON Z ((" + ", ".join(coords) + "))"


def export_topology_payload_to_pg(
    topo: Mapping[str, object],
    ship_type: str,
    scale: str,
    conn,
    *,
    export_face_planes: bool = False,
    export_nodes: bool = True,
) -> None:
    zone_type = str(topo["zone_name"])
    cursor = conn.cursor()
    try:
        _delete_static_zone(cursor, ship_type, scale, zone_type)
        cursor.execute(
            """
            INSERT INTO mesh_metadata (ship_type, scale, zone_type, node_count, element_count, face_count)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ship_type,scale,zone_type) DO UPDATE SET
              node_count=EXCLUDED.node_count,
              element_count=EXCLUDED.element_count,
              face_count=EXCLUDED.face_count
            """,
            (
                ship_type,
                scale,
                zone_type,
                int(topo["node_count"]),
                int(topo["cell_count"]),
                int(topo["face_count"]),
            ),
        )

        cells = topo["cells"]
        _execute_indexed_chunks(
            cursor,
            "INSERT INTO cell_centroid(ship_type,scale,zone_type,cell_id,x,y,z) VALUES %s",
            len(cells),
            lambda cid: (ship_type, scale, zone_type, cid, float(cells[cid][0]), float(cells[cid][1]), float(cells[cid][2])),
            chunk_size=20000, page_size=10000,
        )
        _execute_indexed_chunks(
            cursor,
            """INSERT INTO cell_bounds
               (ship_type,scale,zone_type,cell_id,xmin,xmax,ymin,ymax,zmin,zmax) VALUES %s""",
            len(cells),
            lambda cid: (ship_type, scale, zone_type, cid, *[float(x) for x in cells[cid][3:9]]),
            chunk_size=20000, page_size=10000,
        )
        cell_nodes = topo["cell_nodes"]
        _execute_indexed_chunks(
            cursor,
            "INSERT INTO cell_nodes(ship_type,scale,zone_type,cell_id,node_ids) VALUES %s",
            len(cell_nodes),
            lambda cid: (ship_type, scale, zone_type, cid, [int(x) for x in cell_nodes[cid]]),
            chunk_size=10000, page_size=5000,
        )
        adjacency = topo["adjacency"]
        _execute_indexed_chunks(
            cursor,
            "INSERT INTO cell_adjacency(ship_type,scale,zone_type,cell_id,neighbor_ids) VALUES %s",
            len(adjacency),
            lambda cid: (ship_type, scale, zone_type, cid, [int(x) for x in adjacency[cid]]),
            chunk_size=10000, page_size=5000,
        )

        if export_nodes:
            nodes = topo["nodes"]
            _execute_indexed_chunks(
                cursor,
                "INSERT INTO node_coordinates(ship_type,scale,zone_type,node_id,x,y,z) VALUES %s",
                int(topo["node_count"]),
                lambda nid: (
                    ship_type, scale, zone_type, nid,
                    float(nodes["x"][nid]), float(nodes["y"][nid]), float(nodes["z"][nid]),
                ),
                chunk_size=20000, page_size=10000,
            )

        if export_face_planes and topo.get("face_planes"):
            face_planes = topo["face_planes"]
            _execute_indexed_chunks(
                cursor,
                """INSERT INTO cell_face_plane
                   (ship_type,scale,zone_type,cell_id,neighbor_id,nx,ny,nz,d) VALUES %s""",
                len(face_planes),
                lambda i: (ship_type, scale, zone_type, int(face_planes[i][0]), int(face_planes[i][1]), *map(float, face_planes[i][2:6])),
                chunk_size=20000, page_size=10000,
            )

        # Boundary faces are the W6 geometry contract.  Store the real face
        # polygon and outward normal from the canonical exporter.
        bf = topo.get("boundary_faces", ())
        bnodes = topo.get("boundary_face_nodes", ())
        if bf:
            nodes = topo["nodes"]
            _execute_indexed_chunks(
                cursor,
                """INSERT INTO boundary_face_geom
                   (ship_type,scale,zone_type,cell_id,area,nx,ny,nz,geom,patch_name) VALUES %s""",
                len(bf),
                lambda i: (
                    ship_type, scale, zone_type, int(bf[i][0]), float(bf[i][5]),
                    float(bf[i][2]), float(bf[i][3]), float(bf[i][4]),
                    _polygon_wkt(bnodes[i], nodes), "default",
                ),
                chunk_size=4000, page_size=2000,
                template=(
                    "(%s,%s,%s,%s,%s,%s,%s,%s,"
                    "ST_SetSRID(ST_GeomFromText(%s,0),0),%s)"
                ),
            )
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
    export_face_planes: bool = False,
    export_nodes: bool = True,
    topology=None,
):
    from cfd_bench.ingest.cfd.canonical import load_cfd_topology

    if topology is None:
        if not os.path.isfile(dat_path):
            raise FileNotFoundError(dat_path)
        payloads = load_cfd_topology(dat_path, zone_indices)
    else:
        payloads = topology
    own = conn is None
    if own:
        conn = pg_connect()
    try:
        for topo in payloads.values():
            export_topology_payload_to_pg(
                topo,
                ship_type,
                scale,
                conn,
                export_face_planes=export_face_planes,
                export_nodes=export_nodes,
            )
    finally:
        if own and conn:
            conn.close()
    return payloads
