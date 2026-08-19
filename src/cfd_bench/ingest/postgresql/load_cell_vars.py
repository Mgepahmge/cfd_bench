"""Load legacy CFD cell-centred fields into PostgreSQL.

The DAT parser/topology contract is shared with IoTDB/TileDB.  Static topology
is decoded once; subsequent result files parse only their block values.
"""

from __future__ import annotations

import argparse
from typing import Mapping, Sequence

from psycopg2.extras import execute_values

from cfd_bench.ingest.cfd.canonical import (
    iter_cfd_frames,
    load_cfd_topology,
    max_neighbor_diffs,
    validate_frame_topology,
)
from cfd_bench.ingest.postgresql.pg_io import parse_ship_type_and_scale, pg_connect


def _insert_scalar_array(cur, ship, scale, zone, step, var, values, chunk_size=50000):
    n = len(values)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        execute_values(
            cur,
            """
            INSERT INTO cell_scalar
            (ship_type,scale,zone_type,timestep,var,cell_id,value) VALUES %s
            """,
            [
                (ship, scale, zone, int(step), str(var).upper(), cid, float(values[cid]))
                for cid in range(start, end)
            ],
            page_size=10000,
        )


def load_cell_vars(
    input_path: str,
    ship_type: str,
    scale: str,
    zone_indices: Sequence[int] | None = None,
    db_name: str = "cae_data",
    db_user: str = "postgres",
    db_password: str = "123456",
    db_host: str = "localhost",
    db_port: str = "5432",
    topology=None,
):
    zone_indices = list(zone_indices or [0, 1])
    topology = topology if topology is not None else load_cfd_topology(input_path, zone_indices)
    conn = pg_connect(db_name, db_user, db_password, db_host, db_port)
    cur = conn.cursor()
    try:
        for frame in iter_cfd_frames(input_path, zone_indices):
            validate_frame_topology(frame, topology)
            for zone_frame in frame.zones:
                zone = zone_frame.zone_name
                cur.execute(
                    "DELETE FROM cell_scalar WHERE ship_type=%s AND scale=%s AND zone_type=%s AND timestep=%s",
                    (ship_type, scale, zone, int(frame.step)),
                )
                cur.execute(
                    "DELETE FROM benchmark_max_diff WHERE ship_type=%s AND scale=%s AND zone_type=%s AND timestep=%s",
                    (ship_type, scale, zone, int(frame.step)),
                )
                for var, values in zone_frame.variables.items():
                    _insert_scalar_array(cur, ship_type, scale, zone, frame.step, var, values)

                diffs = max_neighbor_diffs(topology[zone], zone_frame.variables)
                if diffs:
                    execute_values(
                        cur,
                        """
                        INSERT INTO benchmark_max_diff
                        (ship_type,scale,zone_type,timestep,var,max_diff) VALUES %s
                        """,
                        [
                            (ship_type, scale, zone, int(frame.step), str(var).upper(), float(delta))
                            for var, delta in diffs.items()
                        ],
                    )
                conn.commit()
                print(
                    f"PostgreSQL cell vars: {ship_type}_{scale} zone={zone} "
                    f"step={frame.step} cells={zone_frame.cell_count} vars={list(zone_frame.variables)}"
                )
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Load CFD cell variables into PostgreSQL")
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    ship, scale = parse_ship_type_and_scale(args.ship_type, args.scale)
    load_cell_vars(args.input_path, ship, scale, args.zone_indices)


if __name__ == "__main__":
    main()
