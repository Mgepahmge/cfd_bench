"""Load cell-centered variables from DAT directory into PostgreSQL cell_scalar."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

from cfd_bench.ingest.decoder import CAE_Decoder
from cfd_bench.ingest.postgresql.pg_io import parse_ship_type_and_scale, pg_connect


def _timestep_from_filename(name: str) -> int:
    stem = Path(name).stem
    return int(stem)


def load_cell_vars(
    input_path: str,
    ship_type: str,
    scale: str,
    zone_indices=None,
    db_name: str = "cae_data",
    db_user: str = "postgres",
    db_password: str = "123456",
    db_host: str = "localhost",
    db_port: str = "5432",
):
    zone_indices = zone_indices or [0, 1]
    conn = pg_connect(db_name, db_user, db_password, db_host, db_port)
    cur = conn.cursor()
    try:
        for file in sorted(os.listdir(input_path)):
            if not file.lower().endswith(".dat"):
                continue
            filepath = os.path.join(input_path, file)
            timestep = _timestep_from_filename(file)
            data = CAE_Decoder(3)
            data.Decode_dat_file(filepath)
            variables = [v for v in data.Variables[3:]]
            for zi in zone_indices:
                if zi >= len(data.Zones):
                    continue
                zone = data.Zones[zi]
                zone_type = getattr(zone, "Zone_name", f"Zone_{zi}").strip().replace(" ", "_") or f"Zone_{zi}"
                cur.execute(
                    """
                    DELETE FROM cell_scalar
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s AND timestep=%s
                    """,
                    (ship_type, scale, zone_type, timestep),
                )
                rows = []
                for vi, var in enumerate(variables):
                    vals = zone.Element_Variables[vi]
                    for cid, val in enumerate(tqdm(vals, desc=f"{file} {zone_type} {var}", leave=False)):
                        rows.append((ship_type, scale, zone_type, timestep, str(var).upper(), int(cid), float(val)))
                if rows:
                    execute_values(
                        cur,
                        """
                        INSERT INTO cell_scalar
                        (ship_type, scale, zone_type, timestep, var, cell_id, value)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                        """,
                        rows,
                        page_size=5000,
                    )
                conn.commit()
                print(f"Inserted {len(rows)} scalars: {ship_type}/{scale}/{zone_type} step={timestep}")
    finally:
        cur.close()
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Load cell variables into PostgreSQL cell_scalar")
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--ship_type", default="JBC")
    ap.add_argument("--scale", default="615k")
    ap.add_argument("--zone_indices", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    ship, scale = parse_ship_type_and_scale(args.ship_type, args.scale)
    load_cell_vars(args.input_path, ship, scale, args.zone_indices)


if __name__ == "__main__":
    main()
