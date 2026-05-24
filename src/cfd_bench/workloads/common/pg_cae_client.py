"""PostgreSQL cae_simulation_data client for workloads (legacy schema)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import psycopg2
from numpy.typing import NDArray


class CaeSimulationPGClient:
    """Matches legacy DB_Interface_PG against cae_simulation_data tables."""

    def __init__(
        self,
        ship: str,
        scale: str,
        zone_type: str = "fluid",
        db_name: str = "cae_data",
        db_user: str = "postgres",
        db_password: str = "123456",
        db_host: str = "localhost",
        db_port: str = "5432",
    ):
        self.ship = ship
        self.scale = scale
        self.zone_type = zone_type
        self.step: Optional[int] = None
        self.conn = psycopg2.connect(
            database=db_name, user=db_user, password=db_password, host=db_host, port=db_port
        )

    def set_step(self, step: int):
        self.step = int(step)

    def close(self):
        if self.conn:
            self.conn.close()

    def point_query(self, cell_indexes: Sequence[int], attribute_name: str) -> NDArray[np.float64]:
        if self.step is None:
            raise RuntimeError("call set_step() first")
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT data FROM cae_simulation_data
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND variable=%s AND is_element=TRUE
                """,
                (self.ship, self.scale, self.zone_type, self.step, attribute_name),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return np.full(len(cell_indexes), np.nan, dtype=np.float64)
            data = row[0]
            return np.array(
                [float(data[int(i)]) if 0 <= int(i) < len(data) else np.nan for i in cell_indexes],
                dtype=np.float64,
            )
        finally:
            cur.close()

    def range_query_var(self, lower: float, upper: float, attribute_name: str) -> NDArray[np.int32]:
        if self.step is None:
            raise RuntimeError("call set_step() first")
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT data FROM cae_simulation_data
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND variable=%s AND is_element=TRUE
                """,
                (self.ship, self.scale, self.zone_type, self.step, attribute_name),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return np.array([], dtype=np.int32)
            return np.array(
                [i for i, v in enumerate(row[0]) if lower <= float(v) <= upper], dtype=np.int32
            )
        finally:
            cur.close()

    def range_query_coord(
        self, lower_bound: Sequence[float], upper_bound: Sequence[float]
    ) -> NDArray[np.int32]:
        cur = self.conn.cursor()
        try:
            coords = {}
            for name in ("X", "Y", "Z"):
                cur.execute(
                    """
                    SELECT data FROM cae_simulation_data
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                      AND variable=%s AND is_element=TRUE LIMIT 1
                    """,
                    (self.ship, self.scale, self.zone_type, name),
                )
                row = cur.fetchone()
                if not row or not row[0]:
                    return np.array([], dtype=np.int32)
                coords[name] = row[0]
            x0, y0, z0 = map(float, lower_bound)
            x1, y1, z1 = map(float, upper_bound)
            out = []
            for idx in range(len(coords["X"])):
                x, y, z = float(coords["X"][idx]), float(coords["Y"][idx]), float(coords["Z"][idx])
                if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1:
                    out.append(idx)
            return np.array(out, dtype=np.int32)
        finally:
            cur.close()
