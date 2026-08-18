"""Backend client factories for workloads.

Backend libraries are imported lazily so a PostgreSQL-only benchmark does not
require IoTDB, TileDB or VTK to be installed.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from cfd_bench.workloads.common.vtk_files import resolve_vtk_file
from cfd_bench.core.observability import timed_stage


def parse_ship(ship: str) -> Tuple[str, str]:
    if "_" in ship:
        parts = ship.split("_", 1)
        return parts[0], parts[1]
    return ship, ""


def make_iotdb(ship: str, step: int, zone: str = "0_Fluid"):
    from cfd_bench.API.iotdb_api.client import IoTDBMeshClient

    client = IoTDBMeshClient()
    with timed_stage("IoTDB", f"connect dataset={ship} step={step} zone={zone}"):
        client.connect(ship, step, zone)
    return client


def make_tiledb(ship: str, step: int, root: str, zone: str = "0_Fluid"):
    from cfd_bench.API.tiledb_api.client import TileDBMeshClient
    from cfd_bench.infra.tiledb.config import TileDBConfig

    client = TileDBMeshClient(TileDBConfig(root_path=root))
    with timed_stage("TileDB", f"connect dataset={ship} step={step} zone={zone}"):
        client.connect(ship, step, zone)
    return client


def make_pg(ship: str, step: int = 200, zone: str = "0_Fluid"):
    from cfd_bench.API.postgresql_api.client import PostgreSQLMeshClient

    client = PostgreSQLMeshClient()
    with timed_stage("PostgreSQL", f"connect dataset={ship} step={step} zone={zone}"):
        client.connect(ship, step, zone=zone)
    return client


def make_vtk(vtk_dir: str, ship: str, step: int, zone: str = "0_Fluid"):
    from cfd_bench.API.vtk_api.client import VTKMeshClient
    from cfd_bench.workloads.common.vtk_files import list_vtk_files

    files = list_vtk_files(vtk_dir)
    path = os.path.join(vtk_dir, resolve_vtk_file(files, ship, step))
    client = VTKMeshClient()
    client.connect(ship, step, zone, vtk_file=path)
    return client


def mesh_bounds_from_client(client) -> Optional[list]:
    if hasattr(client, "get_mesh_bounds"):
        try:
            bounds = client.get_mesh_bounds()
            if bounds is not None:
                return list(bounds)
        except Exception:
            pass
    if hasattr(client, "runtime") and client.runtime is not None and hasattr(client, "ctx") and client.ctx is not None:
        try:
            data = client.runtime.ensure_cells(client.ctx.dataset_key, client.ctx.zone)
            from cfd_bench.workloads.common.bounds import bbox_from_cell_bboxes, flat_bounds

            gmin, gmax = bbox_from_cell_bboxes(data.cell_bbox)
            return flat_bounds(gmin, gmax)
        except TypeError:
            try:
                data = client.runtime.ensure_cells()
                from cfd_bench.workloads.common.bounds import bbox_from_cell_bboxes, flat_bounds

                gmin, gmax = bbox_from_cell_bboxes(data.cell_bbox)
                return flat_bounds(gmin, gmax)
            except Exception:
                pass
    if hasattr(client, "get_cell_count"):
        from cfd_bench.infra.postgresql import spatial as pg_spatial

        if hasattr(client, "_inner") and client._inner is not None and client._key is not None:
            b = pg_spatial.fetch_mesh_bounds(
                client._inner.conn, client._key.ship, client._key.scale, client._key.zone
            )
            if b:
                return b
    if hasattr(client, "vtk_mesh") and client.vtk_mesh is not None:
        b = client.vtk_mesh.GetBounds()
        return [b[0], b[1], b[2], b[3], b[4], b[5]]
    return None
