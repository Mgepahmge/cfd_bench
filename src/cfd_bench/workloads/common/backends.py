"""Backend client factories for workloads."""

from __future__ import annotations

import os
from typing import Optional, Tuple

from cfd_bench.API.iotdb_api import IoTDBMeshClient
from cfd_bench.API.tiledb_api import TileDBMeshClient
from cfd_bench.API.vtk_api import VTKMeshClient
from cfd_bench.infra.tiledb.config import TileDBConfig
from cfd_bench.workloads.common.pg_cae_client import CaeSimulationPGClient
from cfd_bench.workloads.common.vtk_files import resolve_vtk_file


def parse_ship(ship: str) -> Tuple[str, str]:
    if "_" in ship:
        parts = ship.split("_", 1)
        return parts[0], parts[1]
    return ship, ""


def make_iotdb(ship: str, step: int, zone: str = "0_Fluid") -> IoTDBMeshClient:
    client = IoTDBMeshClient()
    client.connect(ship, step, zone)
    return client


def make_tiledb(ship: str, step: int, root: str, zone: str = "0_Fluid") -> TileDBMeshClient:
    client = TileDBMeshClient(TileDBConfig(root_path=root))
    client.connect(ship, step, zone)
    return client


def make_pg(ship: str, zone_type: str = "fluid") -> CaeSimulationPGClient:
    s, scale = parse_ship(ship)
    return CaeSimulationPGClient(ship=s, scale=scale, zone_type=zone_type)


def make_vtk(vtk_dir: str, ship: str, step: int, zone: str = "0_Fluid") -> VTKMeshClient:
    from cfd_bench.workloads.common.vtk_files import list_vtk_files

    files = list_vtk_files(vtk_dir)
    path = os.path.join(vtk_dir, resolve_vtk_file(files, ship, step))
    client = VTKMeshClient()
    client.connect(ship, step, zone, vtk_file=path)
    return client


def mesh_bounds_from_client(client) -> Optional[list]:
    if hasattr(client, "runtime"):
        data = client.runtime.ensure_cells(client.ctx.dataset_key, client.ctx.zone)
        from cfd_bench.workloads.common.bounds import bbox_from_cell_bboxes, flat_bounds

        gmin, gmax = bbox_from_cell_bboxes(data.cell_bbox)
        return flat_bounds(gmin, gmax)
    if hasattr(client, "vtk_mesh") and client.vtk_mesh is not None:
        b = client.vtk_mesh.GetBounds()
        return [b[0], b[1], b[2], b[3], b[4], b[5]]
    return None
