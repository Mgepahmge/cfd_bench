"""Resolve geometry client and bounds for workloads (db vs vtk engine)."""

from __future__ import annotations

from typing import Any, List, Optional

from cfd_bench.workloads.common.backends import make_vtk, mesh_bounds_from_client
from cfd_bench.workloads.common.config import WorkloadConfig
from cfd_bench.core.observability import timed_stage


def uses_vtk_geom(cfg: WorkloadConfig) -> bool:
    return str(cfg.geom_engine).strip().lower() == "vtk"


def make_geom_client(
    cfg: WorkloadConfig,
    ship: str,
    step: int,
    data_client: Any,
    zone: str = "0_Fluid",
) -> Any:
    if uses_vtk_geom(cfg):
        vtk_dir = cfg.vtk_hull_dir if "hull" in zone.lower() or "wall" in zone.lower() else cfg.vtk_dir
        return make_vtk(vtk_dir, ship, step, zone=zone)
    return data_client


def mesh_bounds(
    cfg: WorkloadConfig,
    ship: str,
    step: int,
    data_client: Any,
    geom_client: Any,
    zone: str = "0_Fluid",
) -> Optional[List[float]]:
    label = type(data_client).__name__
    with timed_stage(label, f"resolve mesh bounds dataset={ship} step={step} zone={zone}"):
        if uses_vtk_geom(cfg):
            return mesh_bounds_from_client(geom_client)
        bounds = mesh_bounds_from_client(data_client)
        if bounds is not None:
            return bounds
        if geom_client is not data_client:
            return mesh_bounds_from_client(geom_client)
        return None


def cell_count(geom_client):
    """Get cell count without materializing the whole runtime mesh when possible."""
    if hasattr(geom_client, "get_cell_count"):
        try:
            return int(geom_client.get_cell_count())
        except Exception:
            pass
    if hasattr(geom_client, "runtime") and geom_client.runtime is not None:
        runtime = geom_client.runtime
        try:
            if hasattr(geom_client, "ctx") and geom_client.ctx is not None:
                try:
                    data = runtime.ensure_cells(geom_client.ctx.dataset_key, geom_client.ctx.zone)
                except TypeError:
                    data = runtime.ensure_cells()
                return len(data.cells)
        except Exception:
            pass
    if hasattr(geom_client, "vtk_mesh") and geom_client.vtk_mesh is not None:
        return int(geom_client.vtk_mesh.GetNumberOfCells())
    return 0


def random_var_range_db(client: Any, attribute_name: str, step: Optional[int] = None) -> tuple:
    """Sample random [lo, hi] for a variable using backend min/max if available."""
    if hasattr(client, "var_value_range"):
        vmin, vmax = client.var_value_range(attribute_name, step=step)
    else:
        vmin, vmax = 0.0, 1.0
    import random

    lo, hi = sorted([random.uniform(vmin, vmax), random.uniform(vmin, vmax)])
    return lo, hi
