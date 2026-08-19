"""VTK backend ingest for canonical CFD and HDF5 benchmark data."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from cfd_bench.core.observability import timed_stage
from cfd_bench.infra.vtk.storage import (
    frame_path,
    relative_frame_path,
    reset_dataset,
    write_manifest,
)
from cfd_bench.ingest.vtk.mesh_builder import build_grid, write_vtu


def _cfd_node_xyz(topo: Mapping[str, object]) -> np.ndarray:
    nodes = topo["nodes"]
    return np.column_stack((nodes["x"], nodes["y"], nodes["z"])).astype(np.float64, copy=False)


def _primary_zone(topology: Mapping[str, dict]) -> str:
    return next((z for z in topology if "fluid" in z.lower()), next(iter(topology)))


def load_cfd_to_vtk(
    input_path: str,
    dataset_key: str,
    *,
    root: str,
    zone_indices: Sequence[int] = (0, 1),
    topology: Optional[Mapping[str, dict]] = None,
) -> None:
    """Write legacy Tecplot CFD data to the unified VTK backend layout.

    ``topology`` is normally the exact canonical payload already built by the
    orchestrator for the database backends, so VTK does not introduce another
    expensive full topology decode.
    """
    from cfd_bench.ingest.cfd.canonical import (
        iter_cfd_frames,
        load_cfd_topology,
        max_neighbor_diffs,
        validate_frame_topology,
    )

    topology = dict(topology or load_cfd_topology(input_path, zone_indices, show_progress=True))
    reset_dataset(root, dataset_key)
    primary = _primary_zone(topology)

    zone_meta: Dict[str, dict] = {}
    for zone_name, topo in topology.items():
        zone_meta[zone_name] = {
            "zone_type": str(topo.get("zone_type") or ""),
            "node_count": int(topo["node_count"]),
            "cell_count": int(topo["cell_count"]),
            "steps": [],
            "variables": [],
            "nodal_variables": [],
            "files": {},
            "max_diff": {},
        }

    var_sets: Dict[str, list] = {z: [] for z in topology}
    seen_steps = []
    for frame in iter_cfd_frames(input_path, zone_indices):
        validate_frame_topology(frame, topology)
        seen_steps.append(int(frame.step))
        for zone_frame in frame.zones:
            zone_name = zone_frame.zone_name
            topo = topology[zone_name]
            fields = {str(k).upper(): np.asarray(v, dtype=np.float64) for k, v in zone_frame.variables.items()}
            var_sets[zone_name].append(set(fields))
            zmeta = zone_meta[zone_name]
            zmeta["steps"].append(int(frame.step))
            zmeta["files"][str(int(frame.step))] = relative_frame_path(dataset_key, zone_name, frame.step)
            zmeta["max_diff"][str(int(frame.step))] = {
                str(k).upper(): float(v)
                for k, v in max_neighbor_diffs(topo, fields).items()
            }
            cell_nodes = topo["cell_nodes"]
            with timed_stage(
                "VTK ingest",
                f"write CFD dataset={dataset_key} zone={zone_name} step={frame.step} "
                f"cells={topo['cell_count']} vars={sorted(fields)}",
            ):
                grid = build_grid(
                    _cfd_node_xyz(topo),
                    cell_nodes,
                    cfd_zone_type=str(topo.get("zone_type") or "FEPolyhedron"),
                    source_node_ids=np.arange(1, int(topo["node_count"]) + 1, dtype=np.int64),
                    source_cell_ids=np.arange(1, int(topo["cell_count"]) + 1, dtype=np.int64),
                    cell_fields=fields,
                )
                write_vtu(grid, str(frame_path(root, dataset_key, zone_name, frame.step)))

    for zone_name, zmeta in zone_meta.items():
        sets = var_sets[zone_name]
        common = sorted(set.intersection(*sets)) if sets else []
        zmeta["steps"] = sorted(set(int(x) for x in zmeta["steps"]))
        zmeta["variables"] = common

    primary_meta = zone_meta[primary]
    manifest = {
        "schema_version": 2,
        "backend": "vtk",
        "dataset_key": str(dataset_key),
        "dataset_type": "cfd",
        "is_h5": False,
        "primary_zone": primary,
        "steps": sorted(set(seen_steps)),
        "variables": list(primary_meta["variables"]),
        "nodal_variables": [],
        "zones": zone_meta,
        "frames": {},
    }
    path = write_manifest(root, dataset_key, manifest)
    print(
        f"VTK dataset metadata: {dataset_key} type=cfd primary_zone={primary} "
        f"steps={manifest['steps']} vars={manifest['variables']} -> {path}"
    )


def _frame_metadata(frame) -> dict:
    info = frame.info
    return {
        "timestep": int(frame.timestep),
        "step_name": str(info.step_name),
        "step_index": int(info.step_index),
        "step_domain": None if info.step_domain is None else int(info.step_domain),
        "frame_name": str(info.frame_name),
        "frame_index": int(info.frame_index),
        "inc_or_mode": None if info.inc_or_mode is None else int(info.inc_or_mode),
        "time_or_frequency": None if info.time_or_frequency is None else float(info.time_or_frequency),
        "description": str(info.description or ""),
        "load_case": str(info.load_case or ""),
        "source_fields": dict(frame.source_fields),
    }


def write_h5_plan_to_vtk(
    dataset_key: str,
    zone_type: str,
    root: str,
    plan,
    mesh,
    frames,
) -> None:
    from cfd_bench.ingest.h5.artifacts import max_neighbor_diffs

    reset_dataset(root, dataset_key)
    files = {}
    diffs = {}
    for frame in frames:
        step = int(frame.timestep)
        files[str(step)] = relative_frame_path(dataset_key, zone_type, step)
        diffs[str(step)] = {
            str(name).upper(): float(max_neighbor_diffs(mesh, np.asarray(values, dtype=np.float64)))
            for name, values in frame.cell_scalars.items()
        }
        with timed_stage(
            "VTK ingest-h5",
            f"write dataset={dataset_key} zone={zone_type} step={step} "
            f"cells={mesh.cell_count} cell_vars={sorted(frame.cell_scalars)} "
            f"node_vars={sorted(frame.node_scalars)}",
        ):
            grid = build_grid(
                np.asarray(mesh.node_coordinates, dtype=np.float64),
                mesh.cell_node_ids,
                cell_element_types=mesh.cell_element_types,
                source_node_ids=np.asarray(mesh.source_node_labels, dtype=np.int64),
                source_cell_ids=np.asarray(mesh.source_cell_labels, dtype=np.int64),
                cell_fields=frame.cell_scalars,
                point_fields=frame.node_scalars,
            )
            write_vtu(grid, str(frame_path(root, dataset_key, zone_type, step)))

    common_cell = sorted(set.intersection(*(set(f.cell_scalars) for f in frames))) if frames else []
    common_node = sorted(set.intersection(*(set(f.node_scalars) for f in frames))) if frames else []
    zmeta = {
        "zone_type": "H5",
        "node_count": int(mesh.node_count),
        "cell_count": int(mesh.cell_count),
        "steps": [int(x) for x in plan.mapped_timesteps],
        "variables": common_cell,
        "nodal_variables": common_node,
        "frame_cell_variables": {
            str(int(f.timestep)): sorted(str(v).upper() for v in f.cell_scalars)
            for f in frames
        },
        "frame_node_variables": {
            str(int(f.timestep)): sorted(str(v).upper() for v in f.node_scalars)
            for f in frames
        },
        "files": files,
        "max_diff": diffs,
        "element_types": list(plan.element_types),
    }
    manifest = {
        "schema_version": 2,
        "backend": "vtk",
        "dataset_key": str(dataset_key),
        "dataset_type": "h5",
        "is_h5": True,
        "primary_zone": str(zone_type),
        "steps": [int(x) for x in plan.mapped_timesteps],
        "variables": common_cell,
        "nodal_variables": common_node,
        "part_name": str(mesh.part_name),
        "instance_name": str(mesh.instance_name),
        "zones": {str(zone_type): zmeta},
        "frames": {str(int(f.timestep)): _frame_metadata(f) for f in frames},
    }
    path = write_manifest(root, dataset_key, manifest)
    print(
        f"VTK H5 metadata: dataset={dataset_key} zone={zone_type} "
        f"steps={manifest['steps']} vars={common_cell} nodal_vars={common_node} -> {path}"
    )


def load_h5_to_vtk(
    h5_path: str,
    dataset_key: str,
    *,
    root: str,
    instance_name=None,
    zone_type: str = "0_Fluid",
    step_names=None,
    vector_field=None,
    scalar_fields=None,
    explicit_mapping=None,
    timestep_mode: str = "sequence",
    include_empty_frames: bool = False,
):
    from cfd_bench.ingest.h5.postgresql import build_ingest_plan

    plan, mesh, frames = build_ingest_plan(
        h5_path,
        instance_name=instance_name,
        step_names=step_names,
        vector_field=vector_field,
        scalar_fields=scalar_fields,
        explicit_mapping=explicit_mapping,
        timestep_mode=timestep_mode,
        include_empty_frames=include_empty_frames,
    )
    write_h5_plan_to_vtk(dataset_key, zone_type, root, plan, mesh, frames)
    return plan
