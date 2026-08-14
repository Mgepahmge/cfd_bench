"""IoTDB adapter for the canonical ODB-like HDF5 model.

The adapter deliberately mirrors the existing IoTDB layout used by W1-W8:
entity ids live in the IoTDB ``Time`` column and each HDF5 frame gets its own
``step_<n>`` device.  H5-specific source labels and frame metadata live in
separate devices so the legacy CFD IoTDB layout remains valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.ingest.h5.artifacts import max_neighbor_diffs
from cfd_bench.ingest.h5.model import CanonicalFrame, CanonicalMesh, H5IngestPlan
from cfd_bench.ingest.h5.postgresql import build_ingest_plan


@dataclass(frozen=True)
class IoTDBConnectionArgs:
    host: str = "127.0.0.1"
    port: str = "6667"
    user: str = "root"
    password: str = "root"
    root_path: str = "root.simulation_data"

    @classmethod
    def from_config(cls, config: Optional[IoTDBConfig] = None) -> "IoTDBConnectionArgs":
        cfg = config or IoTDBConfig()
        return cls(
            host=cfg.host,
            port=str(cfg.port),
            user=cfg.user,
            password=cfg.password,
            root_path=cfg.root_path,
        )

    def connect(self):
        try:
            from iotdb.Session import Session
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "IoTDB H5 ingest requires apache-iotdb. "
                "Install with: pip install 'cfd_bench[iotdb]'"
            ) from exc
        session = Session(self.host, self.port, self.user, self.password)
        session.open()
        return session


def _typed_tablet(
    session, device: str, times, measurements, types, rows, *, chunk_size: int = 10000
) -> None:
    """Insert mixed-type Tablets in bounded chunks."""
    if not rows:
        return
    from iotdb.table_session import Tablet

    times = list(times)
    rows = list(rows)
    if len(times) != len(rows):
        raise ValueError(f"Tablet {device}: times={len(times)} rows={len(rows)}")
    size = max(1, int(chunk_size))
    for start in range(0, len(rows), size):
        end = min(start + size, len(rows))
        tablet = Tablet(
            device,
            list(measurements),
            list(types),
            [list(row) for row in rows[start:end]],
            [int(x) for x in times[start:end]],
        )
        session.insert_tablet(tablet)


def _types():
    from iotdb.utils.Field import TSDataType

    return TSDataType


def _safe_delete_timeseries(session, pattern: str) -> None:
    """Best-effort cleanup for a dataset-specific subtree before re-ingest."""
    try:
        session.execute_non_query_statement(f"DELETE TIMESERIES {pattern}")
    except Exception:
        # A missing subtree is normal on first ingest.  Do not hide insert errors;
        # only cleanup is intentionally best-effort for IoTDB version tolerance.
        pass




def _cell_vars_leaf(zone: str) -> str:
    z = str(zone or "0_Fluid").strip().lower()
    if "hull" in z or "wall" in z or z in {"1_hull", "hull"}:
        return "cell_vars_hull"
    return "cell_vars"


def _paths(root: str, dataset: str, zone: str) -> dict:
    return {
        "mesh": f"{root}.mesh_static.{dataset}.{zone}",
        "post": f"{root}.post_processing_management.{dataset}",
        "meta": f"{root}.h5_metadata.{dataset}",
        "derived": f"{root}.derived.{dataset}",
    }


def _clean_dataset(session, root: str, dataset: str, zone: str) -> None:
    p = _paths(root, dataset, zone)
    for pattern in (
        f"{p['mesh']}.**",
        f"{p['post']}.**",
        f"{p['meta']}.**",
        f"{p['derived']}.**",
    ):
        _safe_delete_timeseries(session, pattern)


def _mesh_arrays(mesh: CanonicalMesh):
    node_counts = [len(x) for x in mesh.cell_node_ids]
    neighbor_counts = [len(x) for x in mesh.cell_adjacency]
    max_nodes = max(node_counts, default=0)
    max_neighbors = max(neighbor_counts, default=0)
    if max_nodes > 64:
        raise ValueError(
            f"IoTDB H5 adapter currently supports <=64 nodes per element; found {max_nodes}"
        )
    if max_neighbors > 64:
        raise ValueError(
            f"IoTDB H5 adapter currently supports <=64 neighbors per element; found {max_neighbors}"
        )

    bboxes = np.empty((mesh.cell_count, 6), dtype=np.float64)
    for cid, node_ids in enumerate(mesh.cell_node_ids):
        pts = mesh.node_coordinates[np.asarray(node_ids, dtype=np.int64)]
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        bboxes[cid] = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]
    return max_nodes, max_neighbors, bboxes


def _insert_mesh(session, root: str, dataset: str, zone: str, mesh: CanonicalMesh) -> None:
    T = _types()
    p = _paths(root, dataset, zone)
    base = p["mesh"]
    max_nodes, max_neighbors, bboxes = _mesh_arrays(mesh)

    # Dense benchmark node ids are the Time values.
    _typed_tablet(
        session,
        f"{base}.nodes",
        mesh.node_ids,
        ["x", "y", "z"],
        [T.DOUBLE, T.DOUBLE, T.DOUBLE],
        mesh.node_coordinates.tolist(),
    )

    cell_rows = []
    for cid in range(mesh.cell_count):
        c = mesh.cell_centroids[cid]
        bb = bboxes[cid]
        # cell_type stays numeric for compatibility with the legacy runtime;
        # source element type is preserved losslessly in cell_source.element_type.
        cell_rows.append(
            [
                float(c[0]), float(c[1]), float(c[2]),
                float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                float(bb[4]), float(bb[5]), int(len(mesh.cell_node_ids[cid])),
            ]
        )
    _typed_tablet(
        session,
        f"{base}.cells",
        mesh.cell_ids,
        ["cx", "cy", "cz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cell_type"],
        [T.DOUBLE] * 9 + [T.INT32],
        cell_rows,
    )

    if max_nodes:
        node_measurements = [f"node_id_{i}" for i in range(max_nodes)]
        node_rows = [
            [int(nodes[i]) if i < len(nodes) else -1 for i in range(max_nodes)]
            for nodes in mesh.cell_node_ids
        ]
        _typed_tablet(
            session,
            f"{base}.cell_nodes",
            mesh.cell_ids,
            node_measurements,
            [T.INT64] * max_nodes,
            node_rows,
        )

    if max_neighbors:
        adj_measurements = [f"neighbor_id_{i}" for i in range(max_neighbors)]
        adj_rows = [
            [int(nbrs[i]) if i < len(nbrs) else -1 for i in range(max_neighbors)]
            for nbrs in mesh.cell_adjacency
        ]
        _typed_tablet(
            session,
            f"{base}.cell_adjacency",
            mesh.cell_ids,
            adj_measurements,
            [T.INT64] * max_neighbors,
            adj_rows,
        )

    if mesh.node_count:
        mins = np.min(mesh.node_coordinates, axis=0)
        maxs = np.max(mesh.node_coordinates, axis=0)
    else:
        mins = maxs = np.zeros(3, dtype=np.float64)
    _typed_tablet(
        session,
        f"{base}.mesh_meta",
        [0],
        [
            "node_count", "cell_count", "face_count",
            "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y", "bbox_min_z", "bbox_max_z",
            "max_nodes_per_cell", "max_neighbors_per_cell",
        ],
        [T.INT64, T.INT64, T.INT64] + [T.DOUBLE] * 6 + [T.INT32, T.INT32],
        [[
            int(mesh.node_count), int(mesh.cell_count), 0,
            float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]),
            int(max_nodes), int(max_neighbors),
        ]],
    )

    # H5 source labels are kept separate from dense benchmark ids.
    _typed_tablet(
        session,
        f"{base}.node_source",
        mesh.node_ids,
        ["source_label"],
        [T.INT64],
        [[int(x)] for x in mesh.source_node_labels],
    )
    _typed_tablet(
        session,
        f"{base}.cell_source",
        mesh.cell_ids,
        ["source_label", "element_type"],
        [T.INT64, T.TEXT],
        [
            [int(label), str(element_type)]
            for label, element_type in zip(mesh.source_cell_labels, mesh.cell_element_types)
        ],
    )


def _insert_frames(
    session,
    root: str,
    dataset: str,
    zone: str,
    mesh: CanonicalMesh,
    frames: Sequence[CanonicalFrame],
    plan: H5IngestPlan,
) -> None:
    T = _types()
    p = _paths(root, dataset, zone)

    # One known metadata device makes backend discovery independent of HDF5 layout.
    common_cell = sorted(set.intersection(
        *(set(frame.cell_scalars) for frame in frames)
    )) if frames else []
    common_node = sorted(set.intersection(
        *(set(frame.node_scalars) for frame in frames)
    )) if frames else []
    _typed_tablet(
        session,
        f"{p['meta']}.dataset_meta",
        [0],
        [
            "is_h5", "zone", "part_name", "instance_name", "variables_csv",
            "nodal_variables_csv", "common_variables_csv", "common_nodal_variables_csv",
            "element_types_csv", "node_count", "cell_count",
        ],
        [
            T.BOOLEAN, T.TEXT, T.TEXT, T.TEXT, T.TEXT, T.TEXT, T.TEXT, T.TEXT,
            T.TEXT, T.INT64, T.INT64,
        ],
        [[
            True,
            str(zone),
            str(mesh.part_name),
            str(mesh.instance_name),
            ",".join(plan.mapped_variables),
            ",".join(plan.mapped_node_variables),
            ",".join(common_cell),
            ",".join(common_node),
            ",".join(plan.element_types),
            int(mesh.node_count),
            int(mesh.cell_count),
        ]],
    )

    frame_rows = []
    for frame in frames:
        info = frame.info
        frame_rows.append([
            str(info.step_name), int(info.step_index),
            int(info.step_domain) if info.step_domain is not None else -1,
            str(info.frame_name), int(info.frame_index),
            int(info.inc_or_mode) if info.inc_or_mode is not None else -1,
            float(info.time_or_frequency) if info.time_or_frequency is not None else float("nan"),
            str(info.description or ""), str(info.load_case or ""),
        ])
    _typed_tablet(
        session,
        f"{p['meta']}.frames",
        [int(frame.timestep) for frame in frames],
        [
            "step_name", "step_index", "step_domain", "frame_name", "frame_index",
            "inc_or_mode", "time_or_frequency", "description", "load_case",
        ],
        [T.TEXT, T.INT32, T.INT32, T.TEXT, T.INT32, T.INT32, T.DOUBLE, T.TEXT, T.TEXT],
        frame_rows,
    )

    for frame in frames:
        step_base = f"{p['post']}.step_{int(frame.timestep)}"
        if frame.cell_scalars:
            vars_ = sorted(frame.cell_scalars)
            rows = np.column_stack(
                [np.asarray(frame.cell_scalars[v], dtype=np.float64) for v in vars_]
            ).tolist()
            _typed_tablet(
                session,
                f"{step_base}.{_cell_vars_leaf(zone)}",
                mesh.cell_ids,
                vars_,
                [T.DOUBLE] * len(vars_),
                rows,
            )
        if frame.node_scalars:
            vars_ = sorted(frame.node_scalars)
            rows = np.column_stack(
                [np.asarray(frame.node_scalars[v], dtype=np.float64) for v in vars_]
            ).tolist()
            _typed_tablet(
                session,
                f"{step_base}.node_vars",
                mesh.node_ids,
                vars_,
                [T.DOUBLE] * len(vars_),
                rows,
            )

        if frame.cell_scalars:
            vars_ = sorted(frame.cell_scalars)
            values = [
                float(max_neighbor_diffs(mesh, np.asarray(frame.cell_scalars[v], dtype=np.float64)))
                for v in vars_
            ]
            _typed_tablet(
                session,
                f"{p['derived']}.step_{int(frame.timestep)}.max_diff",
                [0],
                vars_,
                [T.DOUBLE] * len(vars_),
                [values],
            )


def load_h5_to_iotdb(
    h5_path: str,
    dataset_key: str,
    *,
    instance_name: Optional[str] = None,
    zone_type: str = "0_Fluid",
    step_names: Optional[Sequence[str]] = None,
    vector_field: Optional[str] = None,
    scalar_fields: Optional[Sequence[str]] = None,
    explicit_mapping: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
    timestep_mode: str = "sequence",
    include_empty_frames: bool = False,
    connection: Optional[IoTDBConnectionArgs] = None,
) -> H5IngestPlan:
    """Parse one HDF5 result and load the IoTDB tree-model benchmark layout."""
    connection = connection or IoTDBConnectionArgs.from_config()
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
    session = connection.connect()
    try:
        print(f"[ingest-h5][iotdb] reset dataset={dataset_key} zone={zone_type}")
        _clean_dataset(session, connection.root_path, dataset_key, zone_type)
        print(
            f"[ingest-h5][iotdb] mesh nodes={mesh.node_count} cells={mesh.cell_count} "
            f"types={sorted(set(mesh.cell_element_types))}"
        )
        _insert_mesh(session, connection.root_path, dataset_key, zone_type, mesh)
        print(f"[ingest-h5][iotdb] frames={len(frames)} vars={list(plan.mapped_variables)}")
        _insert_frames(
            session,
            connection.root_path,
            dataset_key,
            zone_type,
            mesh,
            frames,
            plan,
        )
    finally:
        session.close()
    return plan
