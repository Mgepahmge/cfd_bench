"""PostgreSQL adapter for the canonical HDF5 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import max_neighbor_diffs
from .canonical import BENCHMARK_SCALAR_VARS, available_mapping, build_canonical_frame
from .model import CanonicalFrame, CanonicalMesh, FrameInfo, H5IngestPlan
from .reader import OdbH5Reader
from cfd_bench.infra.postgresql.config import PostgreSQLConfig


def _parse_dataset_key(dataset_key: str) -> Tuple[str, str]:
    text = str(dataset_key).strip()
    if not text:
        raise ValueError("dataset key cannot be empty")
    if "_" in text:
        return tuple(text.split("_", 1))  # type: ignore[return-value]
    return text, ""


def _batch_insert(cur, table: str, columns: Sequence[str], rows, batch_size: int = 5000) -> None:
    rows = list(rows)
    if not rows:
        return
    try:
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL H5 ingest requires psycopg2. "
            "Install with: pip install 'cfd_bench[postgresql]'"
        ) from exc
    cols = ",".join(columns)
    execute_values(
        cur,
        f"INSERT INTO {table} ({cols}) VALUES %s ON CONFLICT DO NOTHING",
        rows,
        page_size=int(batch_size),
    )


@dataclass(frozen=True)
class PostgreSQLConnectionArgs:
    db_name: str = "cae_data"
    db_user: str = "postgres"
    db_password: str = "123456"
    db_host: str = "localhost"
    db_port: str = "5432"

    @classmethod
    def from_config(cls, config: Optional[PostgreSQLConfig] = None) -> "PostgreSQLConnectionArgs":
        cfg = config or PostgreSQLConfig()
        return cls(
            db_name=cfg.db_name,
            db_user=cfg.db_user,
            db_password=cfg.db_password,
            db_host=cfg.db_host,
            db_port=cfg.db_port,
        )

    def connect(self):
        try:
            import psycopg2
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL H5 ingest requires psycopg2. "
                "Install with: pip install 'cfd_bench[postgresql]'"
            ) from exc
        return psycopg2.connect(
            database=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
        )


def _timestep_for_frame(frame: FrameInfo, ordinal: int, mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode == "sequence":
        return int(ordinal)
    if mode == "frame-index":
        return int(frame.frame_index)
    if mode == "inc-mode":
        if frame.inc_or_mode is None:
            raise ValueError(
                f"{frame.step_name}/{frame.frame_name} has no Inc/Mode attribute"
            )
        return int(frame.inc_or_mode)
    raise ValueError(f"unknown timestep mode {mode!r}; use sequence, frame-index or inc-mode")


def _mapped_frames(
    reader: OdbH5Reader,
    mesh: CanonicalMesh,
    *,
    step_names: Optional[Sequence[str]],
    vector_field: Optional[str],
    scalar_fields: Optional[Sequence[str]],
    explicit_mapping: Optional[Mapping[str, Tuple[str, Optional[str]]]],
    timestep_mode: str,
    include_empty_frames: bool,
) -> Tuple[List[CanonicalFrame], List[str]]:
    frames: List[CanonicalFrame] = []
    skipped: List[str] = []
    mapped_ordinal = 0
    for frame in reader.iter_frames(step_names=step_names):
        mapping = available_mapping(
            reader,
            mesh,
            frame,
            vector_field=vector_field,
            scalar_fields=scalar_fields,
        )
        if explicit_mapping:
            # Explicit entries only override/add exceptional mappings.  The
            # automatically inferred mappings remain available.
            infos = reader.field_info(frame)
            fields_by_upper = {name.upper(): name for name in infos}
            for target, source in explicit_mapping.items():
                source_name = fields_by_upper.get(str(source[0]).upper())
                if source_name and mesh.instance_name in infos[source_name].instances:
                    mapping[str(target).upper()] = (source_name, source[1])
        if not mapping and not include_empty_frames:
            skipped.append(f"{frame.step_name}/{frame.frame_name}")
            continue
        timestep = _timestep_for_frame(frame, mapped_ordinal, timestep_mode)
        canonical = build_canonical_frame(
            reader,
            mesh,
            frame,
            timestep,
            vector_field=vector_field,
            scalar_fields=scalar_fields,
            explicit_mapping=mapping,
        )
        if not canonical.cell_scalars and not include_empty_frames:
            skipped.append(f"{frame.step_name}/{frame.frame_name}")
            continue
        frames.append(canonical)
        mapped_ordinal += 1
    return frames, skipped


def build_ingest_plan(
    h5_path: str,
    *,
    instance_name: Optional[str] = None,
    step_names: Optional[Sequence[str]] = None,
    vector_field: Optional[str] = None,
    scalar_fields: Optional[Sequence[str]] = None,
    explicit_mapping: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
    timestep_mode: str = "sequence",
    include_empty_frames: bool = False,
) -> Tuple[H5IngestPlan, CanonicalMesh, List[CanonicalFrame]]:
    reader = OdbH5Reader(h5_path)
    mesh = reader.load_mesh(instance_name)
    frames, skipped = _mapped_frames(
        reader,
        mesh,
        step_names=step_names,
        vector_field=vector_field,
        scalar_fields=scalar_fields,
        explicit_mapping=explicit_mapping,
        timestep_mode=timestep_mode,
        include_empty_frames=include_empty_frames,
    )
    mapped_variables = sorted({name for frame in frames for name in frame.cell_scalars})
    mapped_node_variables = sorted({name for frame in frames for name in frame.node_scalars})
    plan = H5IngestPlan(
        h5_path=str(h5_path),
        part_name=mesh.part_name,
        instance_name=mesh.instance_name,
        node_count=mesh.node_count,
        cell_count=mesh.cell_count,
        element_types=tuple(sorted(set(mesh.cell_element_types))),
        frame_count=len(frames),
        mapped_timesteps=tuple(frame.timestep for frame in frames),
        mapped_variables=tuple(mapped_variables),
        mapped_node_variables=tuple(mapped_node_variables),
        skipped_frames=tuple(skipped),
    )
    return plan, mesh, frames


def _delete_zone_data(cur, ship_type: str, scale: str, zone_type: str) -> None:
    # Order is deliberately explicit; these tables have no foreign keys today,
    # but clearing all derived data prevents stale rows after re-ingest.
    for table in (
        "boundary_face_geom",
        "point_locator_grid",
        "cell_geom_full",
        "node_scalar",
        "node_cells",
        "benchmark_max_diff",
        "cell_scalar",
        "cell_face_plane",
        "cell_adjacency",
        "cell_centroid",
        "cell_nodes",
        "node_coordinates",
        "h5_frame_metadata",
        "h5_cell_source",
        "h5_node_source",
    ):
        cur.execute(
            f"DELETE FROM {table} WHERE ship_type=%s AND scale=%s AND zone_type=%s",
            (ship_type, scale, zone_type),
        )


def _insert_mesh(
    cur,
    mesh: CanonicalMesh,
    ship_type: str,
    scale: str,
    zone_type: str,
) -> None:
    cur.execute(
        """
        INSERT INTO mesh_metadata
        (ship_type, scale, zone_type, node_count, element_count, face_count)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (ship_type, scale, zone_type) DO UPDATE SET
            node_count=EXCLUDED.node_count,
            element_count=EXCLUDED.element_count,
            face_count=EXCLUDED.face_count
        """,
        (ship_type, scale, zone_type, mesh.node_count, mesh.cell_count, 0),
    )

    node_rows = [
        (
            ship_type,
            scale,
            zone_type,
            int(nid),
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        )
        for nid, xyz in zip(mesh.node_ids, mesh.node_coordinates)
    ]
    _batch_insert(
        cur,
        "node_coordinates",
        ["ship_type", "scale", "zone_type", "node_id", "x", "y", "z"],
        node_rows,
    )

    cell_node_rows = [
        (ship_type, scale, zone_type, int(cid), [int(x) for x in nodes])
        for cid, nodes in zip(mesh.cell_ids, mesh.cell_node_ids)
    ]
    _batch_insert(
        cur,
        "cell_nodes",
        ["ship_type", "scale", "zone_type", "cell_id", "node_ids"],
        cell_node_rows,
    )

    centroid_rows = [
        (
            ship_type,
            scale,
            zone_type,
            int(cid),
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        )
        for cid, xyz in zip(mesh.cell_ids, mesh.cell_centroids)
    ]
    _batch_insert(
        cur,
        "cell_centroid",
        ["ship_type", "scale", "zone_type", "cell_id", "x", "y", "z"],
        centroid_rows,
    )

    adjacency_rows = [
        (ship_type, scale, zone_type, int(cid), [int(x) for x in nbrs])
        for cid, nbrs in zip(mesh.cell_ids, mesh.cell_adjacency)
    ]
    _batch_insert(
        cur,
        "cell_adjacency",
        ["ship_type", "scale", "zone_type", "cell_id", "neighbor_ids"],
        adjacency_rows,
    )

    source_node_rows = [
        (
            ship_type,
            scale,
            zone_type,
            mesh.part_name,
            mesh.instance_name,
            int(nid),
            int(label),
        )
        for nid, label in zip(mesh.node_ids, mesh.source_node_labels)
    ]
    _batch_insert(
        cur,
        "h5_node_source",
        [
            "ship_type",
            "scale",
            "zone_type",
            "part_name",
            "instance_name",
            "node_id",
            "source_node_label",
        ],
        source_node_rows,
    )

    source_cell_rows = [
        (
            ship_type,
            scale,
            zone_type,
            mesh.part_name,
            mesh.instance_name,
            int(cid),
            int(label),
            str(element_type),
        )
        for cid, label, element_type in zip(
            mesh.cell_ids, mesh.source_cell_labels, mesh.cell_element_types
        )
    ]
    _batch_insert(
        cur,
        "h5_cell_source",
        [
            "ship_type",
            "scale",
            "zone_type",
            "part_name",
            "instance_name",
            "cell_id",
            "source_element_label",
            "element_type",
        ],
        source_cell_rows,
    )


def _insert_frames(
    cur,
    frames: Sequence[CanonicalFrame],
    ship_type: str,
    scale: str,
    zone_type: str,
) -> None:
    frame_rows = []
    scalar_rows = []
    node_scalar_rows = []
    for frame in frames:
        info = frame.info
        frame_rows.append(
            (
                ship_type,
                scale,
                zone_type,
                int(frame.timestep),
                info.step_name,
                int(info.step_index),
                info.step_domain,
                info.frame_name,
                int(info.frame_index),
                info.inc_or_mode,
                info.time_or_frequency,
                info.description,
                info.load_case,
            )
        )
        for var, values in frame.cell_scalars.items():
            for cid, value in enumerate(np.asarray(values, dtype=np.float64)):
                if not np.isfinite(value):
                    continue
                scalar_rows.append(
                    (
                        ship_type,
                        scale,
                        zone_type,
                        int(frame.timestep),
                        str(var).upper(),
                        int(cid),
                        float(value),
                    )
                )
        for var, values in frame.node_scalars.items():
            for nid, value in enumerate(np.asarray(values, dtype=np.float64)):
                if not np.isfinite(value):
                    continue
                node_scalar_rows.append(
                    (
                        ship_type,
                        scale,
                        zone_type,
                        int(frame.timestep),
                        str(var).upper(),
                        int(nid),
                        float(value),
                    )
                )

    _batch_insert(
        cur,
        "h5_frame_metadata",
        [
            "ship_type",
            "scale",
            "zone_type",
            "timestep",
            "step_name",
            "step_index",
            "step_domain",
            "frame_name",
            "frame_index",
            "inc_or_mode",
            "time_or_frequency",
            "description",
            "load_case",
        ],
        frame_rows,
        batch_size=1000,
    )
    _batch_insert(
        cur,
        "cell_scalar",
        ["ship_type", "scale", "zone_type", "timestep", "var", "cell_id", "value"],
        scalar_rows,
        batch_size=5000,
    )
    _batch_insert(
        cur,
        "node_scalar",
        ["ship_type", "scale", "zone_type", "timestep", "var", "node_id", "value"],
        node_scalar_rows,
        batch_size=5000,
    )


def _max_diff_rows(
    mesh: CanonicalMesh,
    frames: Sequence[CanonicalFrame],
    ship_type: str,
    scale: str,
    zone_type: str,
):
    for frame in frames:
        for var, values in sorted(frame.cell_scalars.items()):
            yield (
                ship_type,
                scale,
                zone_type,
                int(frame.timestep),
                str(var).upper(),
                float(max_neighbor_diffs(mesh, values)),
            )


def _insert_max_diffs(
    cur,
    mesh: CanonicalMesh,
    frames: Sequence[CanonicalFrame],
    ship_type: str,
    scale: str,
    zone_type: str,
) -> None:
    _batch_insert(
        cur,
        "benchmark_max_diff",
        ["ship_type", "scale", "zone_type", "timestep", "var", "max_diff"],
        list(_max_diff_rows(mesh, frames, ship_type, scale, zone_type)),
        batch_size=1000,
    )


def load_h5_to_postgresql(
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
    init_schema: bool = True,
    build_spatial: bool = True,
    connection: Optional[PostgreSQLConnectionArgs] = None,
) -> H5IngestPlan:
    """Parse one HDF5 result file and load the benchmark PostgreSQL schema."""
    connection = connection or PostgreSQLConnectionArgs.from_config()
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
    ship_type, scale = _parse_dataset_key(dataset_key)
    conn = connection.connect()
    try:
        if init_schema:
            from cfd_bench.ingest.postgresql.schema import apply_pg_schema
            apply_pg_schema(conn)
        cur = conn.cursor()
        try:
            _delete_zone_data(cur, ship_type, scale, zone_type)
            _insert_mesh(cur, mesh, ship_type, scale, zone_type)
            _insert_frames(cur, frames, ship_type, scale, zone_type)
            _insert_max_diffs(cur, mesh, frames, ship_type, scale, zone_type)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    finally:
        conn.close()

    if build_spatial:
        from cfd_bench.ingest.postgresql.build_cell_geom_full import build_cell_geom_full
        from cfd_bench.ingest.postgresql.build_point_locator_grid import build_point_locator_grid

        db_kw = dict(
            db_name=connection.db_name,
            db_user=connection.db_user,
            db_password=connection.db_password,
            db_host=connection.db_host,
            db_port=connection.db_port,
        )
        element_types_by_cell = {
            int(cid): str(element_type)
            for cid, element_type in zip(mesh.cell_ids, mesh.cell_element_types)
        }
        build_cell_geom_full(
            ship_type,
            scale,
            zone_type,
            element_types_by_cell=element_types_by_cell,
            **db_kw,
        )
        build_point_locator_grid(ship_type, scale, zone_type, **db_kw)

    return plan
