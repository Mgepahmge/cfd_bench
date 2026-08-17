"""HDF5 inspection and multi-backend ingest commands."""

from __future__ import annotations

import argparse
import json
from typing import Dict, Optional, Tuple

from cfd_bench.core.paths import resolve_max_range_dir, resolve_tiledb_root
from cfd_bench.infra.postgresql.config import PostgreSQLConfig
from cfd_bench.ingest.h5.artifacts import write_max_diff_files
from cfd_bench.ingest.h5.postgresql import (
    PostgreSQLConnectionArgs,
    build_ingest_plan,
    load_h5_to_postgresql,
)
from cfd_bench.ingest.h5.reader import OdbH5Reader


def _parse_mapping(items) -> Dict[str, Tuple[str, Optional[str]]]:
    """Parse TARGET=FIELD[.COMPONENT] mappings from CLI."""
    out: Dict[str, Tuple[str, Optional[str]]] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"invalid --map {item!r}; expected TARGET=FIELD[.COMPONENT]")
        target, source = item.split("=", 1)
        target = target.strip().upper()
        source = source.strip()
        if not target or not source:
            raise ValueError(f"invalid --map {item!r}")
        if "." in source:
            field, component = source.split(".", 1)
            out[target] = (field.strip(), component.strip() or None)
        else:
            out[target] = (source, None)
    return out



def _connection_args_from_cli(args: argparse.Namespace) -> PostgreSQLConnectionArgs:
    cfg = PostgreSQLConfig()
    return PostgreSQLConnectionArgs(
        db_name=args.db_name or cfg.db_name,
        db_user=args.db_user or cfg.db_user,
        db_password=args.db_password if args.db_password is not None else cfg.db_password,
        db_host=args.db_host or cfg.db_host,
        db_port=args.db_port or cfg.db_port,
    )

def add_h5_parsers(subparsers: argparse._SubParsersAction) -> None:
    inspect_ap = subparsers.add_parser(
        "inspect-h5",
        help="Inspect the mesh/step/frame/field structure of an ODB-like .h5 result file",
    )
    inspect_ap.add_argument("--h5", required=True, help="Path to the result .h5 file")
    inspect_ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    inspect_ap.set_defaults(func=run_inspect_h5)

    ingest_ap = subparsers.add_parser(
        "ingest-h5",
        help="Load an ODB-like .h5 result file into benchmark storage backends",
    )
    ingest_ap.add_argument("--h5", required=True, help="Path to the result .h5 file")
    ingest_ap.add_argument(
        "--datasets",
        required=True,
        metavar="DATASET",
        help="Dataset key used by benchmark storage backends and workloads",
    )
    ingest_ap.add_argument(
        "--backends",
        nargs="+",
        choices=["postgresql", "iotdb", "tiledb"],
        default=["postgresql"],
        help="H5 target backends (default: postgresql)",
    )
    ingest_ap.add_argument("--instance", default=None, help="Assembly instance (auto if only one)")
    ingest_ap.add_argument("--zone", default="0_Fluid", help="Target benchmark zone")
    ingest_ap.add_argument("--steps", nargs="+", default=None, help="Optional HDF5 Step names")
    ingest_ap.add_argument(
        "--vector-field",
        default=None,
        help="Override auto-selected 3-component source field mapped to U/V/W",
    )
    ingest_ap.add_argument(
        "--scalar-fields",
        nargs="+",
        default=None,
        help="Override scalar fields considered for automatic same-name mapping (default: P K E)",
    )
    ingest_ap.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="TARGET=FIELD[.COMPONENT]",
        help="Explicit benchmark field mapping; repeat as needed, e.g. --map P=S.S11",
    )
    ingest_ap.add_argument(
        "--timestep-mode",
        choices=["sequence", "frame-index", "inc-mode"],
        default="sequence",
        help="How HDF5 frames are represented in integer cell_scalar.timestep",
    )
    ingest_ap.add_argument(
        "--include-empty-frames",
        action="store_true",
        help="Keep frames with no mapped benchmark fields",
    )
    ingest_ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and convert in memory without opening a database",
    )
    ingest_ap.add_argument("--no-init-schema", action="store_true")
    ingest_ap.add_argument("--no-build-spatial", action="store_true")
    ingest_ap.add_argument("--max-range-dir", default=resolve_max_range_dir(), help="Legacy sidecar export directory; PostgreSQL W3 no longer depends on it")
    ingest_ap.add_argument("--no-max-diffs", action="store_true", help="Do not write W3 max_diffs CSV files")
    ingest_ap.add_argument("--db-name", default=None, help="Override CFD_BENCH_PG_DB_NAME")
    ingest_ap.add_argument("--db-user", default=None, help="Override CFD_BENCH_PG_USER")
    ingest_ap.add_argument("--db-password", default=None, help="Override CFD_BENCH_PG_PASSWORD")
    ingest_ap.add_argument("--db-host", default=None, help="Override CFD_BENCH_PG_HOST")
    ingest_ap.add_argument("--db-port", default=None, help="Override CFD_BENCH_PG_PORT")
    ingest_ap.add_argument("--iotdb-host", default=None, help="Override CFD_BENCH_IOTDB_HOST")
    ingest_ap.add_argument("--iotdb-port", default=None, help="Override CFD_BENCH_IOTDB_PORT")
    ingest_ap.add_argument("--iotdb-user", default=None, help="Override CFD_BENCH_IOTDB_USER")
    ingest_ap.add_argument("--iotdb-password", default=None, help="Override CFD_BENCH_IOTDB_PASSWORD")
    ingest_ap.add_argument("--iotdb-root-path", default=None, help="Override CFD_BENCH_IOTDB_ROOT_PATH")
    ingest_ap.add_argument("--tiledb-root", default=resolve_tiledb_root(), help="TileDB root directory")
    ingest_ap.set_defaults(func=run_ingest_h5)


def run_inspect_h5(args: argparse.Namespace) -> int:
    summary = OdbH5Reader(args.h5).inspect()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"H5: {summary['path']}")
    print("Parts:", ", ".join(summary["parts"]) or "<none>")
    print("Instances:")
    for instance, part in summary["instances"].items():
        print(f"  {instance} -> {part}")
    print("Frames:")
    for frame in summary["frames"]:
        fields = ", ".join(frame["fields"].keys()) or "<none>"
        print(
            f"  {frame['step']}/{frame['frame']} "
            f"inc/mode={frame['inc_or_mode']} time/freq={frame['time_or_frequency']}: {fields}"
        )
    return 0


def _print_plan(plan, frames) -> None:
    print(f"instance: {plan.instance_name} -> part {plan.part_name}")
    print(f"mesh: nodes={plan.node_count}, cells={plan.cell_count}, types={list(plan.element_types)}")
    print(f"frames: {plan.frame_count}, timesteps={list(plan.mapped_timesteps)}")
    print(f"variables: {list(plan.mapped_variables)}")
    print(f"nodal variables: {list(plan.mapped_node_variables)}")
    if plan.skipped_frames:
        print(f"skipped frames with no mapped data: {list(plan.skipped_frames)}")
    for frame in frames:
        mappings = ", ".join(
            f"{target}<-{source}" for target, source in sorted(frame.source_fields.items())
        )
        print(
            f"  t={frame.timestep}: {frame.info.step_name}/{frame.info.frame_name} "
            f"[{mappings or 'no mapped variables'}]"
        )


def run_ingest_h5(args: argparse.Namespace) -> int:
    dataset = args.datasets
    selected = list(dict.fromkeys(args.backends or ["postgresql"]))
    explicit = _parse_mapping(args.map)
    explicit_mapping = explicit or None
    if args.dry_run:
        plan, _, frames = build_ingest_plan(
            args.h5,
            instance_name=args.instance,
            step_names=args.steps,
            vector_field=args.vector_field,
            scalar_fields=args.scalar_fields,
            explicit_mapping=explicit_mapping,
            timestep_mode=args.timestep_mode,
            include_empty_frames=args.include_empty_frames,
        )
        print(f"dataset: {dataset}")
        print(f"target backends: {selected}")
        _print_plan(plan, frames)
        return 0

    # Build once for sidecar compatibility and a backend-independent summary.
    plan, mesh, frames = build_ingest_plan(
        args.h5,
        instance_name=args.instance,
        step_names=args.steps,
        vector_field=args.vector_field,
        scalar_fields=args.scalar_fields,
        explicit_mapping=explicit_mapping,
        timestep_mode=args.timestep_mode,
        include_empty_frames=args.include_empty_frames,
    )

    completed = []
    if "postgresql" in selected:
        connection = _connection_args_from_cli(args)
        # Keep the v5 PostgreSQL loader call and defaults unchanged.
        load_h5_to_postgresql(
            args.h5,
            dataset,
            instance_name=args.instance,
            zone_type=args.zone,
            step_names=args.steps,
            vector_field=args.vector_field,
            scalar_fields=args.scalar_fields,
            explicit_mapping=explicit_mapping,
            timestep_mode=args.timestep_mode,
            include_empty_frames=args.include_empty_frames,
            init_schema=not args.no_init_schema,
            build_spatial=not args.no_build_spatial,
            connection=connection,
        )
        completed.append("postgresql")

    if "iotdb" in selected:
        from cfd_bench.infra.iotdb.config import IoTDBConfig
        from cfd_bench.ingest.h5.iotdb import IoTDBConnectionArgs, load_h5_to_iotdb

        cfg = IoTDBConfig()
        connection = IoTDBConnectionArgs(
            host=args.iotdb_host or cfg.host,
            port=args.iotdb_port or cfg.port,
            user=args.iotdb_user or cfg.user,
            password=args.iotdb_password if args.iotdb_password is not None else cfg.password,
            root_path=args.iotdb_root_path or cfg.root_path,
        )
        load_h5_to_iotdb(
            args.h5,
            dataset,
            instance_name=args.instance,
            zone_type=args.zone,
            step_names=args.steps,
            vector_field=args.vector_field,
            scalar_fields=args.scalar_fields,
            explicit_mapping=explicit_mapping,
            timestep_mode=args.timestep_mode,
            include_empty_frames=args.include_empty_frames,
            connection=connection,
        )
        completed.append("iotdb")

    if "tiledb" in selected:
        from cfd_bench.ingest.h5.tiledb import load_h5_to_tiledb

        load_h5_to_tiledb(
            args.h5,
            dataset,
            root_path=args.tiledb_root,
            instance_name=args.instance,
            zone_type=args.zone,
            step_names=args.steps,
            vector_field=args.vector_field,
            scalar_fields=args.scalar_fields,
            explicit_mapping=explicit_mapping,
            timestep_mode=args.timestep_mode,
            include_empty_frames=args.include_empty_frames,
        )
        completed.append("tiledb")

    max_files = []
    if not args.no_max_diffs:
        max_files = list(write_max_diff_files(args.max_range_dir, dataset, mesh, frames))
    print(
        f"H5 ingest OK: dataset={dataset} zone={args.zone} backends={completed} "
        f"nodes={plan.node_count} cells={plan.cell_count} "
        f"frames={plan.frame_count} vars={list(plan.mapped_variables)} "
        f"nodal_vars={list(plan.mapped_node_variables)}"
    )
    if max_files:
        print(f"W3 legacy max_diffs sidecars: {len(max_files)} files -> {args.max_range_dir}")
    return 0

