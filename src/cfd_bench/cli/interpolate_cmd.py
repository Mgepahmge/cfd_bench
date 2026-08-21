"""Standalone IoTDB fluid linear-interpolation command."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from cfd_bench.core.warnings_state import preserve_warning_filters


def add_interpolate_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser(
        "interpolate",
        help="Map IoTDB CFD fields to a target coordinate by linear interpolation",
    )
    ap.add_argument(
        "--datasets",
        nargs=1,
        required=True,
        metavar="DATASET",
        help="Exactly one CFD dataset key (case-sensitive)",
    )
    ap.add_argument("--step", type=int, required=True, help="CFD timestep/frame")
    ap.add_argument(
        "--point",
        type=float,
        nargs=3,
        action="append",
        required=True,
        metavar=("X", "Y", "Z"),
        help="Target coordinate; may be repeated to map several points in one IoTDB session",
    )
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Physical quantities to interpolate (default: all CFD variables)",
    )
    ap.add_argument(
        "--zone",
        default=None,
        help="Override CFD result zone (default: dataset metadata zone)",
    )
    ap.set_defaults(func=run_interpolate)


def _fmt_vector(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(x):.10g}" for x in values) + ")"


def _print_result(index: int, result) -> None:
    print(f"\n=== Fluid interpolation point {index} ===")
    print(f"dataset={result.dataset} step={result.step} zone={result.zone}")
    print(f"target={_fmt_vector(result.target)}")
    print(
        f"containing_cell={result.cell_id} "
        f"tecplot_element_id={result.source_element_id}"
    )
    print(f"cell_vertex_count={len(result.cell_node_ids)}")
    print(f"support_node_ids(dense)={list(result.support_node_ids)}")
    print(f"support_node_ids(tecplot)={list(result.support_source_node_ids)}")
    print(f"barycentric_weights={_fmt_vector(result.weights)}")
    print(f"coordinate_reconstruction_error={result.reconstruction_error:.6e}")
    print(f"vertex_value_source={result.vertex_value_source}")
    print("interpolated_values:")
    for var, value in result.values.items():
        source = result.support_vertex_values.get(var, ())
        print(f"  {var} = {float(value):.12g}    support={_fmt_vector(source)}")
    passed = (
        np.isfinite(result.reconstruction_error)
        and result.reconstruction_error <= 1.0e-7 * max(1.0, float(np.linalg.norm(result.target)))
        and np.all(np.isfinite(result.weights))
        and abs(float(np.sum(result.weights)) - 1.0) <= 1.0e-8
        and all(np.isfinite(float(v)) for v in result.values.values())
    )
    print(f"validation={'PASS' if passed else 'FAIL'}")


def run_interpolate(args: argparse.Namespace) -> int:
    # IoTDB's Python package changes the process-wide DeprecationWarning filter
    # when Session is imported.  Keep this standalone feature isolated just as
    # the benchmark discovery path does.
    with preserve_warning_filters():
        from cfd_bench.features.fluid_interpolation import FluidInterpolationEngine
        from cfd_bench.infra.iotdb.config import IoTDBConfig
        from cfd_bench.infra.iotdb.repository import IoTDBRepository

        dataset = str(args.datasets[0])
        repo = IoTDBRepository(IoTDBConfig())
        repo.open()
        try:
            engine = FluidInterpolationEngine(repo)
            print("Fluid linear interpolation mapping (Apache IoTDB)")
            print(f"  dataset={dataset} step={int(args.step)}")
            print(f"  points={len(args.point)} variables={args.variables or 'auto'} zone={args.zone or 'auto'}")
            for i, point in enumerate(args.point, start=1):
                result = engine.interpolate(
                    dataset,
                    int(args.step),
                    point,
                    variables=args.variables,
                    zone=args.zone,
                )
                _print_result(i, result)
        finally:
            repo.close()
    return 0
