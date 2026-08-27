"""CLI for inspecting the global AABB of an IoTDB CFD mesh."""

from __future__ import annotations

import argparse
import json
import math
from typing import Mapping, Optional, Sequence, Tuple

from cfd_bench.core.warnings_state import preserve_warning_filters


_BOUND_FIELDS = (
    "bbox_min_x",
    "bbox_max_x",
    "bbox_min_y",
    "bbox_max_y",
    "bbox_min_z",
    "bbox_max_z",
)


def add_bounds_parser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser(
        "bounds",
        help="Show the global IoTDB CFD mesh AABB and optionally classify points",
    )
    ap.add_argument(
        "--datasets",
        nargs=1,
        required=True,
        metavar="DATASET",
        help="Exactly one CFD dataset key (case-sensitive)",
    )
    ap.add_argument(
        "--zone",
        default=None,
        help="Override CFD mesh zone (default: dataset metadata zone)",
    )
    ap.add_argument(
        "--point",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "Z"),
        help="Coordinate to test against the global AABB; may be repeated",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the compact text view",
    )
    ap.set_defaults(func=run_bounds)


def _finite_bounds(meta: Mapping[str, object]) -> Tuple[float, float, float, float, float, float]:
    values = []
    for field in _BOUND_FIELDS:
        value = meta.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"CFD mesh metadata is missing a valid {field}") from None
        if not math.isfinite(number):
            raise ValueError(f"CFD mesh metadata contains non-finite {field}={value!r}")
        values.append(number)

    xmin, xmax, ymin, ymax, zmin, zmax = values
    if xmin > xmax or ymin > ymax or zmin > zmax:
        raise ValueError("CFD mesh metadata contains an invalid bounding box")
    return xmin, xmax, ymin, ymax, zmin, zmax


def _point_diagnostic(
    point: Sequence[float],
    bounds: Sequence[float],
    tolerance: float,
) -> dict:
    x, y, z = (float(v) for v in point)
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    coords = (x, y, z)
    lows = (xmin, ymin, zmin)
    highs = (xmax, ymax, zmax)
    labels = ("x", "y", "z")

    outside_axes = []
    delta = []
    for label, value, low, high in zip(labels, coords, lows, highs):
        if value < low - tolerance:
            outside_axes.append(f"{label}-")
            delta.append(low - value)
        elif value > high + tolerance:
            outside_axes.append(f"{label}+")
            delta.append(value - high)
        else:
            delta.append(0.0)

    distance = math.sqrt(sum(d * d for d in delta))
    return {
        "point": [x, y, z],
        "inside_aabb": not outside_axes,
        "outside_axes": outside_axes,
        "distance_to_aabb": float(distance),
    }


def _build_payload(
    dataset: str,
    zone: str,
    mesh_meta: Mapping[str, object],
    points: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    bounds = _finite_bounds(mesh_meta)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extent = [xmax - xmin, ymax - ymin, zmax - zmin]
    scale = max(max(extent), 1.0)
    # Match the global-AABB tolerance used by StructureCfdCouplingEngine.
    tolerance = max(1.0e-10 * scale, 1.0e-12)

    return {
        "dataset": str(dataset),
        "zone": str(zone),
        "bounds": {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        },
        "extent": {"x": extent[0], "y": extent[1], "z": extent[2]},
        "tolerance": tolerance,
        "points": [
            _point_diagnostic(point, bounds, tolerance)
            for point in (points or ())
        ],
    }


def _print_text(payload: Mapping[str, object]) -> None:
    bounds = payload["bounds"]
    extent = payload["extent"]
    print("CFD mesh global bounds (IoTDB)")
    print(f"  dataset={payload['dataset']} zone={payload['zone']}")
    print(
        f"  X: [{bounds['xmin']:.12g}, {bounds['xmax']:.12g}]  "
        f"extent={extent['x']:.12g}"
    )
    print(
        f"  Y: [{bounds['ymin']:.12g}, {bounds['ymax']:.12g}]  "
        f"extent={extent['y']:.12g}"
    )
    print(
        f"  Z: [{bounds['zmin']:.12g}, {bounds['zmax']:.12g}]  "
        f"extent={extent['z']:.12g}"
    )
    print(f"  tolerance={payload['tolerance']:.6e}")

    for index, result in enumerate(payload["points"], start=1):
        p = result["point"]
        if result["inside_aabb"]:
            status = "INSIDE_AABB"
        else:
            axes = ",".join(result["outside_axes"])
            status = f"OUTSIDE_AABB axes={axes} distance={result['distance_to_aabb']:.12g}"
        print(f"  point[{index}]=({p[0]:.12g}, {p[1]:.12g}, {p[2]:.12g})  {status}")

    if payload["points"]:
        print("  note=INSIDE_AABB only means inside the global box; it does not prove a containing CFD cell exists.")


def run_bounds(args: argparse.Namespace) -> int:
    with preserve_warning_filters():
        from cfd_bench.infra.iotdb.config import IoTDBConfig
        from cfd_bench.infra.iotdb.repository import IoTDBRepository

        dataset = str(args.datasets[0])
        repo = IoTDBRepository(IoTDBConfig())
        repo.open()
        try:
            dataset_meta = repo.cfd_dataset_metadata(dataset)
            if dataset_meta and not bool(dataset_meta.get("is_cfd")):
                raise ValueError(f"dataset={dataset!r} is not a CFD dataset in IoTDB")
            zone = str(args.zone or dataset_meta.get("zone") or "0_Fluid")
            mesh_meta = repo.fetch_mesh_meta(dataset, zone)
            if not mesh_meta:
                raise ValueError(
                    f"no CFD mesh metadata found for dataset={dataset!r} zone={zone!r}"
                )
            payload = _build_payload(dataset, zone, mesh_meta, args.point)
        finally:
            repo.close()

    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        _print_text(payload)
    return 0
